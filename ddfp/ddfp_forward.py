# ddfp/ddfp_forward.py
import numpy as np
import core.config as C
from ddfp.noise import get_noise_pack, get_w_noise
from ddfp.quant import search_alpha_min_mse
from core.utils import fp32_to_fp15


def _adc_utilization_stats(out_adc):
    abs_out = np.abs(out_adc.astype(np.int32))
    abs_out = np.minimum(abs_out, C.ADC_MAX)
    maxabs = float(np.max(abs_out))
    full = maxabs / C.ADC_MAX
    eff = maxabs / (C.ADC_MAX * C.ADC_EFF)
    sat = float(np.mean((out_adc == C.ADC_MAX) | (out_adc == C.ADC_MIN)))
    return full, eff, sat


def acim_hw(mac_output, adc_scale, gain, adc_noise):
    mac_noisy = mac_output * gain
    adc_pre = mac_noisy / adc_scale + adc_noise
    adc_q = np.rint(adc_pre).clip(C.ADC_MIN, C.ADC_MAX)
    return adc_q.astype(np.int32)


def _conv2d_int(x_int, w_int, stride: int, padding: int):
    K = w_int.shape[2]
    if padding > 0:
        x_int = np.pad(x_int, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    H, W = x_int.shape[2:]
    out_h = (H - K) // stride + 1
    out_w = (W - K) // stride + 1
    out_int = np.zeros((w_int.shape[0], out_h, out_w), dtype=np.float32)
    for oc in range(w_int.shape[0]):
        for ic in range(w_int.shape[1]):
            for oh in range(out_h):
                hs = oh * stride
                for ow in range(out_w):
                    ws = ow * stride
                    patch = x_int[0, ic, hs : hs + K, ws : ws + K]
                    out_int[oc, oh, ow] += np.sum(patch * w_int[oc, ic])
    return out_int


def _apply_gn_activation(out_fp, meta):
    if meta is None:
        return out_fp
    groups = meta.get("groups", 1)
    eps = meta.get("eps", 1e-5)
    weight = meta.get("weight")
    bias = meta.get("bias")
    if weight is not None and bias is not None:
        N, C, H, W = out_fp.shape
        x = out_fp.reshape(N, groups, C // groups, H, W)
        mean = x.mean(axis=(2, 3, 4), keepdims=True)
        var = x.var(axis=(2, 3, 4), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        out_fp = x_norm.reshape(N, C, H, W)
        out_fp = out_fp * weight.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)
    if meta.get("activation", True):
        out_fp = np.maximum(out_fp, 0) + 0.0
        out_fp = out_fp / (1 + np.exp(-out_fp))
    return out_fp


def calibrate_ddfp(cal_imgs, gn_meta=None):
    deltas, alphas, betas_scale, betas_gain, p_syms = [], [], [], [], []
    x_cur = [img.copy() for img in cal_imgs]
    target_util_full = C.ADC_EFF / C.BETA_MARGIN
    gn_meta = gn_meta or [None] * len(C.kernels)

    for layer_idx, kernel in enumerate(C.kernels, 1):
        print(f"[Calib L{layer_idx}/{C.NUM_LAYERS}] start")

        all_vals = np.concatenate([np.abs(x).flatten() for x in x_cur])
        if all_vals.size == 0:
            p_in = 1.0
        else:
            p_in = np.percentile(all_vals, C.DELTA_PERCENTILE)

        delta_fp32 = (p_in * C.DELTA_MARGIN) / C.INPUT_MAX
        delta_floor = max(1e-8, float(p_in) * 1e-3 / max(1, C.INPUT_MAX))
        delta_fp32 = max(delta_fp32, delta_floor)
        delta = fp32_to_fp15(delta_fp32)

        alpha = search_alpha_min_mse(kernel)

        # MAC distribution for percentile estimation
        mac_vals = []
        stride = C.KERNEL_STRIDES[layer_idx - 1] if layer_idx - 1 < len(C.KERNEL_STRIDES) else 1
        padding = C.KERNEL_PADDINGS[layer_idx - 1] if layer_idx - 1 < len(C.KERNEL_PADDINGS) else 0
        for x in x_cur:
            x_int = np.rint(x / delta).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)
            w_int = np.rint(kernel / alpha).clip(C.WEIGHT_MIN, C.WEIGHT_MAX).astype(np.int32)
            mac_map = _conv2d_int(x_int, w_int, stride=stride, padding=padding)
            mac_vals.append(mac_map.flatten())

        mac = np.concatenate(mac_vals).astype(np.float32)
        mac_pos, mac_neg = mac[mac > 0], -mac[mac < 0]
        ppos = np.percentile(mac_pos, C.BETA_PERCENTILE) if mac_pos.size > 0 else 0.0
        pneg = np.percentile(mac_neg, C.BETA_PERCENTILE) if mac_neg.size > 0 else 0.0
        p_sym = max(0.5 * (ppos + pneg), 1e-6)

        beta_scale = (p_sym * C.BETA_MARGIN) / (C.ADC_MAX * C.ADC_EFF)  # MAC/LSB

        # Beta alignment iterations
        for _ in range(C.BETA_ALIGN_ITERS):
            util_meas = []
            for idx_img, x in enumerate(x_cur, 1):
                x_int = np.rint(x / delta).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)
                w_int = np.rint(kernel / alpha).clip(C.WEIGHT_MIN, C.WEIGHT_MAX).astype(np.int32)

                mac_map = _conv2d_int(x_int, w_int, stride=stride, padding=padding)

                noise = get_noise_pack(layer_idx, mac_map.shape)
                out_adc = acim_hw(mac_map, beta_scale, noise["gain"], noise["adc_noise"])
                u_full, _, sat = _adc_utilization_stats(out_adc)
                util_meas.append(u_full)

                if idx_img % 20 == 0 or idx_img == len(x_cur):
                    print(
                        f"[Calib L{layer_idx}/{C.NUM_LAYERS} {idx_img:3d}/{len(x_cur)}] "
                        f"δ={delta:.6e} α={alpha:.6e} β={beta_scale:.3f} "
                        f"util={u_full * 100:.2f}% sat={sat * 100:.2f}%"
                    )

            u_full_avg = float(np.mean(util_meas)) if len(util_meas) else 0.0
            if u_full_avg <= 0:
                break
            corr = np.clip(
                target_util_full / u_full_avg,
                C.BETA_ALIGN_CLIP[0],
                C.BETA_ALIGN_CLIP[1],
            )
            beta_scale *= corr

        beta_gain = 1.0 / beta_scale

        deltas.append(float(delta))
        alphas.append(float(alpha))
        betas_scale.append(float(fp32_to_fp15(beta_scale)))
        betas_gain.append(float(beta_gain))
        p_syms.append(float(p_sym))

        # Forward propagate to update x_cur
        x_next = []
        for x in x_cur:
            x_int = np.rint(x / delta).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)
            w_int = np.rint(kernel / alpha).clip(C.WEIGHT_MIN, C.WEIGHT_MAX).astype(np.int32)
            w_int = w_int * (1.0 + get_w_noise(layer_idx, w_int.shape))

            mac_map = _conv2d_int(x_int, w_int, stride=stride, padding=padding)

            noise = get_noise_pack(layer_idx, mac_map.shape)
            out_adc = acim_hw(mac_map, betas_scale[-1], noise["gain"], noise["adc_noise"])
            out_fp = out_adc.astype(np.float32) * float(delta) * float(alpha) * float(betas_scale[-1])
            out_fp = _apply_gn_activation(out_fp[np.newaxis, :, :, :], gn_meta[layer_idx - 1])
            x_next.append(out_fp)

        x_cur = x_next

    C.DELTAS = deltas
    C.ALPHAS = alphas
    C.BETAS_SCALE = betas_scale
    C.BETAS_GAIN = betas_gain
    C.P_SYMS = p_syms

    return deltas, alphas, betas_scale, betas_gain, p_syms


def run_network_ddfp(x_fp, gn_meta=None):
    out = x_fp
    adc_usage_full = []
    adc_usage_eff = []
    per_layer_params = []
    gn_meta = gn_meta or [None] * len(C.kernels)

    for layer_idx, kernel in enumerate(C.kernels, 1):
        delta = C.DELTAS[layer_idx - 1]
        alpha = C.ALPHAS[layer_idx - 1]
        beta_scale = C.BETAS_SCALE[layer_idx - 1]
        beta_gain = C.BETAS_GAIN[layer_idx - 1]
        p_sym = C.P_SYMS[layer_idx - 1]

        per_layer_params.append((float(delta), float(alpha), float(beta_scale), float(beta_gain), float(p_sym)))

        x_int = np.rint(out / delta).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)
        w_int = np.rint(kernel / alpha).clip(C.WEIGHT_MIN, C.WEIGHT_MAX).astype(np.int32)
        w_int = w_int * (1.0 + get_w_noise(layer_idx, w_int.shape))

        stride = C.KERNEL_STRIDES[layer_idx - 1] if layer_idx - 1 < len(C.KERNEL_STRIDES) else 1
        padding = C.KERNEL_PADDINGS[layer_idx - 1] if layer_idx - 1 < len(C.KERNEL_PADDINGS) else 0
        mac = _conv2d_int(x_int, w_int, stride=stride, padding=padding)

        noise = get_noise_pack(layer_idx, mac.shape)
        out_adc = acim_hw(mac, beta_scale, noise["gain"], noise["adc_noise"])
        u_full, u_eff, _ = _adc_utilization_stats(out_adc)
        adc_usage_full.append(u_full)
        adc_usage_eff.append(u_eff)

        out_fp = out_adc.astype(np.float32) * float(delta) * float(alpha) * float(beta_scale)
        out_fp = _apply_gn_activation(out_fp[np.newaxis, :, :, :], gn_meta[layer_idx - 1])
        out = out_fp

    return out, adc_usage_full, adc_usage_eff, per_layer_params
