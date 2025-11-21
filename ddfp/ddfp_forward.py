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


def calibrate_ddfp(cal_imgs):
    deltas, alphas, betas_scale, betas_gain, p_syms = [], [], [], [], []
    x_cur = [img.copy() for img in cal_imgs]
    target_util_full = C.ADC_EFF / C.BETA_MARGIN

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

        # MAC 分布
        mac_vals = []
        for x in x_cur:
            x_int = np.rint(x / delta).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)
            w_int = np.rint(kernel / alpha).clip(C.WEIGHT_MIN, C.WEIGHT_MAX).astype(np.int32)[0, 0]
            H, W = x_int.shape[2:]
            for i in range(H - C.KERNEL_SIZE + 1):
                for j in range(W - C.KERNEL_SIZE + 1):
                    patch = x_int[0, 0, i:i + C.KERNEL_SIZE, j:j + C.KERNEL_SIZE]
                    mac_vals.append(np.sum(patch * w_int))

        mac = np.array(mac_vals, dtype=np.float32)
        mac_pos, mac_neg = mac[mac > 0], -mac[mac < 0]
        ppos = np.percentile(mac_pos, C.BETA_PERCENTILE) if mac_pos.size > 0 else 0.0
        pneg = np.percentile(mac_neg, C.BETA_PERCENTILE) if mac_neg.size > 0 else 0.0
        p_sym = max(0.5 * (ppos + pneg), 1e-6)

        beta_scale = (p_sym * C.BETA_MARGIN) / (C.ADC_MAX * C.ADC_EFF)  # MAC/LSB

        # beta 对齐迭代
        for _ in range(C.BETA_ALIGN_ITERS):
            util_meas = []
            for idx_img, x in enumerate(x_cur, 1):
                x_int = np.rint(x / delta).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)
                w_int = np.rint(kernel / alpha).clip(C.WEIGHT_MIN, C.WEIGHT_MAX).astype(np.int32)[0, 0]

                H, W = x_int.shape[2:]
                mac_map = np.zeros((H - C.KERNEL_SIZE + 1, W - C.KERNEL_SIZE + 1), dtype=np.float32)
                for i in range(mac_map.shape[0]):
                    for j in range(mac_map.shape[1]):
                        patch = x_int[0, 0, i:i + C.KERNEL_SIZE, j:j + C.KERNEL_SIZE]
                        mac_map[i, j] = np.sum(patch * w_int)

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

        # 前向传播更新 x_cur
        x_next = []
        for x in x_cur:
            x_int = np.rint(x / delta).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)
            w_int = np.rint(kernel / alpha).clip(C.WEIGHT_MIN, C.WEIGHT_MAX).astype(np.int32)[0, 0]
            w_int = w_int * (1.0 + get_w_noise(layer_idx, w_int.shape))

            H, W = x_int.shape[2:]
            mac_map = np.zeros((H - C.KERNEL_SIZE + 1, W - C.KERNEL_SIZE + 1), dtype=np.float32)
            for i in range(mac_map.shape[0]):
                for j in range(mac_map.shape[1]):
                    patch = x_int[0, 0, i:i + C.KERNEL_SIZE, j:j + C.KERNEL_SIZE]
                    mac_map[i, j] = np.sum(patch * w_int)

            noise = get_noise_pack(layer_idx, mac_map.shape)
            out_adc = acim_hw(mac_map, betas_scale[-1], noise["gain"], noise["adc_noise"])
            out_fp = out_adc.astype(np.float32) * float(delta) * float(alpha) * float(betas_scale[-1])
            x_next.append(out_fp[np.newaxis, np.newaxis, :, :])

        x_cur = x_next

    C.DELTAS = deltas
    C.ALPHAS = alphas
    C.BETAS_SCALE = betas_scale
    C.BETAS_GAIN = betas_gain
    C.P_SYMS = p_syms

    return deltas, alphas, betas_scale, betas_gain, p_syms


def run_network_ddfp(x_fp):
    out = x_fp
    adc_usage_full = []
    adc_usage_eff = []
    per_layer_params = []

    for layer_idx, kernel in enumerate(C.kernels, 1):
        delta = C.DELTAS[layer_idx - 1]
        alpha = C.ALPHAS[layer_idx - 1]
        beta_scale = C.BETAS_SCALE[layer_idx - 1]
        beta_gain = C.BETAS_GAIN[layer_idx - 1]
        p_sym = C.P_SYMS[layer_idx - 1]

        per_layer_params.append((float(delta), float(alpha), float(beta_scale), float(beta_gain), float(p_sym)))

        x_int = np.rint(out / delta).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)
        w_int = np.rint(kernel / alpha).clip(C.WEIGHT_MIN, C.WEIGHT_MAX).astype(np.int32)[0, 0]
        w_int = w_int * (1.0 + get_w_noise(layer_idx, w_int.shape))

        H, W = x_int.shape[2:]
        mac = np.zeros((H - C.KERNEL_SIZE + 1, W - C.KERNEL_SIZE + 1), dtype=np.float32)
        for i in range(mac.shape[0]):
            for j in range(mac.shape[1]):
                patch = x_int[0, 0, i:i + C.KERNEL_SIZE, j:j + C.KERNEL_SIZE]
                mac[i, j] = np.sum(patch * w_int)

        noise = get_noise_pack(layer_idx, mac.shape)
        out_adc = acim_hw(mac, beta_scale, noise["gain"], noise["adc_noise"])
        u_full, u_eff, _ = _adc_utilization_stats(out_adc)
        adc_usage_full.append(u_full)
        adc_usage_eff.append(u_eff)

        out = out_adc.astype(np.float32) * float(delta) * float(alpha) * float(beta_scale)
        out = out[np.newaxis, np.newaxis, :, :]

    return out, adc_usage_full, adc_usage_eff, per_layer_params
