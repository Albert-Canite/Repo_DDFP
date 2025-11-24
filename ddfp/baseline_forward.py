# ddfp/baseline_forward.py
import numpy as np
import core.config as C
from ddfp.noise import get_noise_pack, get_w_noise


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
        out_fp = out_fp / (1 + np.exp(-out_fp))  # SiLU approximation
    return out_fp


def run_network_baseline(x_fp, gn_meta=None):
    x_int = np.rint(x_fp / C.DELTA_BASELINE).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)

    adc_usage_full = []
    adc_usage_eff = []
    gn_meta = gn_meta or [None] * len(C.kernels)

    out_fp = None
    for layer_idx, kernel in enumerate(C.kernels, 1):
        w_int = np.rint(kernel / C.WEIGHT_SCALE_BASELINE).clip(C.WEIGHT_MIN, C.WEIGHT_MAX).astype(np.int32)
        w_int = w_int * (1.0 + get_w_noise(layer_idx, w_int.shape))

        stride = C.KERNEL_STRIDES[layer_idx - 1] if layer_idx - 1 < len(C.KERNEL_STRIDES) else 1
        padding = C.KERNEL_PADDINGS[layer_idx - 1] if layer_idx - 1 < len(C.KERNEL_PADDINGS) else 0
        out_int = _conv2d_int(x_int, w_int, stride=stride, padding=padding)

        noise = get_noise_pack(layer_idx, out_int.shape)
        out_adc = acim_hw(out_int, C.BASELINE_ADC_SCALE, noise["gain"], noise["adc_noise"])

        u_full, u_eff, _ = _adc_utilization_stats(out_adc)
        adc_usage_full.append(u_full)
        adc_usage_eff.append(u_eff)

        out_fp = out_adc.astype(np.float32) * float(C.DELTA_BASELINE) * float(C.WEIGHT_SCALE_BASELINE) * float(C.BASELINE_ADC_SCALE)
        out_fp = _apply_gn_activation(out_fp[np.newaxis, :, :, :], gn_meta[layer_idx - 1])
        x_int = np.rint(out_fp / C.DELTA_BASELINE).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)

    return out_fp, adc_usage_full, adc_usage_eff
