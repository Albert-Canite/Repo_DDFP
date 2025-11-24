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


def run_network_baseline(x_fp, apply_relu: bool = False):
    x_int = np.rint(x_fp / C.DELTA_BASELINE).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)

    adc_usage_full = []
    adc_usage_eff = []

    for layer_idx, kernel in enumerate(C.kernels, 1):
        w_int = np.rint(kernel / C.WEIGHT_SCALE_BASELINE).clip(C.WEIGHT_MIN, C.WEIGHT_MAX).astype(np.int32)
        w_int = w_int * (1.0 + get_w_noise(layer_idx, w_int.shape))

        stride = C.KERNEL_STRIDES[layer_idx - 1] if layer_idx - 1 < len(C.KERNEL_STRIDES) else 1
        padding = C.KERNEL_PADDINGS[layer_idx - 1] if layer_idx - 1 < len(C.KERNEL_PADDINGS) else 0
        out_int = _conv2d_int(x_int, w_int, stride=stride, padding=padding)

        noise = get_noise_pack(layer_idx, out_int.shape)
        out_adc = acim_hw(out_int, C.BASELINE_ADC_SCALE, noise["gain"], noise["adc_noise"])
        if apply_relu:
            out_adc = np.maximum(out_adc, 0)

        u_full, u_eff, _ = _adc_utilization_stats(out_adc)
        adc_usage_full.append(u_full)
        adc_usage_eff.append(u_eff)

        x_int = out_adc[np.newaxis, :, :, :]

    composite_scale = C.DELTA_BASELINE * ((C.WEIGHT_SCALE_BASELINE * C.BASELINE_ADC_SCALE) ** C.NUM_LAYERS)
    out_fp = x_int.astype(np.float32) * composite_scale
    if apply_relu:
        out_fp = np.maximum(out_fp, 0.0)

    return out_fp, adc_usage_full, adc_usage_eff
