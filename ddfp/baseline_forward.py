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


def run_network_baseline(x_fp):
    x_int = np.rint(x_fp / C.DELTA_BASELINE).clip(C.INPUT_MIN, C.INPUT_MAX).astype(np.int32)

    adc_usage_full = []
    adc_usage_eff = []

    for layer_idx, kernel in enumerate(C.kernels, 1):
        w_int = np.rint(kernel / C.WEIGHT_SCALE_BASELINE).clip(C.WEIGHT_MIN, C.WEIGHT_MAX).astype(np.int32)[0, 0]
        w_int = w_int * (1.0 + get_w_noise(layer_idx, w_int.shape))

        H, W = x_int.shape[2:]
        out_int = np.zeros((H - C.KERNEL_SIZE + 1, W - C.KERNEL_SIZE + 1), dtype=np.float32)

        for i in range(out_int.shape[0]):
            for j in range(out_int.shape[1]):
                patch = x_int[0, 0, i:i + C.KERNEL_SIZE, j:j + C.KERNEL_SIZE]
                out_int[i, j] = np.sum(patch * w_int)

        noise = get_noise_pack(layer_idx, out_int.shape)
        out_adc = acim_hw(out_int, C.BASELINE_ADC_SCALE, noise["gain"], noise["adc_noise"])

        u_full, u_eff, _ = _adc_utilization_stats(out_adc)
        adc_usage_full.append(u_full)
        adc_usage_eff.append(u_eff)

        x_int = out_adc[np.newaxis, np.newaxis, :, :]

    composite_scale = C.DELTA_BASELINE * ((C.WEIGHT_SCALE_BASELINE * C.BASELINE_ADC_SCALE) ** C.NUM_LAYERS)
    out_fp = x_int.astype(np.float32) * composite_scale

    return out_fp, adc_usage_full, adc_usage_eff
