# ddfp/noise.py
import numpy as np
import core.config as C


def get_noise_pack(layer_idx, shape):
    key = (layer_idx, shape)
    if key not in C._noise_bank:
        gain = C.rng.normal(1.0, C.GAIN_SIGMA)
        adc_noise = C.rng.normal(0.0, C.ADC_NOISE_STD, size=shape)
        C._noise_bank[key] = dict(gain=gain, adc_noise=adc_noise)
    return C._noise_bank[key]


def get_w_noise(layer_idx, w_shape):
    key = (layer_idx, w_shape)
    if key not in C._w_noise_bank:
        C._w_noise_bank[key] = C.rng.normal(0.0, C.WEIGHT_NOISE_STD, size=w_shape)
    return C._w_noise_bank[key]
