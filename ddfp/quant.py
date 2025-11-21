import numpy as np
import core.config as C
from core.utils import fp32_to_fp15


def weight_quant_mse(w, scale):
    q = np.rint(w / scale).clip(C.WEIGHT_MIN, C.WEIGHT_MAX)
    return np.mean((q * scale - w) ** 2)


def search_alpha_min_mse(kernel):
    w = kernel[0, 0].copy()
    w_absmax = np.max(np.abs(w)) + 1e-12

    alpha_min = w_absmax / max(1, C.WEIGHT_MAX)
    grid = np.geomspace(0.5, 1.5, 41) * alpha_min

    best_alpha = alpha_min
    best_mse = weight_quant_mse(w, alpha_min)

    for a in grid:
        mse = weight_quant_mse(w, a)
        if mse < best_mse:
            best_alpha = a
            best_mse = mse

    return fp32_to_fp15(best_alpha)
