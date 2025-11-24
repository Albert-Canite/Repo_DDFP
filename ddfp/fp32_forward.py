import torch
import torch.nn.functional as F
import numpy as np
import core.config as C


def run_network_fp32(x_fp, kernels, apply_relu: bool = False):
    out = x_fp
    for idx, kernel in enumerate(kernels):
        stride = C.KERNEL_STRIDES[idx] if idx < len(C.KERNEL_STRIDES) else 1
        padding = C.KERNEL_PADDINGS[idx] if idx < len(C.KERNEL_PADDINGS) else 0
        xt = torch.tensor(out, dtype=torch.float32)
        wt = torch.tensor(kernel, dtype=torch.float32)
        y = F.conv2d(xt, wt, stride=stride, padding=padding).numpy()
        if apply_relu:
            y = np.maximum(y, 0.0)
        out = y
    return out
