import torch
import torch.nn.functional as F
import numpy as np


def run_network_fp32(x_fp, kernels, apply_relu: bool = False):
    out = x_fp  # [1,1,H,W]
    for kernel in kernels:
        xt = torch.tensor(out, dtype=torch.float32)
        wt = torch.tensor(kernel, dtype=torch.float32)
        y = F.conv2d(xt, wt).numpy()   # shape = [1,1,H',W']
        if apply_relu:
            y = np.maximum(y, 0.0)
        out = y
    return out  # [1,1,H',W']
