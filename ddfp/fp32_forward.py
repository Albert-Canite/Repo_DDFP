import torch
import torch.nn.functional as F
import numpy as np
import core.config as C


def _group_norm(x: torch.Tensor, meta):
    if meta is None:
        return x
    groups = meta.get("groups", 1)
    eps = meta.get("eps", 1e-5)
    weight = torch.tensor(meta.get("weight"), dtype=x.dtype, device=x.device)
    bias = torch.tensor(meta.get("bias"), dtype=x.dtype, device=x.device)
    return F.group_norm(x, groups=groups, weight=weight, bias=bias, eps=eps)


def run_network_fp32(x_fp, kernels, gn_meta=None, apply_activation: bool = True):
    out = torch.tensor(x_fp, dtype=torch.float32)
    gn_meta = gn_meta or [None] * len(kernels)
    for idx, kernel in enumerate(kernels):
        stride = C.KERNEL_STRIDES[idx] if idx < len(C.KERNEL_STRIDES) else 1
        padding = C.KERNEL_PADDINGS[idx] if idx < len(C.KERNEL_PADDINGS) else 0
        xt = torch.tensor(out, dtype=torch.float32)
        wt = torch.tensor(kernel, dtype=torch.float32)
        y = F.conv2d(xt, wt, stride=stride, padding=padding)
        y = _group_norm(y, gn_meta[idx])
        if apply_activation and (gn_meta[idx] is None or gn_meta[idx].get("activation", True)):
            y = F.silu(y)
        out = y
    return out.detach().cpu().numpy()
