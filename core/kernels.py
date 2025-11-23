import numpy as np
from .config import rng, CONV_CHANNELS


def generate_kernels(num_layers, K, in_channels=1, out_channels=None):
    out_channels = CONV_CHANNELS if out_channels is None else out_channels
    kernels = []
    for layer in range(num_layers):
        oc = out_channels
        ic = in_channels if layer == 0 else out_channels
        kernels.append(rng.uniform(-0.5, 0.5, (oc, ic, K, K)).astype(np.float32))
    return kernels
