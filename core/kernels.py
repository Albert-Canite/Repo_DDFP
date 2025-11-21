import numpy as np
from .config import rng

def generate_kernels(num_layers, K):
    kernels = [
        rng.uniform(-0.5,0.5,(1,1,K,K)).astype(np.float32)
        for _ in range(num_layers)
    ]
    return kernels
