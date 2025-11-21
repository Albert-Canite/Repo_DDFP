import numpy as np
import torch
import os
from pathlib import Path

YOLO_ROOT = Path(__file__).resolve().parents[1]
RSNA_DIR = YOLO_ROOT / "rsna-pneumonia-detection-challenge"
RSNA_TRAIN_IMG_DIR = RSNA_DIR / "stage_2_train_images"


IMAGE_SIZE = 512
INPUT_SIGNED = True
NUM_CALIBRATION = 50
NUM_TEST = 2
KERNEL_SIZE = 4

NUM_LAYERS_LIST = [1]
INPUT_BITS_LIST = [5]
WEIGHT_BITS_LIST = [5]
ADC_BITS_LIST = [8]

WEIGHT_SIGNED = True
ADC_SIGNED = True

GAIN_SIGMA = 0.10
ADC_EFF = 0.80
ADC_NOISE_STD = 1.0
WEIGHT_NOISE_STD = 0.05

BETA_PERCENTILE = 99.5
BETA_MARGIN = 1.1
DELTA_PERCENTILE = 98
DELTA_MARGIN = 1.0
BETA_ALIGN_ITERS = 3
BETA_ALIGN_CLIP = (0.85, 1.15)

OUTPUT_DIR = YOLO_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SEED = 2025
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


NUM_LAYERS = None
INPUT_BITS = None
WEIGHT_BITS = None
ADC_BITS = None

INPUT_MAX = None
INPUT_MIN = None
WEIGHT_MAX = None
WEIGHT_MIN = None
ADC_MAX = None
ADC_MIN = None

DELTA_BASELINE = None
WEIGHT_SCALE_BASELINE = None
BASELINE_ADC_SCALE = None

kernels = []
_noise_bank = {}
_w_noise_bank = {}

DELTAS = []
ALPHAS = []
BETAS_SCALE = []
BETAS_GAIN = []
P_SYMS = []


def setup_config(num_layers, input_bits, weight_bits, adc_bits):
    """
    初始化 bit 数、量化范围、baseline scale、kernel 随机初始化等。
    """

    global NUM_LAYERS, INPUT_BITS, WEIGHT_BITS, ADC_BITS
    global INPUT_MAX, INPUT_MIN, WEIGHT_MAX, WEIGHT_MIN, ADC_MAX, ADC_MIN
    global DELTA_BASELINE, WEIGHT_SCALE_BASELINE, BASELINE_ADC_SCALE
    global kernels, _noise_bank, _w_noise_bank

    from core.utils import fp32_to_fp15
    from core.kernels import generate_kernels

    NUM_LAYERS = int(num_layers)
    INPUT_BITS = int(input_bits)
    WEIGHT_BITS = int(weight_bits)
    ADC_BITS = int(adc_bits)

    # 动态范围
    INPUT_MAX = 2**(INPUT_BITS-1)-1 if INPUT_SIGNED else 2**INPUT_BITS-1
    INPUT_MIN = -2**(INPUT_BITS-1) if INPUT_SIGNED else 0

    WEIGHT_MAX = 2**(WEIGHT_BITS-1)-1 if WEIGHT_SIGNED else 2**WEIGHT_BITS-1
    WEIGHT_MIN = -2**(WEIGHT_BITS-1) if WEIGHT_SIGNED else 0

    ADC_MAX = 2**(ADC_BITS-1)-1 if ADC_SIGNED else 2**ADC_BITS-1
    ADC_MIN = -2**(ADC_BITS-1) if ADC_SIGNED else 0

    # baseline scales
    BASELINE_ABSMAX = 0.5 if INPUT_SIGNED else 1.0
    DELTA_BASELINE_FP32 = BASELINE_ABSMAX / INPUT_MAX
    DELTA_BASELINE = fp32_to_fp15(DELTA_BASELINE_FP32)

    WEIGHT_SCALE_BASELINE_FP32 = 0.5 / WEIGHT_MAX
    WEIGHT_SCALE_BASELINE = fp32_to_fp15(WEIGHT_SCALE_BASELINE_FP32)

    THEORETICAL_MAC_MAX = INPUT_MAX * WEIGHT_MAX * (KERNEL_SIZE**2)
    BASELINE_ADC_SCALE_FP32 = THEORETICAL_MAC_MAX / (ADC_MAX * ADC_EFF)
    BASELINE_ADC_SCALE = fp32_to_fp15(BASELINE_ADC_SCALE_FP32)

    # 随机初始化 kernels
    kernels = generate_kernels(NUM_LAYERS, KERNEL_SIZE)

    # 清空噪声库
    _noise_bank = {}
    _w_noise_bank = {}

    print(f"\n===== Setup Done: L={NUM_LAYERS}, IN={INPUT_BITS}, W={WEIGHT_BITS}, ADC={ADC_BITS} =====")
    print(f"RSNA DIR = {RSNA_DIR}")
    print(f"ImageSize = {IMAGE_SIZE}")
