import os
from pathlib import Path
import numpy as np
import torch

# Root and output paths
YOLO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = YOLO_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset: BCCD (Blood Cell Detection Dataset)
_bccd_env = os.environ.get("BCCD_DIR")
_bccd_default = Path(r"E:\\OneDrive - KAUST\\ONN codes\\yolo\\archive")
BCCD_DIR = Path(_bccd_env) if _bccd_env else (_bccd_default if _bccd_default.exists() else YOLO_ROOT / "archive")
BCCD_IMG_DIR = BCCD_DIR / "images"
BCCD_ANNO_DIR = BCCD_DIR / "annotations"
BCCD_ANNO_CSV = BCCD_DIR / "annotations.csv"

# Image & loader settings
IMAGE_SIZE = 512
INPUT_SIGNED = True
NUM_CALIBRATION = 20
NUM_TEST = 2
KERNEL_SIZE = 3

CONV_CHANNELS = 32

TASK_TYPE = "simple"

NUM_LAYERS_LIST = [1]
INPUT_BITS_LIST = [5]
WEIGHT_BITS_LIST = [5]
ADC_BITS_LIST = [8]

# BCCD detection
BCCD_CLASSES = ["RBC", "WBC", "Platelets"]
BCCD_ANCHORS = [
    (12, 12),
    (24, 24),
    (36, 36),
]
BCCD_BATCH_SIZE = 4
BCCD_NUM_WORKERS = 2
BCCD_LR = 2e-3
BCCD_MIN_LR = 2e-5
BCCD_WEIGHT_DECAY = 1e-4
BCCD_WARMUP_EPOCHS = 8
BCCD_EPOCHS = 80
BCCD_LOG_INTERVAL = 20
BCCD_CKPT = OUTPUT_DIR / "bccd_yolotiny.ckpt"
BCCD_TRAIN_CURVE = OUTPUT_DIR / "bccd_training.png"
BCCD_BEST_CKPT = OUTPUT_DIR / "bccd_yolotiny_best.ckpt"
BCCD_VIS_DIR = OUTPUT_DIR / "bccd_vis"
BCCD_QUANT_FIG = OUTPUT_DIR / "bccd_quant_compare.png"
BCCD_FP32_TRACKING = OUTPUT_DIR / "bccd_fp32_tracking.csv"
BCCD_BASELINE_TRACKING = OUTPUT_DIR / "bccd_baseline_tracking.csv"
BCCD_DDFP_TRACKING = OUTPUT_DIR / "bccd_ddfp_tracking.csv"
BCCD_CLS_LOSS_WEIGHT = 0.5
BCCD_OBJ_LOSS_WEIGHT = 1.0
BCCD_BOX_LOSS_WEIGHT = 5.0
BCCD_IOU_LOSS_WEIGHT = 2.0
BCCD_GN_GROUPS = 8
BCCD_GRID_SIZE = 16
BCCD_MAX_BOXES = 50
BCCD_SCORE_THRESH = 0.25
BCCD_NMS_IOU = 0.5
BCCD_WARMUP_ITERS = 100
BCCD_MOMENTUM = 0.9
BCCD_WEIGHT_STANDARDIZE = True
BCCD_COLOR_JITTER = 0.1
BCCD_HFLIP_PROB = 0.5
BCCD_VAL_SPLIT = 0.15
BCCD_TEST_SPLIT = 0.05
BCCD_SEED = 2025
BCCD_PLOTS_SAMPLES = 4
BCCD_METRIC_IMG = OUTPUT_DIR / "bccd_metric_curve.png"
BCCD_ERROR_IMG = OUTPUT_DIR / "bccd_error_curve.png"
BCCD_COMPARISON_IMG = OUTPUT_DIR / "bccd_fp32_baseline_ddfp.png"

# Legacy RSNA settings retained for backward compatibility with prior experiments
_rsna_env = os.environ.get("RSNA_DIR")
_rsna_default = Path(r"E:\\OneDrive - KAUST\\ONN codes\\yolo\\rsna-pneumonia-detection-challenge")
RSNA_DIR = Path(_rsna_env) if _rsna_env else (_rsna_default if _rsna_default.exists() else YOLO_ROOT / "rsna-pneumonia-detection-challenge")
RSNA_TRAIN_IMG_DIR = RSNA_DIR / "stage_2_train_images"
RSNA_TEST_IMG_DIR = RSNA_DIR / "stage_2_test_images"
RSNA_LABEL_CSV = RSNA_DIR / "stage_2_train_labels.csv"
RSNA_TRAIN_SAMPLES = 2000
RSNA_VAL_SAMPLES = 400
RSNA_BATCH_SIZE = 4
RSNA_NUM_WORKERS = 2
RSNA_LOG_INTERVAL = 20
RSNA_EPOCHS = 80
RSNA_LR = 2e-3
RSNA_MIN_LR = 2e-5
RSNA_WARMUP_EPOCHS = 8
RSNA_WEIGHT_DECAY = 1e-4
RSNA_MODEL_CHANNELS = 1
REGRESSION_HEAD_HIDDEN1 = 128
REGRESSION_HEAD_HIDDEN2 = 64
BBOX_IOU_LOSS_WEIGHT = 2.0
REGRESSION_SIZE_PRIOR = 0.25
RSNA_SPATIAL_POOL = 8
RSNA_DEBUG_VIS_SAMPLES = 4
REGRESSION_OUTPUT_IMG = OUTPUT_DIR / "rsna_regression_comparison.png"
REGRESSION_FP32_IMG = OUTPUT_DIR / "rsna_regression_fp32.png"
REGRESSION_FP32_TRACKING = OUTPUT_DIR / "rsna_regression_fp32_tracking.csv"
REGRESSION_TRAIN_CURVE = OUTPUT_DIR / "rsna_regression_training.png"
REGRESSION_CKPT = OUTPUT_DIR / "rsna_regression.ckpt"

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
KERNEL_STRIDES = []
KERNEL_PADDINGS = []
KERNEL_GN = []
_noise_bank = {}
_w_noise_bank = {}

DELTAS = []
ALPHAS = []
BETAS_SCALE = []
BETAS_GAIN = []
P_SYMS = []


def setup_config(num_layers, input_bits, weight_bits, adc_bits):
    """
    Initialize bit widths, quantization ranges, baseline scale, and kernel initialization.
    """

    global NUM_LAYERS, INPUT_BITS, WEIGHT_BITS, ADC_BITS
    global INPUT_MAX, INPUT_MIN, WEIGHT_MAX, WEIGHT_MIN, ADC_MAX, ADC_MIN
    global DELTA_BASELINE, WEIGHT_SCALE_BASELINE, BASELINE_ADC_SCALE
    global kernels, KERNEL_STRIDES, KERNEL_PADDINGS, KERNEL_GN, _noise_bank, _w_noise_bank

    from core.utils import fp32_to_fp15
    from core.kernels import generate_kernels

    NUM_LAYERS = int(num_layers)
    INPUT_BITS = int(input_bits)
    WEIGHT_BITS = int(weight_bits)
    ADC_BITS = int(adc_bits)

    INPUT_MAX = 2 ** (INPUT_BITS - 1) - 1 if INPUT_SIGNED else 2 ** INPUT_BITS - 1
    INPUT_MIN = -2 ** (INPUT_BITS - 1) if INPUT_SIGNED else 0

    WEIGHT_MAX = 2 ** (WEIGHT_BITS - 1) - 1 if WEIGHT_SIGNED else 2 ** WEIGHT_BITS - 1
    WEIGHT_MIN = -2 ** (WEIGHT_BITS - 1) if WEIGHT_SIGNED else 0

    ADC_MAX = 2 ** (ADC_BITS - 1) - 1 if ADC_SIGNED else 2 ** ADC_BITS - 1
    ADC_MIN = -2 ** (ADC_BITS - 1) if ADC_SIGNED else 0

    BASELINE_ABSMAX = 0.5 if INPUT_SIGNED else 1.0
    DELTA_BASELINE_FP32 = BASELINE_ABSMAX / INPUT_MAX
    DELTA_BASELINE = fp32_to_fp15(DELTA_BASELINE_FP32)

    WEIGHT_SCALE_BASELINE_FP32 = 0.5 / WEIGHT_MAX
    WEIGHT_SCALE_BASELINE = fp32_to_fp15(WEIGHT_SCALE_BASELINE_FP32)

    THEORETICAL_MAC_MAX = INPUT_MAX * WEIGHT_MAX * (KERNEL_SIZE ** 2)
    BASELINE_ADC_SCALE_FP32 = THEORETICAL_MAC_MAX / (ADC_MAX * ADC_EFF)
    BASELINE_ADC_SCALE = fp32_to_fp15(BASELINE_ADC_SCALE_FP32)

    kernels = generate_kernels(NUM_LAYERS, KERNEL_SIZE)
    KERNEL_STRIDES = [1] * NUM_LAYERS
    KERNEL_PADDINGS = [0] * NUM_LAYERS
    KERNEL_GN = [None] * NUM_LAYERS

    _noise_bank = {}
    _w_noise_bank = {}

    print(f"\n===== Setup Done: L={NUM_LAYERS}, IN={INPUT_BITS}, W={WEIGHT_BITS}, ADC={ADC_BITS} =====")
    print(f"BCCD DIR = {BCCD_DIR}")
    print(f"ImageSize = {IMAGE_SIZE}")


def set_kernels(custom_kernels):
    """Override default random kernels (used for regression model conv weights)."""
    global kernels, KERNEL_STRIDES, KERNEL_PADDINGS, _noise_bank, _w_noise_bank
    kernels = custom_kernels
    KERNEL_STRIDES = [1] * len(custom_kernels)
    KERNEL_PADDINGS = [0] * len(custom_kernels)
    _noise_bank = {}
    _w_noise_bank = {}


def set_kernel_metadata(strides=None, paddings=None, gn=None):
    """Attach stride/padding metadata for quantized forward paths."""
    global KERNEL_STRIDES, KERNEL_PADDINGS, KERNEL_GN
    if strides is not None:
        KERNEL_STRIDES = list(strides)
    if paddings is not None:
        KERNEL_PADDINGS = list(paddings)
    if gn is not None:
        KERNEL_GN = gn
