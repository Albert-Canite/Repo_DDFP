import numpy as np
from pathlib import Path
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
import cv2
import torch
import torch.nn.functional as F

from core.config import (
    RSNA_TRAIN_IMG_DIR,
    IMAGE_SIZE,
    INPUT_SIGNED,
    rng
)


def load_dicom_image(path, return_shape: bool = False):
    dcm = pydicom.dcmread(path)
    orig_shape = dcm.pixel_array.shape  # (H, W)

    img = apply_modality_lut(dcm.pixel_array, dcm).astype(np.float32)
    if hasattr(dcm, "WindowCenter") and hasattr(dcm, "WindowWidth"):
        img = apply_voi_lut(img, dcm)
    else:
        center, width = -600.0, 1500.0
        img = np.clip(img, center - width / 2, center + width / 2)

    vmin, vmax = img.min(), img.max()
    img = (img - vmin) / (vmax - vmin + 1e-8)
    if INPUT_SIGNED:
        img = img - 0.5

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    img = img[None, None]
    if return_shape:
        return img, orig_shape
    return img


def load_images_rsna(num):
    if not RSNA_TRAIN_IMG_DIR.exists():
        raise FileNotFoundError(
            f"RSNA_TRAIN_IMG_DIR path does not exist: {RSNA_TRAIN_IMG_DIR}"
        )

    files = list(RSNA_TRAIN_IMG_DIR.glob("*.dcm"))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No DICOM images found under: {RSNA_TRAIN_IMG_DIR}"
        )

    sel = rng.choice(files, size=min(num, len(files)), replace=False)
    images = [load_dicom_image(p) for p in sel]
    return images


def fp32_to_fp15(value):
    if isinstance(value, (float, int)) and not np.isfinite(value):
        value = 0.0
    v = float(value)
    if v == 0.0:
        return 0.0
    sign = -1.0 if v < 0.0 else 1.0
    a = abs(v)
    EXPONENT_BITS = 8
    MANTISSA_BITS = 6
    EXPONENT_BIAS = 127
    e_unb = np.floor(np.log2(a))
    e = int(np.clip(e_unb + EXPONENT_BIAS, 0, (1 << EXPONENT_BITS) - 1))
    mantissa_float = a / (2.0 ** (e - EXPONENT_BIAS)) - 1.0
    mantissa = int(np.clip(np.round(mantissa_float * (1 << MANTISSA_BITS)), 0, (1 << MANTISSA_BITS) - 1))
    recon = (1.0 + mantissa / float(1 << MANTISSA_BITS)) * (2.0 ** (e - EXPONENT_BIAS))
    return sign * recon


def snr_db(ref, test):
    s = ref.flatten()
    t = test.flatten()
    mse = np.mean((s - t) ** 2)
    ps = np.mean(s ** 2)
    if mse < 1e-20 or ps < 1e-20:
        return 100.0
    return 10 * np.log10(ps / mse)


def conv2d_fp(x, w):
    """
    FP32 reference convolution using PyTorch.
    x: [1,1,H,W]
    w: [1,1,K,K]
    """
    xt = torch.tensor(x, dtype=torch.float32)
    wt = torch.tensor(w, dtype=torch.float32)
    out = F.conv2d(xt, wt, stride=1)
    return out.numpy()[0, 0]
