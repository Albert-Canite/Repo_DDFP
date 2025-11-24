import csv
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import core.config as C
from core.utils import load_dicom_image, snr_db
from ddfp.baseline_forward import run_network_baseline
from ddfp.ddfp_forward import calibrate_ddfp, run_network_ddfp
from ddfp.fp32_forward import run_network_fp32


@dataclass
class BBoxItem:
    pid: str
    box: Tuple[float, float, float, float]  # raw pixel (x, y, w, h) in original image size


class RSNABBoxDataset(Dataset):
    def __init__(self, items: List[BBoxItem], augment: bool = False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = C.RSNA_TRAIN_IMG_DIR / f"{item.pid}.dcm"
        img, orig_shape = load_dicom_image(img_path, return_shape=True)
        h0, w0 = orig_shape
        x, y, w, h = item.box
        box = torch.tensor(
            [x / w0, y / h0, (x + w) / w0, (y + h) / h0], dtype=torch.float32
        )

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=-1).copy()
                x1, y1, x2, y2 = box
                box = torch.tensor([1 - x2, y1, 1 - x1, y2], dtype=torch.float32)
            if np.random.rand() < 0.3:
                scale = np.random.uniform(0.9, 1.1)
                img = np.clip(img * scale, -0.5 if C.INPUT_SIGNED else 0.0, 0.5 if C.INPUT_SIGNED else 1.0)

        return torch.tensor(img, dtype=torch.float32), box


def load_bbox_items(limit=None):
    if not C.RSNA_LABEL_CSV.exists():
        raise FileNotFoundError(f"Label file not found: {C.RSNA_LABEL_CSV}")

    items = []
    seen = set()
    with open(C.RSNA_LABEL_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Target", "0") != "1":
                continue
            pid = row["patientId"]
            if pid in seen:
                continue
            x = float(row["x"])
            y = float(row["y"])
            w = float(row["width"])
            h = float(row["height"])
            items.append(BBoxItem(pid=pid, box=(x, y, w, h)))
            seen.add(pid)
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise RuntimeError("No positive bbox samples were loaded")
    return items


class RegressionNet(nn.Module):
    """Single-channel CNN compatible with RSNA grayscale input and DDFP/Baseline convolution."""

    def __init__(self, center_prior=(0.5, 0.5), size_prior=(C.REGRESSION_SIZE_PRIOR, C.REGRESSION_SIZE_PRIOR)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    1 if idx == 0 else C.CONV_CHANNELS,
                    C.CONV_CHANNELS,
                    kernel_size=C.KERNEL_SIZE,
                    padding=C.KERNEL_SIZE // 2,
                    bias=True,
                )
                for idx in range(5)
            ]
        )
        pool_out = 8
        stride = C.IMAGE_SIZE // pool_out
        if stride * pool_out != C.IMAGE_SIZE:
            raise ValueError(
                f"IMAGE_SIZE={C.IMAGE_SIZE} must be divisible by {pool_out} for deterministic pooling"
            )
        # AvgPool2d is deterministic on CUDA, unlike AdaptiveAvgPool2d backward
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        pooled_feat = pool_out * pool_out * C.CONV_CHANNELS
        coord_channels = 2  # x,y coordinate maps
        # After concatenating x/y coordinate channels, the feature width becomes
        # (C.CONV_CHANNELS + coord_channels) * pool_out * pool_out. Use this exact
        # dimension to keep the head input aligned with the encoder output.
        self.head_in = pooled_feat + coord_channels * pool_out * pool_out
        self.head = nn.Sequential(
            nn.Linear(self.head_in, C.REGRESSION_HEAD_HIDDEN1),
            nn.ReLU(),
            nn.Linear(C.REGRESSION_HEAD_HIDDEN1, C.REGRESSION_HEAD_HIDDEN2),
            nn.ReLU(),
            nn.Linear(C.REGRESSION_HEAD_HIDDEN2, 4),
        )

        with torch.no_grad():
            cx, cy = center_prior
            sw, sh = size_prior
            prior = torch.tensor([cx, cy, sw, sh], dtype=torch.float32)
            prior = prior.clamp(1e-4, 1 - 1e-4)
            prior_logits = torch.logit(prior)
            self.head[-1].bias.copy_(prior_logits)

        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def _append_coords(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        device, dtype = feat.device, feat.dtype
        y_grid = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        x_grid = torch.linspace(0.0, 1.0, steps=w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([feat, x_grid, y_grid], dim=1)

    def _encode_features(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(feat)
        with_coords = self._append_coords(pooled)
        return with_coords.flatten(1)

    def _decode_raw(self, raw: torch.Tensor):
        center = torch.sigmoid(raw[:, :2])
        size = torch.sigmoid(raw[:, 2:]).clamp(min=1e-4)
        half = 0.5 * size
        x1y1 = center - half
        x2y2 = center + half
        boxes = torch.cat([x1y1, x2y2], dim=1)
        return boxes, torch.cat([center, size], dim=1)

    def _forward_internal(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        for conv in self.convs:
            x = F.relu(conv(x))
        encoded = self._encode_features(x)
        raw = self.head(encoded)
        return self._decode_raw(raw)

    def forward(self, x):
        boxes, _ = self._forward_internal(x)
        return boxes

    def forward_with_aux(self, x):
        return self._forward_internal(x)

    def decode_from_features(self, feat: np.ndarray) -> torch.Tensor:
        feat_t = torch.tensor(feat, dtype=torch.float32)
        boxes, _ = self._forward_internal(feat_t)
        return boxes

    def feature_kernels(self):
        return [conv.weight.detach().cpu().numpy() for conv in self.convs]


_cached_splits = None


def build_rsna_splits():
    global _cached_splits
    if _cached_splits is not None:
        return _cached_splits

    needed = C.RSNA_TRAIN_SAMPLES + C.RSNA_VAL_SAMPLES + C.NUM_CALIBRATION + C.NUM_TEST
    items = load_bbox_items(limit=needed)

    generator = torch.Generator().manual_seed(C.SEED)
    indices = torch.randperm(len(items), generator=generator).tolist()

    train_items = [items[i] for i in indices[: C.RSNA_TRAIN_SAMPLES]]
    val_items = [items[i] for i in indices[C.RSNA_TRAIN_SAMPLES : C.RSNA_TRAIN_SAMPLES + C.RSNA_VAL_SAMPLES]]
    calib_items = [
        items[i]
        for i in indices[
            C.RSNA_TRAIN_SAMPLES
            + C.RSNA_VAL_SAMPLES : C.RSNA_TRAIN_SAMPLES
            + C.RSNA_VAL_SAMPLES
            + C.NUM_CALIBRATION
        ]
    ]
    test_items = [
        items[i]
        for i in indices[
            C.RSNA_TRAIN_SAMPLES
            + C.RSNA_VAL_SAMPLES
            + C.NUM_CALIBRATION : C.RSNA_TRAIN_SAMPLES
            + C.RSNA_VAL_SAMPLES
            + C.NUM_CALIBRATION
            + C.NUM_TEST
        ]
    ]

    train_set = RSNABBoxDataset(train_items, augment=True)
    val_set = RSNABBoxDataset(val_items)
    calib_set = RSNABBoxDataset(calib_items)
    test_set = RSNABBoxDataset(test_items)

    _cached_splits = (train_set, val_set, calib_set, test_set)
    return _cached_splits


def compute_priors(train_items: List[BBoxItem]):
    centers, sizes = [], []
    for item in train_items:
        img_path = C.RSNA_TRAIN_IMG_DIR / f"{item.pid}.dcm"
        _, orig_shape = load_dicom_image(img_path, return_shape=True)
        h0, w0 = orig_shape
        x, y, w, h = item.box
        x1 = np.clip(x / w0, 0.0, 1.0)
        y1 = np.clip(y / h0, 0.0, 1.0)
        x2 = np.clip((x + w) / w0, 0.0, 1.0)
        y2 = np.clip((y + h) / h0, 0.0, 1.0)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        sizes.append([x2 - x1, y2 - y1])
        centers.append([cx, cy])
    centers = np.array(centers)
    sizes = np.array(sizes)
    center_prior = centers.mean(axis=0).tolist()
    size_prior = np.clip(sizes.mean(axis=0), 1e-3, 0.9).tolist()
    return center_prior, size_prior


def train_regression_model(train_set, val_set, priors):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_set, batch_size=C.RSNA_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=C.RSNA_BATCH_SIZE)

    model = RegressionNet(center_prior=priors[0], size_prior=priors[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.RSNA_LR, weight_decay=C.RSNA_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.3, patience=4, min_lr=1e-5
    )
    loss_fn = nn.L1Loss()
    history = []
    best_state = None
    best_val = float("inf")
    best_metrics = (float("inf"), 0.0)

    for epoch in range(C.RSNA_EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            opt.zero_grad()
            boxes, center_sizes = model.forward_with_aux(imgs)
            boxes = order_and_clip_boxes(boxes)
            targets = order_and_clip_boxes(targets)
            targets_cs = boxes_to_center_size(targets)

            boxes_pix = boxes * C.IMAGE_SIZE
            targets_pix = targets * C.IMAGE_SIZE
            cs_pix = center_sizes * C.IMAGE_SIZE
            targets_cs_pix = targets_cs * C.IMAGE_SIZE

            loss_cs = loss_fn(cs_pix, targets_cs_pix)
            loss_corner = loss_fn(boxes_pix, targets_pix)
            loss_iou = 1.0 - bbox_iou_tensor(boxes_pix, targets_pix)
            loss = loss_cs + 0.5 * loss_corner + C.BBOX_IOU_LOSS_WEIGHT * loss_iou
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                boxes, _ = model.forward_with_aux(imgs)
                boxes = order_and_clip_boxes(boxes)
                targets = order_and_clip_boxes(targets)
                preds_pix = boxes * C.IMAGE_SIZE
                targets_pix = targets * C.IMAGE_SIZE
                loss_l1 = loss_fn(preds_pix, targets_pix)
                loss_iou = 1.0 - bbox_iou_tensor(preds_pix, targets_pix)
                batch_loss = (loss_l1 + C.BBOX_IOU_LOSS_WEIGHT * loss_iou).item()
                val_loss += batch_loss * imgs.size(0)
                val_preds.append(preds_pix.detach().cpu().numpy())
                val_targets.append(targets_pix.detach().cpu().numpy())
        val_loss = val_loss / max(len(val_loader.dataset), 1)
        if val_preds and val_targets:
            val_mae, val_iou = compute_metrics(
                np.concatenate(val_preds), np.concatenate(val_targets)
            )
        else:
            val_mae, val_iou = float("nan"), float("nan")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_iou": val_iou,
                "lr": opt.param_groups[0]["lr"],
            }
        )

        print(
            f"[Train] epoch={epoch+1} loss={avg_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mae={val_mae:.2f} val_iou={val_iou:.3f}"
        )

        # Drive learning-rate scheduling by validation MAE to match the target metric
        scheduler.step(val_mae)

        if val_mae < best_val:
            best_val = val_mae
            best_state = model.state_dict()
            best_metrics = (val_mae, val_iou)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"[Load] best validation model: val_mae={best_metrics[0]:.2f}, "
            f"val_iou={best_metrics[1]:.3f}"
        )

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), C.REGRESSION_CKPT)
    plot_training_history(history)
    return model_cpu


def load_or_train_model(train_set, val_set, priors):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionNet(center_prior=priors[0], size_prior=priors[1]).to(device)
    if C.REGRESSION_CKPT.exists():
        try:
            state = torch.load(C.REGRESSION_CKPT, map_location="cpu")
            model.load_state_dict(state)
            print(f"[Load] using existing model {C.REGRESSION_CKPT}")
            return model.cpu()
        except RuntimeError as err:
            backup = C.REGRESSION_CKPT.with_suffix(".ckpt.incompatible")
            try:
                C.REGRESSION_CKPT.rename(backup)
                print(
                    f"[Load] incompatible checkpoint detected, backed up to {backup}, retraining. Error: {err}"
                )
            except OSError:
                print(
                    f"[Load] incompatible checkpoint detected ({err}); cannot back up automatically. "
                    f"Please delete {C.REGRESSION_CKPT} and retry."
                )
    return train_regression_model(train_set, val_set, priors)


def boxes_to_center_size(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    w = torch.clamp(x2 - x1, min=1e-6)
    h = torch.clamp(y2 - y1, min=1e-6)
    return torch.stack([center_x, center_y, w, h], dim=1)


def order_and_clip_boxes(boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    ordered = torch.stack([x1, y1, x2, y2], dim=1)
    return ordered.clamp(0.0, 1.0)


def bbox_iou_tensor(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    px1, py1, px2, py2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    tx1, ty1, tx2, ty2 = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)

    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    area_p = torch.clamp(px2 - px1, min=0) * torch.clamp(py2 - py1, min=0)
    area_t = torch.clamp(tx2 - tx1, min=0) * torch.clamp(ty2 - ty1, min=0)
    union = area_p + area_t - inter + 1e-8

    iou = inter / union
    return iou.mean()


def compute_metrics(preds, targets):
    preds = np.array([sanitize_box(p) for p in preds])
    targets = np.array([sanitize_box(t) for t in targets])
    mae = np.mean(np.abs(preds - targets))

    ious = []
    for p, t in zip(preds, targets):
        px1, py1, px2, py2 = p
        tx1, ty1, tx2, ty2 = t
        ix1, iy1 = max(px1, tx1), max(py1, ty1)
        ix2, iy2 = min(px2, tx2), min(py2, ty2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_p = max(0.0, px2 - px1) * max(0.0, py2 - py1)
        area_t = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
        union = area_p + area_t - inter + 1e-8
        ious.append(inter / union)
    return mae, float(np.mean(ious))


def sanitize_box(box):
    x1, y1, x2, y2 = box
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1 = np.clip(x1, 0.0, C.IMAGE_SIZE)
    y1 = np.clip(y1, 0.0, C.IMAGE_SIZE)
    x2 = np.clip(x2, 0.0, C.IMAGE_SIZE)
    y2 = np.clip(y2, 0.0, C.IMAGE_SIZE)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def plot_training_history(history):
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_mae = [h["val_mae"] for h in history]
    val_iou = [h["val_iou"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curves")
    axes[0].legend()

    axes[1].plot(epochs, val_mae, label="val_mae")
    axes[1].plot(epochs, val_iou, label="val_iou")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation metrics")
    axes[1].legend()

    plt.tight_layout()
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.REGRESSION_TRAIN_CURVE, dpi=150)
    print(f"[Saved] {C.REGRESSION_TRAIN_CURVE}")


def evaluate_fp32(model, test_set):
    fp_preds, gts = [], []
    with torch.no_grad():
        for img, target in test_set:
            fp_out = model(img.unsqueeze(0))
            fp_out = order_and_clip_boxes(fp_out).squeeze(0).numpy() * C.IMAGE_SIZE
            fp_preds.append(sanitize_box(fp_out))
            gts.append(sanitize_box(target.numpy() * C.IMAGE_SIZE))
    mae_fp, iou_fp = compute_metrics(fp_preds, gts)
    return fp_preds, gts, (mae_fp, iou_fp)


def per_sample_metrics(pred, target):
    pred_box = sanitize_box(pred)
    target_box = sanitize_box(target)
    mae = float(np.mean(np.abs(pred_box - target_box)))
    px1, py1, px2, py2 = pred_box
    tx1, ty1, tx2, ty2 = target_box
    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_p = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    area_t = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union = area_p + area_t - inter + 1e-8
    iou = float(inter / union)
    return mae, iou


def save_fp32_tracking(test_set, fp_preds, gts, metrics):
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.REGRESSION_FP32_TRACKING, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "patientId",
                "gt_x1",
                "gt_y1",
                "gt_x2",
                "gt_y2",
                "pred_x1",
                "pred_y1",
                "pred_x2",
                "pred_y2",
                "mae",
                "iou",
            ]
        )
        for item, pred, gt in zip(test_set.items, fp_preds, gts):
            mae, iou = per_sample_metrics(pred, gt)
            writer.writerow([item.pid, *gt, *pred, mae, iou])
    print(
        f"[Saved] FP32 tracking {C.REGRESSION_FP32_TRACKING} (overall MAE={metrics[0]:.2f}, IoU={metrics[1]:.3f})"
    )


def save_fp32_fig(test_set, fp_preds, gts, fp_metrics):
    rows = min(3, len(test_set))
    fig, axes = plt.subplots(rows, 1, figsize=(4, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for idx in range(rows):
        img, _ = test_set[idx]
        img_vis = img.squeeze().numpy()
        ax = axes[idx]
        ax.imshow(img_vis, cmap="gray")
        draw_box(ax, gts[idx], "yellow", "GT")
        draw_box(ax, fp_preds[idx], "green", "FP32")
        ax.axis("off")
        ax.set_title("FP32 annotation")

    plt.tight_layout()
    fig.text(
        0.5,
        0.02,
        f"FP32 only: MAE={fp_metrics[0]:.2f}, IoU={fp_metrics[1]:.3f}",
        ha="center",
        fontsize=10,
    )
    fig.savefig(C.REGRESSION_FP32_IMG, dpi=150)
    print(f"[Saved] {C.REGRESSION_FP32_IMG}")


def run_regression_flow():
    train_set, val_set, calib_set, test_set = build_rsna_splits()
    priors = compute_priors(train_set.items)
    print(f"[Prior] center={priors[0]}, size={priors[1]}")
    model = load_or_train_model(train_set, val_set, priors)

    fp_preds, gts, fp_metrics = evaluate_fp32(model, test_set)
    save_fp32_tracking(test_set, fp_preds, gts, fp_metrics)
    save_fp32_fig(test_set, fp_preds, gts, fp_metrics)

    calib_imgs = [img.numpy() for img, _ in calib_set]

    kernels = [k for k in model.feature_kernels()]
    C.setup_config(len(kernels), C.INPUT_BITS_LIST[0], C.WEIGHT_BITS_LIST[0], C.ADC_BITS_LIST[0])
    C.set_kernels(kernels)

    calibrate_ddfp(calib_imgs, apply_relu=True)

    ddfp_preds, base_preds = [], []
    snr_records = []

    for img, target in test_set:
        img_np = img.numpy()
        gt = target.numpy()

        feat_fp32 = run_network_fp32(img_np, C.kernels, apply_relu=True)
        ddfp_out, _, _, _ = run_network_ddfp(img_np, apply_relu=True)
        base_out, _, _ = run_network_baseline(img_np, apply_relu=True)

        for name, feat in [("DDFP", ddfp_out), ("BASE", base_out)]:
            with torch.no_grad():
                pred = model.decode_from_features(feat)
                pred = order_and_clip_boxes(pred).squeeze(0).numpy() * C.IMAGE_SIZE
                pred = sanitize_box(pred)
            if name == "DDFP":
                ddfp_preds.append(pred)
            else:
                base_preds.append(pred)

        snr_records.append(
            (
                snr_db(feat_fp32, ddfp_out),
                snr_db(feat_fp32, base_out),
            )
        )

    mae_fp, iou_fp = fp_metrics
    mae_ddfp, iou_ddfp = compute_metrics(ddfp_preds, gts)
    mae_base, iou_base = compute_metrics(base_preds, gts)

    print(
        f"[Metric] FP32 mae={mae_fp:.2f} iou={iou_fp:.3f} | "
        f"DDFP mae={mae_ddfp:.2f} iou={iou_ddfp:.3f} | "
        f"Baseline mae={mae_base:.2f} iou={iou_base:.3f}"
    )

    save_regression_fig(
        test_set,
        fp_preds,
        ddfp_preds,
        base_preds,
        gts,
        (mae_fp, iou_fp),
        (mae_ddfp, iou_ddfp),
        (mae_base, iou_base),
    )


def draw_box(ax, box, color, label):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(x1, y1, label, color=color, fontsize=8)


def save_regression_fig(
    test_set,
    fp_preds,
    ddfp_preds,
    base_preds,
    gts,
    fp_metrics,
    ddfp_metrics,
    base_metrics,
):
    rows = min(3, len(test_set))
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for idx in range(rows):
        img, _ = test_set[idx]
        img_vis = img.squeeze().numpy()
        for col, (preds, title, color) in enumerate(
            [
                (fp_preds, "FP32 annotation", "green"),
                (ddfp_preds, "DDFP annotation", "red"),
                (base_preds, "Baseline annotation", "blue"),
            ]
        ):
            ax = axes[idx, col]
            ax.imshow(img_vis, cmap="gray")
            draw_box(ax, gts[idx], "yellow", "GT")
            draw_box(ax, preds[idx], color, title)
            ax.axis("off")
            ax.set_title(title)

    plt.tight_layout()
    fig.text(
        0.5,
        0.02,
        (
            f"FP32: MAE={fp_metrics[0]:.2f}, IoU={fp_metrics[1]:.3f} | "
            f"DDFP: MAE={ddfp_metrics[0]:.2f}, IoU={ddfp_metrics[1]:.3f} | "
            f"Baseline: MAE={base_metrics[0]:.2f}, IoU={base_metrics[1]:.3f}"
        ),
        ha="center",
        fontsize=10,
    )
    fig.savefig(C.REGRESSION_OUTPUT_IMG, dpi=150)
    print(f"[Saved] {C.REGRESSION_OUTPUT_IMG}")


if __name__ == "__main__":
    print("[Task] RSNA bbox regression + DDFP comparison")
    run_regression_flow()
