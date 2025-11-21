import csv
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import core.config as C
from core.utils import load_dicom_image, snr_db
from ddfp.baseline_forward import run_network_baseline
from ddfp.ddfp_forward import calibrate_ddfp, run_network_ddfp
from ddfp.fp32_forward import run_network_fp32


@dataclass
class BBoxItem:
    pid: str
    box: Tuple[float, float, float, float]


class RSNABBoxDataset(Dataset):
    def __init__(self, items: List[BBoxItem]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = C.RSNA_TRAIN_IMG_DIR / f"{item.pid}.dcm"
        img = load_dicom_image(img_path)
        target = torch.tensor(item.box, dtype=torch.float32)
        return torch.tensor(img, dtype=torch.float32), target


def load_bbox_items(limit=None):
    if not C.RSNA_LABEL_CSV.exists():
        raise FileNotFoundError(f"未找到标签文件: {C.RSNA_LABEL_CSV}")

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
            box = (
                x / C.IMAGE_SIZE,
                y / C.IMAGE_SIZE,
                (x + w) / C.IMAGE_SIZE,
                (y + h) / C.IMAGE_SIZE,
            )
            items.append(BBoxItem(pid=pid, box=box))
            seen.add(pid)
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise RuntimeError("未能加载任何正样本 bbox")
    return items


class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=C.KERNEL_SIZE)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=C.KERNEL_SIZE)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16, C.REGRESSION_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(C.REGRESSION_HEAD_HIDDEN, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(x))
        out = self.head(feat)
        return out

    def feature_kernels(self):
        return [
            self.conv1.weight.detach().cpu().numpy(),
            self.conv2.weight.detach().cpu().numpy(),
        ]


def train_regression_model():
    items = load_bbox_items(limit=C.RSNA_TRAIN_SAMPLES + C.RSNA_VAL_SAMPLES)
    dataset = RSNABBoxDataset(items)
    val_size = min(C.RSNA_VAL_SAMPLES, max(len(dataset) // 5, 1))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("训练样本不足，请提高 RSNA_TRAIN_SAMPLES 或检查标签文件")
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=C.RSNA_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=C.RSNA_BATCH_SIZE)

    model = RegressionNet()
    opt = torch.optim.Adam(model.parameters(), lr=C.RSNA_LR)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(C.RSNA_EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            opt.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                preds = model(imgs)
                val_loss += loss_fn(preds, targets).item() * imgs.size(0)
        val_loss = val_loss / max(len(val_loader.dataset), 1)

        print(f"[Train] epoch={epoch+1} loss={avg_loss:.4f} val={val_loss:.4f}")

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), C.REGRESSION_CKPT)
    return model


def load_or_train_model():
    model = RegressionNet()
    if C.REGRESSION_CKPT.exists():
        model.load_state_dict(torch.load(C.REGRESSION_CKPT, map_location="cpu"))
        print(f"[Load] 使用已有模型 {C.REGRESSION_CKPT}")
        return model
    return train_regression_model()


def compute_metrics(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
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


def run_regression_flow():
    model = load_or_train_model()
    items = load_bbox_items(limit=C.NUM_TEST + C.NUM_CALIBRATION + C.RSNA_VAL_SAMPLES)
    dataset = RSNABBoxDataset(items)

    if len(dataset) <= C.NUM_CALIBRATION:
        raise RuntimeError("数据量不足以划分校准集和测试集，请增大 NUM_TEST/NUM_CALIBRATION 配置")

    calib_set, test_set = random_split(dataset, [C.NUM_CALIBRATION, len(dataset) - C.NUM_CALIBRATION])
    calib_imgs = [img.numpy() for img, _ in calib_set]

    kernels = [k for k in model.feature_kernels()]
    C.setup_config(len(kernels), C.INPUT_BITS_LIST[0], C.WEIGHT_BITS_LIST[0], C.ADC_BITS_LIST[0])
    C.set_kernels(kernels)

    calibrate_ddfp(calib_imgs)

    fp_preds, ddfp_preds, base_preds, gts = [], [], [], []
    snr_records = []

    for img, target in test_set:
        img_np = img.numpy()
        gt = target.numpy()

        with torch.no_grad():
            fp_out = model(img.unsqueeze(0)).squeeze(0).numpy() * C.IMAGE_SIZE
        fp_preds.append(fp_out)
        gts.append(gt * C.IMAGE_SIZE)

        feat_fp32 = run_network_fp32(img_np, C.kernels)
        ddfp_out, _, _, _ = run_network_ddfp(img_np)
        base_out, _, _ = run_network_baseline(img_np)

        for name, feat in [("DDFP", ddfp_out), ("BASE", base_out)]:
            tensor_feat = torch.tensor(feat, dtype=torch.float32)
            with torch.no_grad():
                pred = model.head(tensor_feat).squeeze(0).numpy() * C.IMAGE_SIZE
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

    mae_fp, iou_fp = compute_metrics(fp_preds, gts)
    mae_ddfp, iou_ddfp = compute_metrics(ddfp_preds, gts)
    mae_base, iou_base = compute_metrics(base_preds, gts)

    print(
        f"[Metric] FP32 mae={mae_fp:.2f} iou={iou_fp:.3f} | "
        f"DDFP mae={mae_ddfp:.2f} iou={iou_ddfp:.3f} | "
        f"Baseline mae={mae_base:.2f} iou={iou_base:.3f}"
    )

    save_regression_fig(test_set, fp_preds, ddfp_preds, base_preds, gts)


def draw_box(ax, box, color, label):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(x1, y1, label, color=color, fontsize=8)


def save_regression_fig(test_set, fp_preds, ddfp_preds, base_preds, gts):
    rows = min(3, len(test_set))
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for idx in range(rows):
        img, _ = test_set[idx]
        img_vis = img[0].numpy()
        for col, (preds, title, color) in enumerate([
            (fp_preds, "FP32 标注", "green"),
            (ddfp_preds, "DDFP 标注", "red"),
            (base_preds, "Baseline 标注", "blue"),
        ]):
            ax = axes[idx, col]
            ax.imshow(img_vis, cmap="gray")
            draw_box(ax, gts[idx], "yellow", "GT")
            draw_box(ax, preds[idx], color, title)
            ax.axis("off")
            ax.set_title(title)

    plt.tight_layout()
    fig.savefig(C.REGRESSION_OUTPUT_IMG, dpi=150)
    print(f"[Saved] {C.REGRESSION_OUTPUT_IMG}")


if __name__ == "__main__":
    print("[Task] RSNA bbox regression + DDFP 对比")
    run_regression_flow()
