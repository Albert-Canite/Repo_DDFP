import csv
import json
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import core.config as C
from core.utils import snr_db
from ddfp.baseline_forward import run_network_baseline
from ddfp.ddfp_forward import calibrate_ddfp, run_network_ddfp
from ddfp.fp32_forward import run_network_fp32


@dataclass
class DetectionSample:
    path: Path
    boxes: np.ndarray  # [N, 4] xyxy absolute pixels
    labels: List[int]
    size: Tuple[int, int]


class BCCDDataset(Dataset):
    def __init__(self, samples: List[DetectionSample], augment: bool = False):
        self.samples = samples
        self.augment = augment
        self.color_jitter = C.BCCD_COLOR_JITTER
        self.hflip_prob = C.BCCD_HFLIP_PROB

    def __len__(self):
        return len(self.samples)

    def _augment_boxes(self, boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        if np.random.rand() < self.hflip_prob:
            boxes = boxes.copy()
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = img_w - x2
            boxes[:, 2] = img_w - x1
        return boxes

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32) / 255.0
        if self.color_jitter > 0:
            delta = self.color_jitter
            img = img * np.random.uniform(1 - delta, 1 + delta)
            img = img + np.random.uniform(-delta, delta)
        img = np.clip(img, 0.0, 1.0)
        return img

    def _normalize_boxes(self, boxes: np.ndarray, w: int, h: int) -> np.ndarray:
        boxes_norm = boxes.copy().astype(np.float32)
        boxes_norm[:, [0, 2]] /= w
        boxes_norm[:, [1, 3]] /= h
        return boxes_norm

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = plt.imread(sample.path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        h, w, _ = img.shape

        boxes = sample.boxes.copy()
        labels = np.array(sample.labels, dtype=np.int64)

        if self.augment:
            flip_flag = np.random.rand() < self.hflip_prob
            if flip_flag:
                boxes = self._augment_boxes(boxes, w, h)
                img = np.flip(img, axis=1).copy()
            img = self._augment_image(img)

        boxes_norm = self._normalize_boxes(boxes, w, h)
        img_resized = torch.tensor(
            F.interpolate(
                torch.tensor(img).permute(2, 0, 1).unsqueeze(0),
                size=(C.IMAGE_SIZE, C.IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        )
        img_resized = img_resized.clamp(0.0, 1.0)
        boxes_scaled = boxes_norm.copy()
        return img_resized.float(), torch.tensor(boxes_scaled, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def _load_coco(anno_path: Path) -> List[DetectionSample]:
    data = json.loads(anno_path.read_text())
    id_to_file = {img["id"]: img["file_name"] for img in data.get("images", [])}
    id_to_size = {img["id"]: (img["width"], img["height"]) for img in data.get("images", [])}
    cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
    samples: Dict[int, List[Tuple[List[float], int]]] = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        bbox = ann["bbox"]  # x, y, w, h
        x1, y1, w, h = bbox
        boxes = [x1, y1, x1 + w, y1 + h]
        cls_name = cat_map.get(ann["category_id"], "")
        if cls_name not in C.BCCD_CLASSES:
            continue
        cls_id = C.BCCD_CLASSES.index(cls_name)
        samples.setdefault(img_id, []).append((boxes, cls_id))

    output = []
    for img_id, items in samples.items():
        file_name = id_to_file[img_id]
        w, h = id_to_size[img_id]
        boxes = np.array([it[0] for it in items], dtype=np.float32)
        labels = [it[1] for it in items]
        output.append(
            DetectionSample(
                path=C.BCCD_IMG_DIR / file_name,
                boxes=boxes,
                labels=labels,
                size=(w, h),
            )
        )
    return output


def _load_voc(anno_dir: Path) -> List[DetectionSample]:
    samples = []
    for xml_file in sorted(anno_dir.glob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.findtext("filename")
        size_node = root.find("size")
        w = int(size_node.findtext("width"))
        h = int(size_node.findtext("height"))
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in C.BCCD_CLASSES:
                continue
            cls = C.BCCD_CLASSES.index(name)
            bnd = obj.find("bndbox")
            x1 = float(bnd.findtext("xmin"))
            y1 = float(bnd.findtext("ymin"))
            x2 = float(bnd.findtext("xmax"))
            y2 = float(bnd.findtext("ymax"))
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
        if boxes:
            samples.append(
                DetectionSample(
                    path=C.BCCD_IMG_DIR / filename,
                    boxes=np.array(boxes, dtype=np.float32),
                    labels=labels,
                    size=(w, h),
                )
            )
    return samples


def _load_csv(anno_path: Path) -> List[DetectionSample]:
    if not anno_path.exists():
        return []

    samples: Dict[str, Dict[str, object]] = {}
    cls_lookup = {c.lower(): i for i, c in enumerate(C.BCCD_CLASSES)}
    with anno_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV annotation has no header: {anno_path}")
        header_map = {name.lower(): name for name in reader.fieldnames}
        def _find(keys: List[str]):
            for k in keys:
                if k in header_map:
                    return header_map[k]
            return None

        fname_key = _find(["filename", "image", "img", "image_path", "path"])
        label_key = _find(["class", "label", "category", "type"])
        x1_key = _find(["xmin", "x1", "left"])
        y1_key = _find(["ymin", "y1", "top"])
        x2_key = _find(["xmax", "x2", "right"])
        y2_key = _find(["ymax", "y2", "bottom"])
        w_key = _find(["width", "img_width", "w"])
        h_key = _find(["height", "img_height", "h"])

        required = [fname_key, label_key, x1_key, y1_key, x2_key, y2_key]
        if any(k is None for k in required):
            raise ValueError(
                f"CSV annotation missing required columns: have {reader.fieldnames}"
            )

        for row in reader:
            fname = Path(row[fname_key]).name
            cls_name = row[label_key].strip()
            cls_key = cls_name.lower()
            if cls_key not in cls_lookup:
                continue
            try:
                box = [
                    float(row[x1_key]),
                    float(row[y1_key]),
                    float(row[x2_key]),
                    float(row[y2_key]),
                ]
            except ValueError:
                continue
            entry = samples.setdefault(fname, {"boxes": [], "labels": [], "size": (None, None)})
            entry["boxes"].append(box)
            entry["labels"].append(cls_lookup[cls_key])
            if w_key and h_key and entry["size"] == (None, None):
                try:
                    entry["size"] = (int(float(row[w_key])), int(float(row[h_key])))
                except ValueError:
                    entry["size"] = (None, None)

    output: List[DetectionSample] = []
    for fname, data in samples.items():
        img_path = C.BCCD_IMG_DIR / fname
        if not img_path.exists():
            print(f"[Warning] image not found for annotation: {img_path}")
            continue
        boxes = np.array(data["boxes"], dtype=np.float32)
        labels = data["labels"]
        w, h = data["size"]
        if w is None or h is None:
            img = plt.imread(img_path)
            h, w = img.shape[:2]
        output.append(
            DetectionSample(
                path=img_path,
                boxes=boxes,
                labels=labels,
                size=(w, h),
            )
        )
    return output


def load_bccd_samples() -> List[DetectionSample]:
    # Priority: COCO JSON, VOC XML folder, CSV fallback
    if C.BCCD_ANNO_DIR.exists() and C.BCCD_ANNO_DIR.is_dir():
        coco_files = list(C.BCCD_ANNO_DIR.glob("*.json"))
        if coco_files:
            samples = _load_coco(coco_files[0])
        else:
            samples = _load_voc(C.BCCD_ANNO_DIR)
        if samples:
            return samples
    if C.BCCD_ANNO_CSV.exists():
        samples = _load_csv(C.BCCD_ANNO_CSV)
        if samples:
            return samples
    raise FileNotFoundError(
        f"No annotations found. Expected COCO/VOC under {C.BCCD_ANNO_DIR} or CSV at {C.BCCD_ANNO_CSV}."
    )


class StdConv(nn.Conv2d):
    def forward(self, x):
        if C.BCCD_WEIGHT_STANDARDIZE:
            w = self.weight
            mean = w.mean(dim=(1, 2, 3), keepdim=True)
            std = w.std(dim=(1, 2, 3), keepdim=True) + 1e-5
            w = (w - mean) / std
            return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return super().forward(x)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, g=None, act=True):
        super().__init__()
        g = g or C.BCCD_GN_GROUPS
        self.conv = StdConv(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2, bias=True)
        self.gn = nn.GroupNorm(g, out_ch, eps=1e-5)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class YoloTiny(nn.Module):
    def __init__(self, num_classes: int, anchors: List[Tuple[int, int]]):
        super().__init__()
        self.anchors = anchors
        width = [16, 32, 64, 128, 256, 256]
        layers = []
        in_ch = 3
        strides = []
        paddings = []
        gn_meta = []
        for idx, out_ch in enumerate(width):
            stride = 2 if idx < 5 else 1
            block = ConvGNAct(in_ch, out_ch, k=3, s=stride)
            layers.append(block)
            strides.append(stride)
            paddings.append(1)
            gn_meta.append((block.gn.num_groups, block.gn.eps, block.gn.weight, block.gn.bias, True))
            in_ch = out_ch
        self.backbone = nn.ModuleList(layers)
        self.head = nn.Conv2d(
            in_ch,
            len(anchors) * (5 + num_classes),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        gn_meta.append(None)
        strides.append(1)
        paddings.append(0)
        self.kernel_strides = strides
        self.kernel_paddings = paddings
        self.kernel_gn = gn_meta

    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        pred = self.head(x)
        return pred

    def export_kernels(self):
        kernels = []
        gn_meta = []
        strides = []
        paddings = []
        for layer in self.backbone:
            conv_w = layer.conv.weight.detach().cpu().numpy()
            kernels.append(conv_w)
            gn_meta.append(
                {
                    "groups": layer.gn.num_groups,
                    "eps": layer.gn.eps,
                    "weight": layer.gn.weight.detach().cpu().numpy(),
                    "bias": layer.gn.bias.detach().cpu().numpy(),
                    "activation": True,
                }
            )
            strides.append(layer.conv.stride[0])
            paddings.append(layer.conv.padding[0])
        kernels.append(self.head.weight.detach().cpu().numpy())
        gn_meta.append(None)
        strides.append(self.head.stride[0])
        paddings.append(self.head.padding[0])
        return kernels, strides, paddings, gn_meta


def build_targets(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    anchors: List[Tuple[int, int]],
    grid_size: int,
    num_classes: int,
):
    device = boxes.device
    bs = boxes.shape[0]
    num_anchors = len(anchors)
    targets = torch.zeros(bs, num_anchors, grid_size, grid_size, 5 + num_classes, device=device)
    anchor_tensor = torch.tensor(anchors, device=device, dtype=torch.float32)

    for b in range(bs):
        for box, cls in zip(boxes[b], labels[b]):
            if box.numel() == 0:
                continue
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            bw = (box[2] - box[0])
            bh = (box[3] - box[1])
            gi = int(cx * grid_size)
            gj = int(cy * grid_size)
            if gi >= grid_size or gj >= grid_size:
                continue
            box_wh = torch.tensor([bw * C.IMAGE_SIZE, bh * C.IMAGE_SIZE], device=device)
            anchor_wh = anchor_tensor
            iou = bbox_wh_iou(box_wh[None], anchor_wh)
            best = torch.argmax(iou)
            targets[b, best, gj, gi, 0:4] = torch.tensor([cx, cy, bw, bh], device=device)
            targets[b, best, gj, gi, 4] = 1.0
            targets[b, best, gj, gi, 5 + cls] = 1.0
    return targets


def bbox_wh_iou(box1, box2):
    b1w, b1h = box1[..., 0], box1[..., 1]
    b2w, b2h = box2[..., 0], box2[..., 1]
    inter = torch.min(b1w, b2w) * torch.min(b1h, b2h)
    union = (b1w * b1h) + (b2w * b2h) - inter + 1e-8
    return inter / union


def bbox_ciou(pred, target):
    px, py, pw, ph = pred.unbind(-1)
    tx, ty, tw, th = target.unbind(-1)
    pred_x1 = px - pw / 2
    pred_y1 = py - ph / 2
    pred_x2 = px + pw / 2
    pred_y2 = py + ph / 2
    tgt_x1 = tx - tw / 2
    tgt_y1 = ty - th / 2
    tgt_x2 = tx + tw / 2
    tgt_y2 = ty + th / 2

    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)

    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    union = (pred_x2 - pred_x1) * (pred_y2 - pred_y1) + (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1) - inter + 1e-8
    iou = inter / union

    cw = torch.max(pred_x2, tgt_x2) - torch.min(pred_x1, tgt_x1)
    ch = torch.max(pred_y2, tgt_y2) - torch.min(pred_y1, tgt_y1)
    c2 = cw ** 2 + ch ** 2 + 1e-8

    rho2 = (px - tx) ** 2 + (py - ty) ** 2
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(tw / th) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / torch.clamp(1 - iou + v, min=1e-6)
    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou


class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, targets):
        obj_mask = targets[..., 4:5]
        noobj_mask = 1.0 - obj_mask
        box_pred = preds[..., 0:4]
        box_tgt = targets[..., 0:4]

        ciou = bbox_ciou(box_pred, box_tgt)
        box_loss = (1.0 - ciou) * obj_mask.squeeze(-1)

        obj_loss = self.bce(preds[..., 4:5], obj_mask) * (C.BCCD_OBJ_LOSS_WEIGHT)
        cls_loss = self.bce(preds[..., 5:], targets[..., 5:]) * obj_mask
        cls_loss = cls_loss.sum(dim=-1, keepdim=True)

        noobj_loss = self.bce(preds[..., 4:5], torch.zeros_like(obj_mask)) * noobj_mask

        total = (
            C.BCCD_BOX_LOSS_WEIGHT * box_loss.mean()
            + obj_loss.mean()
            + noobj_loss.mean() * 0.5
            + C.BCCD_CLS_LOSS_WEIGHT * cls_loss.mean()
        )
        return total, {
            "box": box_loss.mean().item(),
            "obj": obj_loss.mean().item(),
            "cls": cls_loss.mean().item(),
            "noobj": noobj_loss.mean().item(),
            "ciou": ciou.mean().item(),
        }


def decode_predictions(preds: torch.Tensor, anchors: List[Tuple[int, int]], score_thresh: float):
    bs, _, h, w = preds.shape
    num_anchors = len(anchors)
    num_classes = len(C.BCCD_CLASSES)
    preds = preds.view(bs, num_anchors, 5 + num_classes, h, w).permute(0, 1, 3, 4, 2)
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    grid_x = grid_x.to(preds.device)
    grid_y = grid_y.to(preds.device)
    anchor_w = torch.tensor([a[0] for a in anchors], device=preds.device).view(1, num_anchors, 1, 1)
    anchor_h = torch.tensor([a[1] for a in anchors], device=preds.device).view(1, num_anchors, 1, 1)

    pred_xy = torch.sigmoid(preds[..., 0:2])
    pred_wh = torch.exp(preds[..., 2:4])
    pred_obj = torch.sigmoid(preds[..., 4])
    pred_cls = torch.sigmoid(preds[..., 5:])

    bx = (pred_xy[..., 0] + grid_x) / w
    by = (pred_xy[..., 1] + grid_y) / h
    bw = (pred_wh[..., 0] * anchor_w) / C.IMAGE_SIZE
    bh = (pred_wh[..., 1] * anchor_h) / C.IMAGE_SIZE

    scores, cls_idx = torch.max(pred_cls, dim=-1)
    scores = scores * pred_obj

    boxes = torch.stack([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2], dim=-1)
    mask = scores > score_thresh
    outputs = []
    for b in range(bs):
        b_mask = mask[b]
        if b_mask.sum() == 0:
            outputs.append((torch.empty((0, 4)), torch.empty((0,), dtype=torch.long), torch.empty((0,))))
            continue
        b_boxes = boxes[b][b_mask]
        b_scores = scores[b][b_mask]
        b_cls = cls_idx[b][b_mask]
        keep = nms(b_boxes, b_scores, C.BCCD_NMS_IOU)
        outputs.append((b_boxes[keep], b_cls[keep], b_scores[keep]))
    return outputs


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float):
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    x1, y1, x2, y2 = boxes.unbind(-1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        idx = (iou <= iou_thresh).nonzero(as_tuple=False).squeeze(1)
        order = order[idx + 1]
    return torch.tensor(keep, dtype=torch.long)


def collate_fn(batch):
    imgs, boxes, labels = zip(*batch)
    max_boxes = max([b.shape[0] for b in boxes]) if boxes else 0
    padded_boxes = []
    padded_labels = []
    for b, l in zip(boxes, labels):
        pad_b = torch.zeros((max_boxes, 4))
        pad_l = torch.full((max_boxes,), -1, dtype=torch.long)
        pad_b[: b.shape[0]] = b
        pad_l[: l.shape[0]] = l
        padded_boxes.append(pad_b)
        padded_labels.append(pad_l)
    return torch.stack(imgs, 0), torch.stack(padded_boxes, 0), torch.stack(padded_labels, 0)


def visualize_samples(dataset: Dataset, path: Path, max_samples: int = 4):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(max_samples, 2)
    rows = math.ceil(max_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = axes.flatten()
    for idx in range(max_samples):
        img, boxes, labels = dataset[idx]
        ax = axes[idx]
        ax.imshow(img.permute(1, 2, 0))
        for box, cls in zip(boxes, labels):
            if cls < 0:
                continue
            x1, y1, x2, y2 = box * C.IMAGE_SIZE
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1, C.BCCD_CLASSES[int(cls)], color="yellow", fontsize=8)
        ax.axis("off")
    for ax in axes[max_samples:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_epoch(model, loader, optimizer, scheduler, device, anchors, loss_fn):
    model.train()
    total_loss = 0
    logs = []
    for step, (imgs, boxes, labels) in enumerate(loader, 1):
        imgs = imgs.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        grid = preds.shape[-1]
        targets = build_targets(boxes, labels, anchors, grid, len(C.BCCD_CLASSES))
        preds = preds.view(imgs.size(0), len(anchors), 5 + len(C.BCCD_CLASSES), grid, grid).permute(0, 1, 3, 4, 2)
        loss, detail = loss_fn(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        if step % max(1, C.BCCD_LOG_INTERVAL) == 0:
            print(
                f"[Train] step {step}/{len(loader)} loss={loss.item():.4f} ciou={detail['ciou']:.4f}"
            )
        logs.append(detail)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, logs


def evaluate(model, loader, device, anchors):
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for imgs, boxes, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            outputs = decode_predictions(preds, anchors, C.BCCD_SCORE_THRESH)
            for out, gt_boxes, gt_labels in zip(outputs, boxes, labels):
                b_boxes, b_cls, _ = out
                preds_all.append((b_boxes.cpu(), b_cls.cpu()))
                gts_all.append((gt_boxes, gt_labels))
    return compute_metrics(preds_all, gts_all)


def compute_metrics(preds, gts):
    total_iou, total_mae, total = 0.0, 0.0, 0
    for (pb, pl), (gb, gl) in zip(preds, gts):
        if gb.numel() == 0:
            continue
        total += 1
        if pb.numel() == 0:
            continue
        ious = []
        maes = []
        for pbox in pb:
            iou = box_iou_single(pbox, gb[0])
            ious.append(iou)
            maes.append(torch.abs(pbox - gb[0]).mean())
        total_iou += float(torch.tensor(ious).mean()) if ious else 0.0
        total_mae += float(torch.tensor(maes).mean()) if maes else 0.0
    if total == 0:
        return 0.0, 0.0
    return total_mae / total, total_iou / total


def box_iou_single(box1, box2):
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = torch.clamp(box1[2] - box1[0], min=0) * torch.clamp(box1[3] - box1[1], min=0)
    area2 = torch.clamp(box2[2] - box2[0], min=0) * torch.clamp(box2[3] - box2[1], min=0)
    union = area1 + area2 - inter + 1e-8
    return inter / union


def plot_curves(history, path_loss: Path, path_metric: Path):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    ax[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(epochs, [h["mae"] for h in history], label="MAE")
    ax[1].plot(epochs, [h["iou"] for h in history], label="IoU")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Metric")
    ax[1].legend()

    fig.tight_layout()
    path_loss.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_loss, dpi=150)
    fig.savefig(path_metric, dpi=150)
    plt.close(fig)


def prepare_dataloaders():
    samples = load_bccd_samples()
    if len(samples) == 0:
        raise ValueError(
            f"No labeled samples found. Check annotation file names and class labels at {C.BCCD_ANNO_DIR} or {C.BCCD_ANNO_CSV}."
        )
    random.seed(C.BCCD_SEED)
    random.shuffle(samples)
    n_total = len(samples)
    n_val = int(n_total * C.BCCD_VAL_SPLIT)
    n_test = max(int(n_total * C.BCCD_TEST_SPLIT), C.NUM_TEST)
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Not enough samples for the requested splits (total={n_total}, val={n_val}, test={n_test}). "
            "Reduce validation/test split ratios or ensure annotations are correctly parsed."
        )
    train_set = BCCDDataset(samples[:n_train], augment=True)
    val_set = BCCDDataset(samples[n_train : n_train + n_val])
    test_set = BCCDDataset(samples[n_train + n_val :])

    train_loader = DataLoader(
        train_set,
        batch_size=C.BCCD_BATCH_SIZE,
        shuffle=True,
        num_workers=C.BCCD_NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=C.BCCD_BATCH_SIZE,
        shuffle=False,
        num_workers=C.BCCD_NUM_WORKERS,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader, train_set


def save_checkpoint(model, optimizer, scheduler, epoch, best=False):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    path = C.BCCD_BEST_CKPT if best else C.BCCD_CKPT
    torch.save(ckpt, path)
    print(f"[Checkpoint] saved to {path}")


def load_checkpoint(model, optimizer=None, scheduler=None):
    if not C.BCCD_CKPT.exists():
        return 0
    ckpt = torch.load(C.BCCD_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[Checkpoint] loaded from {C.BCCD_CKPT}")
    return ckpt.get("epoch", 0)


def export_for_quant(model: YoloTiny):
    kernels, strides, paddings, gn_meta = model.export_kernels()
    import core.config as C

    C.set_kernels(kernels)
    C.set_kernel_metadata(strides, paddings, gn_meta)


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, train_set = prepare_dataloaders()
    anchors = C.BCCD_ANCHORS
    model = YoloTiny(len(C.BCCD_CLASSES), anchors).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=C.BCCD_LR,
        momentum=C.BCCD_MOMENTUM,
        weight_decay=C.BCCD_WEIGHT_DECAY,
    )
    total_steps = C.BCCD_EPOCHS * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=C.BCCD_MIN_LR)
    start_epoch = load_checkpoint(model, optimizer, scheduler)
    loss_fn = DetectionLoss(len(C.BCCD_CLASSES))

    history = []
    best_iou = -1
    if start_epoch == 0:
        visualize_samples(train_set, C.BCCD_VIS_DIR / "train_samples.png", max_samples=C.BCCD_PLOTS_SAMPLES)

    step_counter = 0
    for epoch in range(start_epoch, C.BCCD_EPOCHS):
        print(f"[Epoch {epoch+1}/{C.BCCD_EPOCHS}] starting")
        train_loss, logs = train_epoch(model, train_loader, optimizer, scheduler, device, anchors, loss_fn)
        val_mae, val_iou = evaluate(model, val_loader, device, anchors)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": train_loss, "mae": val_mae, "iou": val_iou})
        save_checkpoint(model, optimizer, scheduler, epoch)
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, optimizer, scheduler, epoch, best=True)
        print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} val_mae={val_mae:.4f} val_iou={val_iou:.4f}")
        step_counter += len(train_loader)
    plot_curves(history, C.BCCD_TRAIN_CURVE, C.BCCD_METRIC_IMG)

    model.load_state_dict(torch.load(C.BCCD_BEST_CKPT, map_location=device)["model"])
    test_mae, test_iou = evaluate(model, test_loader, device, anchors)
    print(f"[Test] MAE={test_mae:.4f} IoU={test_iou:.4f}")

    export_for_quant(model)
    compare_quantization(model, test_loader, device, anchors)


def group_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, groups: int, eps: float) -> np.ndarray:
    N, C, H, W = x.shape
    x = x.reshape(N, groups, C // groups, H, W)
    mean = x.mean(axis=(2, 3, 4), keepdims=True)
    var = x.var(axis=(2, 3, 4), keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, H, W)
    return x_norm * weight.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)


def compare_quantization(model: YoloTiny, loader, device, anchors):
    model.eval()
    kernels, strides, paddings, gn_meta = model.export_kernels()
    C.set_kernels(kernels)
    C.set_kernel_metadata(strides, paddings, gn_meta)

    setup_quant(num_layers=len(kernels))

    # calibration
    cal_imgs = []
    for idx, (imgs, _, _) in enumerate(loader):
        if idx >= C.NUM_CALIBRATION:
            break
        cal_imgs.append(imgs.numpy())
    calibrate_ddfp(cal_imgs, gn_meta)

    fp_preds_all = []
    base_preds_all = []
    dd_preds_all = []
    gts_all = []

    for imgs, boxes, labels in loader:
        imgs_np = imgs.numpy()
        fp_out = run_network_fp32(imgs_np, kernels, gn_meta)
        base_out, _, _ = run_network_baseline(imgs_np, gn_meta)
        dd_out, _, _, _ = run_network_ddfp(imgs_np, gn_meta)

        fp_preds = decode_predictions(torch.tensor(fp_out), anchors, C.BCCD_SCORE_THRESH)
        base_preds = decode_predictions(torch.tensor(base_out), anchors, C.BCCD_SCORE_THRESH)
        dd_preds = decode_predictions(torch.tensor(dd_out), anchors, C.BCCD_SCORE_THRESH)
        fp_preds_all.extend(fp_preds)
        base_preds_all.extend(base_preds)
        dd_preds_all.extend(dd_preds)
        gts_all.extend([(b, l) for b, l in zip(boxes, labels)])

    metrics_fp = compute_metrics([(b, c) for b, c, _ in fp_preds_all], gts_all)
    metrics_base = compute_metrics([(b, c) for b, c, _ in base_preds_all], gts_all)
    metrics_dd = compute_metrics([(b, c) for b, c, _ in dd_preds_all], gts_all)

    print(f"[Quant] FP32 MAE={metrics_fp[0]:.4f} IoU={metrics_fp[1]:.4f}")
    print(f"[Quant] Baseline MAE={metrics_base[0]:.4f} IoU={metrics_base[1]:.4f}")
    print(f"[Quant] DDFP MAE={metrics_dd[0]:.4f} IoU={metrics_dd[1]:.4f}")

    save_tracking(fp_preds_all, gts_all, C.BCCD_FP32_TRACKING)
    save_tracking(base_preds_all, gts_all, C.BCCD_BASELINE_TRACKING)
    save_tracking(dd_preds_all, gts_all, C.BCCD_DDFP_TRACKING)

    plot_quant_curves(metrics_fp, metrics_base, metrics_dd)
    visualize_comparison(loader, fp_preds_all, base_preds_all, dd_preds_all)


def plot_quant_curves(fp, base, dd):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    labels = ["FP32", "Baseline", "DDFP"]
    mae = [fp[0], base[0], dd[0]]
    iou = [fp[1], base[1], dd[1]]
    ax.plot(labels, mae, label="MAE")
    ax.plot(labels, iou, label="IoU")
    ax.set_title("Quantization comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.BCCD_QUANT_FIG, dpi=150)
    plt.close(fig)


def visualize_comparison(loader, fp_preds, base_preds, dd_preds):
    C.BCCD_VIS_DIR.mkdir(parents=True, exist_ok=True)
    for idx, (batch, fp, ba, dd) in enumerate(zip(loader, fp_preds, base_preds, dd_preds)):
        if idx >= C.BCCD_PLOTS_SAMPLES:
            break
        imgs, gt_boxes, gt_labels = batch
        img = imgs[0].permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img)
        draw_pred(ax, fp, "green", "FP32")
        draw_pred(ax, ba, "blue", "Baseline")
        draw_pred(ax, dd, "red", "DDFP")
        for box, cls in zip(gt_boxes[0], gt_labels[0]):
            if cls < 0:
                continue
            x1, y1, x2, y2 = (box * C.IMAGE_SIZE).numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="yellow", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1, C.BCCD_CLASSES[int(cls)], color="yellow")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(C.BCCD_VIS_DIR / f"compare_{idx}.png", dpi=150)
        plt.close(fig)


def draw_pred(ax, pred, color, label):
    boxes, cls, scores = pred
    for b, c, s in zip(boxes, cls, scores):
        x1, y1, x2, y2 = (b * C.IMAGE_SIZE).numpy()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1, f"{label}-{C.BCCD_CLASSES[int(c)]}:{s:.2f}", color=color, fontsize=6)


def save_tracking(preds, gts, path: Path):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_idx", "gt", "pred", "mae", "iou", "snr"])
        for idx, (pred, gt) in enumerate(zip(preds, gts)):
            pred_box = pred[0][0].cpu().numpy() * C.IMAGE_SIZE if pred[0].numel() > 0 else np.zeros(4)
            gt_box = gt[0][0].numpy() * C.IMAGE_SIZE if gt[0].numel() > 0 else np.zeros(4)
            mae = np.mean(np.abs(pred_box - gt_box))
            iou = float(box_iou_single(torch.tensor(pred_box / C.IMAGE_SIZE), torch.tensor(gt_box / C.IMAGE_SIZE)))
            snr = snr_db(pred_box, gt_box)
            writer.writerow([idx, gt_box.tolist(), pred_box.tolist(), mae, iou, snr])
    print(f"[Saved] tracking to {path}")


def setup_quant(num_layers: int):
    C.setup_config(num_layers, C.INPUT_BITS_LIST[0], C.WEIGHT_BITS_LIST[0], C.ADC_BITS_LIST[0])


if __name__ == "__main__":
    run_training()
