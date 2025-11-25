"""Diagnostics for BCCD YOLO-tiny without external dependencies."""

import csv
import math
import random
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

IMAGE_SIZE = 512
BCCD_CLASSES = ["RBC", "WBC", "Platelets"]
BCCD_ANCHORS = [(12, 12), (24, 24), (36, 36)]
BCCD_DIR = ROOT_DIR / "archive"
BCCD_IMG_DIR = BCCD_DIR / "images"
BCCD_ANNO_CSV = BCCD_DIR / "annotations.csv"
BCCD_IGNORE_IOU = 0.5


# -----------------------------------------------------------------------------
# Data loading helpers (CSV + PNG header parsing only)
# -----------------------------------------------------------------------------

def _read_png_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    width = int.from_bytes(header[16:20], byteorder="big")
    height = int.from_bytes(header[20:24], byteorder="big")
    return width, height


def _load_annotations() -> List[Dict]:
    mapping = {c.lower(): i for i, c in enumerate(BCCD_CLASSES)}
    samples: Dict[str, Dict] = {}
    with BCCD_ANNO_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().lower()
            if label not in mapping:
                continue
            x1, y1, x2, y2 = (
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
            )
            entry = samples.setdefault(row["image"], {"boxes": [], "labels": [], "size": None})
            entry["boxes"].append([x1, y1, x2, y2])
            entry["labels"].append(mapping[label])

    for fname, entry in samples.items():
        img_path = BCCD_IMG_DIR / fname
        entry["size"] = _read_png_size(img_path)

    return [
        {
            "image": BCCD_IMG_DIR / fname,
            "boxes": entry["boxes"],
            "labels": entry["labels"],
            "size": entry["size"],
        }
        for fname, entry in samples.items()
    ]


def _normalize_boxes(boxes: List[List[float]], size: Tuple[int, int]) -> List[List[float]]:
    w, h = size
    return [[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in boxes]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _bbox_wh_iou(box1: Tuple[float, float], anchors: List[Tuple[float, float]]) -> List[float]:
    ious = []
    b1w, b1h = box1
    for aw, ah in anchors:
        inter = min(b1w, aw) * min(b1h, ah)
        union = b1w * b1h + aw * ah - inter + 1e-8
        ious.append(inter / union)
    return ious


# -----------------------------------------------------------------------------
# Audits
# -----------------------------------------------------------------------------

def audit_decode_math(anchors: List[Tuple[float, float]]):
    print("===== Decode math correctness (20 samples) =====")
    grid = IMAGE_SIZE // 16  # stride-16 output grid (32 for 512 input)
    stride = IMAGE_SIZE / float(grid)
    rng = random.Random(0)

    # Generate deterministic raw outputs
    raw = [
        [
            [
                (
                    rng.uniform(-2, 2),
                    rng.uniform(-2, 2),
                    rng.uniform(-2, 2),
                    rng.uniform(-2, 2),
                )
                for _ in range(grid)
            ]
            for _ in range(grid)
        ]
        for _ in anchors
    ]

    flat_indices = list(range(len(anchors) * grid * grid))
    rng.shuffle(flat_indices)
    flat_indices = flat_indices[:20]

    for idx, flat in enumerate(flat_indices, 1):
        a = flat // (grid * grid)
        rem = flat % (grid * grid)
        gy = rem // grid
        gx = rem % grid
        tx, ty, tw, th = raw[a][gy][gx]
        sig_tx, sig_ty = _sigmoid(tx), _sigmoid(ty)
        exp_tw, exp_th = math.exp(tw), math.exp(th)
        center_no_grid = (sig_tx, sig_ty)
        center_with_grid = ((sig_tx + gx) / grid, (sig_ty + gy) / grid)
        bw = (exp_tw * anchors[a][0]) / IMAGE_SIZE
        bh = (exp_th * anchors[a][1]) / IMAGE_SIZE
        x1 = center_with_grid[0] - bw / 2
        y1 = center_with_grid[1] - bh / 2
        x2 = center_with_grid[0] + bw / 2
        y2 = center_with_grid[1] + bh / 2
        print(
            f"[{idx:02d}] anchor={a} cell=({gy},{gx}) raw(tx,ty,tw,th)=({tx:.6f},{ty:.6f},{tw:.6f},{th:.6f}) "
            f"sigmoid(tx,ty)=({sig_tx:.6f},{sig_ty:.6f}) grid_x={gx} grid_y={gy} stride={stride:.4f} "
            f"center_no_grid=({center_no_grid[0]:.6f},{center_no_grid[1]:.6f}) center_with_grid=({center_with_grid[0]:.6f},{center_with_grid[1]:.6f}) "
            f"wh_no_anchor=(exp_tw={exp_tw:.6f},exp_th={exp_th:.6f}) anchor_w={anchors[a][0]:.6f} anchor_h={anchors[a][1]:.6f} "
            f"final_xyxy=({x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f})"
        )


def audit_anchor_assignment(samples: List[Dict], anchors: List[Tuple[float, float]]):
    print("\n===== Anchor assignment (10 GT boxes) =====")
    grid = IMAGE_SIZE // 16
    total_cells = len(anchors) * grid * grid
    total_pos = 0
    collected = 0
    rng = random.Random(0)

    for sample in rng.sample(samples, k=len(samples)):
        if collected >= 10:
            break
        norm_boxes = _normalize_boxes(sample["boxes"], sample["size"])
        targets = [[[ [0.0] * (5 + len(BCCD_CLASSES)) for _ in range(grid)] for _ in range(grid)] for _ in anchors]

        for box, cls in zip(norm_boxes, sample["labels"]):
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            bw = (box[2] - box[0]) * IMAGE_SIZE
            bh = (box[3] - box[1]) * IMAGE_SIZE
            gi = int(cx * grid)
            gj = int(cy * grid)
            ious = _bbox_wh_iou((bw, bh), anchors)
            order = sorted(range(len(anchors)), key=lambda i: ious[i], reverse=True)
            placed = False
            for idx_anchor in order[: min(3, len(anchors))]:
                if targets[idx_anchor][gj][gi][4] == 0:
                    targets[idx_anchor][gj][gi][0:4] = [cx, cy, bw / IMAGE_SIZE, bh / IMAGE_SIZE]
                    targets[idx_anchor][gj][gi][4] = 1.0
                    targets[idx_anchor][gj][gi][5 + cls] = 1.0
                    placed = True
                    break
            if collected < 10:
                pos_list = [i for i in range(len(anchors)) if targets[i][gj][gi][4] > 0]
                print(
                    f"GT[{collected:02d}] file={sample['image'].name} cls={cls} raw_box={box} grid_cell=({gj},{gi}) "
                    f"IoU_vs_anchors={[round(v, 4) for v in ious]} positives={pos_list}"
                )
                collected += 1
            if placed:
                total_pos += 1
            if collected >= 10:
                break

    ratio = total_pos / float(total_cells)
    status = "ERROR" if ratio > 0.2 else "OK"
    print(f"pos/grid ratio={ratio:.4f} ({status})")
    if ratio > 0.2:
        print("[Error] too many positive anchors relative to grid cells, assignments are overly dense.")


def audit_gt_pipeline(samples: List[Dict], anchors: List[Tuple[float, float]]):
    print("\n===== GT coordinate pipeline =====")
    sample = next((s for s in samples if s["boxes"]), None)
    if sample is None:
        print("No GT boxes found in annotations.")
        return

    raw_box = sample["boxes"][0]
    w, h = sample["size"]
    norm_box = _normalize_boxes([raw_box], sample["size"])[0]
    resized_box = [v * IMAGE_SIZE for v in norm_box]
    cxcywh_norm = [
        0.5 * (norm_box[0] + norm_box[2]),
        0.5 * (norm_box[1] + norm_box[3]),
        norm_box[2] - norm_box[0],
        norm_box[3] - norm_box[1],
    ]
    grid = IMAGE_SIZE // 16
    gi = int(cxcywh_norm[0] * grid)
    gj = int(cxcywh_norm[1] * grid)

    targets = [[[ [0.0] * (5 + len(BCCD_CLASSES)) for _ in range(grid)] for _ in range(grid)] for _ in anchors]
    ious = _bbox_wh_iou((cxcywh_norm[2] * IMAGE_SIZE, cxcywh_norm[3] * IMAGE_SIZE), anchors)
    order = sorted(range(len(anchors)), key=lambda i: ious[i], reverse=True)
    selected_anchor = order[0]
    targets[selected_anchor][gj][gi][0:4] = cxcywh_norm
    targets[selected_anchor][gj][gi][4] = 1.0
    targets[selected_anchor][gj][gi][5 + sample["labels"][0]] = 1.0
    target_vals = targets[selected_anchor][gj][gi][0:4]

    print(f"raw GT (annotation pixels)={raw_box} image_size=({w},{h})")
    print(f"after normalization (0-1)={norm_box}")
    print(f"after resize to {IMAGE_SIZE} (pixels)={resized_box}")
    print(f"cxcywh normalized={cxcywh_norm}")
    print(f"cxcywh after resize(normalized same grid units)={cxcywh_norm}")
    print(f"grid size={grid} assigned cell=({gj},{gi}) anchor={selected_anchor}")
    print(f"GT used for assignment (targets tensor)={target_vals}")
    print(f"GT used for loss (same as targets)={target_vals}")


def main():
    random.seed(0)
    samples = _load_annotations()
    if not samples:
        print("No samples found for auditing.")
        return
    anchors = BCCD_ANCHORS
    audit_decode_math(anchors)
    audit_anchor_assignment(samples, anchors)
    audit_gt_pipeline(samples, anchors)


if __name__ == "__main__":
    main()
