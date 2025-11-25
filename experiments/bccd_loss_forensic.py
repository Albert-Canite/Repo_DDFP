import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

import core.config as C
from experiments import bccd_yolotiny as bccd


def _layer_norm(layer: nn.Parameter) -> float:
    if layer.grad is None:
        return 0.0
    return float(layer.grad.data.norm(2).item())


def _param_delta(before: torch.Tensor, after: torch.Tensor) -> float:
    return float((after - before).pow(2).sum().sqrt().item())


def _tensor_stats(name: str, tensor: torch.Tensor, max_items: int = 10):
    flat = tensor.flatten()
    subset = flat[: max_items].detach().cpu().numpy().tolist()
    print(f"{name} shape={tuple(tensor.shape)} first_{len(subset)}={subset}")


def _legacy_build_targets(boxes, labels, anchors, grid_size, num_classes, ignore_thresh=0.5):
    device = boxes.device
    bs = boxes.shape[0]
    num_anchors = len(anchors)
    targets = torch.zeros(bs, num_anchors, grid_size, grid_size, 5 + num_classes, device=device)
    ignore_mask = torch.zeros(bs, num_anchors, grid_size, grid_size, device=device)
    anchor_tensor = torch.tensor(anchors, device=device, dtype=torch.float32)

    for b in range(bs):
        for box, cls in zip(boxes[b], labels[b]):
            if box.numel() == 0 or cls < 0:
                continue
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            bw = (box[2] - box[0])
            bh = (box[3] - box[1])
            if bw <= 0 or bh <= 0:
                continue
            gi = int(cx * grid_size)
            gj = int(cy * grid_size)
            if gi >= grid_size or gj >= grid_size:
                continue
            box_wh = torch.tensor([bw * C.IMAGE_SIZE, bh * C.IMAGE_SIZE], device=device)
            anchor_wh = anchor_tensor
            iou = bccd.bbox_wh_iou(box_wh[None], anchor_wh).squeeze(0)
            order = torch.argsort(iou, descending=True)
            placed = False
            for idx_anchor in order[: min(3, num_anchors)]:
                if targets[b, idx_anchor, gj, gi, 4] == 0:
                    targets[b, idx_anchor, gj, gi, 0:4] = torch.tensor(
                        [cx, cy, bw, bh], device=device
                    )
                    targets[b, idx_anchor, gj, gi, 4] = 1.0
                    targets[b, idx_anchor, gj, gi, 5 + cls] = 1.0
                    placed = True
                    break
            if placed:
                ignore_mask[b, :, gj, gi] = torch.where(iou > ignore_thresh, 1.0, ignore_mask[b, :, gj, gi])
    return targets, ignore_mask


def _assignment_stats(name, targets, ignore_mask, boxes, labels, anchors, show_map: bool = False):
    obj_mask = targets[..., 4]
    noobj_mask = (1.0 - obj_mask) * (1.0 - ignore_mask)
    grid = targets.shape[2]
    grid_cells = float(grid * grid)
    print(f"===== {name} assignment stats =====")
    for b in range(targets.shape[0]):
        valid = (labels[b] >= 0) & ((boxes[b].sum(dim=1)) > 0)
        gt_count = int(valid.sum().item())
        pos = int(obj_mask[b].sum().item())
        noobj = int((noobj_mask[b] > 0).sum().item())
        ign = int((ignore_mask[b] > 0).sum().item())
        ratio = (pos / grid_cells) * 100.0
        print(
            f"img{b}: gt={gt_count} obj_mask={pos} noobj_mask={noobj} ignore_mask={ign} pos/grid={ratio:.2f}%"
        )

    if show_map:
        for b in range(targets.shape[0]):
            valid_idx = ((labels[b] >= 0) & ((boxes[b].sum(dim=1)) > 0)).nonzero(as_tuple=False)
            if valid_idx.numel() == 0:
                continue
            idx = int(valid_idx[0])
            box = boxes[b, idx]
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            bw = (box[2] - box[0])
            bh = (box[3] - box[1])
            gi = int(cx * grid)
            gj = int(cy * grid)
            box_wh = torch.tensor([bw * C.IMAGE_SIZE, bh * C.IMAGE_SIZE], device=targets.device)
            anchor_wh = torch.tensor(anchors, device=targets.device, dtype=torch.float32)
            iou = bccd.bbox_wh_iou(box_wh[None], anchor_wh).squeeze(0)
            best_anchor = int(torch.argmax(iou).item())
            pos_map = obj_mask[b, best_anchor].int().cpu().numpy()
            print(f"Positive map for img{b} GT#{idx} (anchor={best_anchor}, cell=({gj},{gi})):")
            print(pos_map)
            break


def _print_objectness_distribution(scores: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor):
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    for label, vals in [("pos", pos_scores), ("neg", neg_scores)]:
        if vals.numel() == 0:
            print(f"objectness {label}: empty")
            continue
        stats = torch.quantile(vals, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=vals.device))
        print(
            f"objectness {label}: count={vals.numel()} min={stats[0].item():.4f} q25={stats[1].item():.4f} "
            f"median={stats[2].item():.4f} q75={stats[3].item():.4f} max={stats[4].item():.4f}"
        )


def _find_best_matches(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor):
    from experiments.bccd_yolotiny import box_iou_single

    # Ensure both tensors are on the same device for fair IoU / error computation
    device = pred_boxes.device
    gt_boxes = gt_boxes.to(device)

    results = []
    for gt in gt_boxes:
        if gt.numel() == 0:
            continue
        if pred_boxes.numel() == 0:
            results.append((0.0, float("nan"), float("nan")))
            continue
        ious = [box_iou_single(pb, gt) for pb in pred_boxes]
        ious_t = torch.stack(ious)
        best_idx = int(torch.argmax(ious_t))
        best_iou = float(ious_t[best_idx])
        best_pred = pred_boxes[best_idx]
        center_err = torch.abs((best_pred[:2] + best_pred[2:]) / 2 - (gt[:2] + gt[2:]) / 2).mean()
        size_err = torch.abs((best_pred[2:] - best_pred[:2]) - (gt[2:] - gt[:2])).mean()
        results.append((best_iou, float(center_err), float(size_err)))
    return results


def _match_gt_to_targets(gt: torch.Tensor, targets: torch.Tensor) -> List[Tuple[int, int, int]]:
    matches: List[Tuple[int, int, int]] = []
    pos = (targets[..., 4] > 0).nonzero(as_tuple=False)
    gt = gt.to(targets.device)
    for idx in pos:
        a, j, i = idx.tolist()
        t = targets[a, j, i, 0:4]
        if torch.allclose(t, gt, atol=1e-4):
            matches.append((a, j, i))
    return matches


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, _, anchors = bccd.prepare_dataloaders()
    anchors = anchors if anchors else C.BCCD_ANCHORS

    model = bccd.YoloTiny(len(C.BCCD_CLASSES), anchors).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=C.BCCD_LR, momentum=C.BCCD_MOMENTUM, weight_decay=C.BCCD_WEIGHT_DECAY
    )
    if Path(C.BCCD_BEST_CKPT).exists():
        ckpt = torch.load(C.BCCD_BEST_CKPT, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[Load] best checkpoint loaded from {C.BCCD_BEST_CKPT}")
    elif Path(C.BCCD_CKPT).exists():
        ckpt = torch.load(C.BCCD_CKPT, map_location=device)
        model.load_state_dict(ckpt.get("model", {}), strict=False)
        print(f"[Load] last checkpoint loaded from {C.BCCD_CKPT}")
    else:
        print("[Load] no checkpoint found; using randomly initialized weights")

    loss_fn = bccd.DetectionLoss(len(C.BCCD_CLASSES))

    imgs, boxes, labels = next(iter(train_loader))
    imgs = imgs.to(device)
    boxes = boxes.to(device)
    labels = labels.to(device)

    raw = model(imgs)
    pred_boxes, pred_obj, pred_cls, grid = bccd._decode_raw(raw, anchors)
    targets, ignore_mask = bccd.build_targets(
        boxes, labels, anchors, grid, len(C.BCCD_CLASSES), ignore_thresh=C.BCCD_IGNORE_IOU
    )
    legacy_targets, legacy_ignore = _legacy_build_targets(
        boxes, labels, anchors, grid, len(C.BCCD_CLASSES), ignore_thresh=C.BCCD_IGNORE_IOU
    )

    _assignment_stats("Legacy (pre-fix)", legacy_targets, legacy_ignore, boxes, labels, anchors, show_map=False)
    _assignment_stats("Fixed (best-anchor)", targets, ignore_mask, boxes, labels, anchors, show_map=True)
    preds = torch.cat([pred_boxes, pred_obj, pred_cls], dim=-1)

    loss, detail = loss_fn(preds, targets, ignore_mask)
    obj_mask = targets[..., 4:5]
    pos_mask = obj_mask.squeeze(-1) > 0
    noobj_mask = (1.0 - obj_mask) * (1.0 - ignore_mask.unsqueeze(-1))

    print("===== Loss breakdown =====")
    print(f"total_loss={loss.item():.6f} box={detail['box']:.6f} obj={detail['obj']:.6f} "
          f"noobj={detail['noobj']:.6f} cls={detail['cls']:.6f} ciou_mean={detail['ciou']:.6f}")
    print(
        f"weights: box={C.BCCD_BOX_LOSS_WEIGHT} obj={C.BCCD_OBJ_LOSS_WEIGHT} "
        f"cls={C.BCCD_CLS_LOSS_WEIGHT} noobj_scale=0.5"
    )
    print(f"targets shape={targets.shape} obj_mask positives={pos_mask.sum().item()} "
          f"noobj_mask count={(noobj_mask > 0).sum().item()} ignore_mask count={(ignore_mask > 0).sum().item()}")

    if pos_mask.any():
        pos_preds = pred_boxes[pos_mask]
        pos_tgts = targets[..., 0:4][pos_mask]
        _tensor_stats("pred_boxes_pos", pos_preds)
        _tensor_stats("target_boxes_pos", pos_tgts)

        # Compare IoU used in loss (cxcywh CIoU) vs. IoU in xyxy eval space for the first positive sample
        ciou_used = bccd.bbox_ciou(pos_preds[:1], pos_tgts[:1]).squeeze().item()
        first_pred_xyxy = bccd._cxcywh_to_xyxy(pos_preds[:1]).squeeze(0)
        first_tgt_xyxy = bccd._cxcywh_to_xyxy(pos_tgts[:1]).squeeze(0)
        iou_xyxy = float(bccd.box_iou_single(first_pred_xyxy, first_tgt_xyxy))
        print(
            f"first_pos_ciou_current={ciou_used:.6f} (cxcywh space) iou_xyxy_eval_space={iou_xyxy:.6f}"
        )
        print(f"first_pred_xyxy={first_pred_xyxy.tolist()} first_tgt_xyxy={first_tgt_xyxy.tolist()}")
    else:
        print("no positive samples in this batch for regression loss")

    before = {
        "backbone0.conv.weight": model.backbone[0].conv.weight.detach().clone(),
        "backbone3.conv.weight": model.backbone[3].conv.weight.detach().clone(),
        "head.weight": model.head.weight.detach().clone(),
        "head.bias": model.head.bias.detach().clone(),
    }

    loss.backward()

    print("===== Gradient norms after backward =====")
    for name, param in [
        ("backbone0.conv.weight", model.backbone[0].conv.weight),
        ("backbone3.conv.weight", model.backbone[3].conv.weight),
        ("head.weight", model.head.weight),
        ("head.bias", model.head.bias),
    ]:
        print(f"{name} grad_norm={_layer_norm(param):.6f}")

    optimizer.step()

    print("===== Parameter deltas after optimizer step =====")
    for name, param in [
        ("backbone0.conv.weight", model.backbone[0].conv.weight),
        ("backbone3.conv.weight", model.backbone[3].conv.weight),
        ("head.weight", model.head.weight),
        ("head.bias", model.head.bias),
    ]:
        delta = _param_delta(before[name], param.detach())
        print(f"{name} delta_L2={delta:.6f}")

    model.eval()
    with torch.no_grad():
        imgs_val, boxes_val, labels_val = next(iter(val_loader))
        imgs_val = imgs_val.to(device)
        boxes_val = boxes_val.to(device)
        labels_val = labels_val.to(device)
        raw_val = model(imgs_val)
        boxes_dec, obj_logit, cls_logit, grid_val = bccd._decode_raw(raw_val, anchors)
        targets_val, ignore_val = bccd.build_targets(
            boxes_val, labels_val, anchors, grid_val, len(C.BCCD_CLASSES), ignore_thresh=C.BCCD_IGNORE_IOU
        )
        obj_scores = torch.sigmoid(obj_logit.squeeze(-1))
        pos_mask_val = targets_val[..., 4] > 0
        neg_mask_val = (targets_val[..., 4] == 0) & (ignore_val == 0)
        print("===== Objectness score distribution (validation) =====")
        _print_objectness_distribution(obj_scores, pos_mask_val, neg_mask_val)

        boxes_xyxy = bccd._cxcywh_to_xyxy(boxes_dec).clamp(0.0, 1.0)
        cls_scores = torch.sigmoid(cls_logit)
        scores, cls_idx = torch.max(cls_scores, dim=-1)
        scores = scores * torch.sigmoid(obj_logit.squeeze(-1))
        mask = scores > C.BCCD_SCORE_THRESH
        print(
            f"viz/eval thresholds: score_thresh={C.BCCD_SCORE_THRESH} nms_iou={C.BCCD_NMS_IOU} "
            f"decode_used_for_eval_and_viz=decode_predictions"
        )
        kept_before_nms = mask.sum(dim=(1, 2, 3)).cpu().tolist()
        outputs = bccd.decode_predictions(raw_val.cpu(), anchors, C.BCCD_SCORE_THRESH)
        kept_after_nms = [o[0].shape[0] for o in outputs]
        print(f"boxes kept per image before_nms={kept_before_nms} after_nms={kept_after_nms}")

        print("===== GT vs prediction sanity check (validation) =====")
        for b in range(min(1, boxes_val.shape[0])):
            gt_valid = (labels_val[b] >= 0) & (boxes_val[b].sum(dim=1) > 0)
            gt_list = boxes_val[b][gt_valid][:10]
            preds_b = outputs[b][0]
            matches = _find_best_matches(gt_list, preds_b)
            for idx, (gt_box, match) in enumerate(zip(gt_list, matches)):
                best_iou, center_err, size_err = match
                gt_targets = _match_gt_to_targets(gt_box, targets_val[b])
                print(
                    f"GT[{idx}]={gt_box.tolist()} matches={gt_targets} best_pred_iou={best_iou:.4f} "
                    f"center_err={center_err:.4f} size_err={size_err:.4f} supervised={bool(gt_targets)}"
                )


if __name__ == "__main__":
    main()
