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
