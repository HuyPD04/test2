from __future__ import annotations

import numpy as np
import torch
import torchvision.ops

from rl_sahi.common.box_types import as_boxes


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, nms_type: str = "standard") -> np.ndarray:
    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    scores_t = torch.as_tensor(scores, dtype=torch.float32, device=device)

    nms_name = str(nms_type).lower().replace("-", "_")
    if nms_name in {"standard", "iou", "nms"}:
        keep_t = torchvision.ops.nms(boxes_t, scores_t, iou_threshold)
    elif nms_name == "diou":
        keep_t = _diou_nms_pytorch(boxes_t, scores_t, iou_threshold)
    elif nms_name in {"cdn", "cluster_diou", "cluster_diou_nms"}:
        keep_t = _cluster_diou_nms_pytorch(boxes_t, scores_t, iou_threshold)
    else:
        raise ValueError(
            f"Unsupported nms_type={nms_type!r}. Expected standard, diou, or cdn."
        )

    return keep_t.cpu().numpy().astype(np.int64)


def _diou_nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Distance-IoU NMS implementation."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort(descending=True)
    keep = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            i = order[0]
            keep.append(i)
            break
            
        i = order[0]
        keep.append(i)
        
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        
        # Convex diagonal length squared
        cw = torch.maximum(x2[i], x2[order[1:]]) - torch.minimum(x1[i], x1[order[1:]])
        ch = torch.maximum(y2[i], y2[order[1:]]) - torch.minimum(y1[i], y1[order[1:]])
        c2 = cw ** 2 + ch ** 2 + 1e-7
        
        # Center distance squared
        center_x1 = (x1[i] + x2[i]) / 2
        center_y1 = (y1[i] + y2[i]) / 2
        center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
        center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
        rho2 = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2
        
        # DIoU = IoU - rho2 / c2
        # We suppress if DIoU > iou_threshold, which means we KEEP if DIoU <= iou_threshold
        # Note: DIoU paper originally suppresses if (IoU - R_DIoU) >= threshold
        # This is equivalent to removing boxes with diou_val > threshold.
        diou_val = iou - rho2 / c2
        inds = torch.where(diou_val <= iou_threshold)[0]
        
        order = order[inds + 1]
        
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def _pairwise_diou_pytorch(boxes: torch.Tensor) -> torch.Tensor:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = torch.clamp(x2 - x1, min=0.0) * torch.clamp(y2 - y1, min=0.0)

    inter_x1 = torch.maximum(x1[:, None], x1[None, :])
    inter_y1 = torch.maximum(y1[:, None], y1[None, :])
    inter_x2 = torch.minimum(x2[:, None], x2[None, :])
    inter_y2 = torch.minimum(y2[:, None], y2[None, :])
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter = inter_w * inter_h

    union = areas[:, None] + areas[None, :] - inter
    iou = inter / torch.clamp(union, min=1e-7)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    rho2 = (center_x[:, None] - center_x[None, :]) ** 2 + (
        center_y[:, None] - center_y[None, :]
    ) ** 2

    convex_w = torch.maximum(x2[:, None], x2[None, :]) - torch.minimum(x1[:, None], x1[None, :])
    convex_h = torch.maximum(y2[:, None], y2[None, :]) - torch.minimum(y1[:, None], y1[None, :])
    c2 = convex_w ** 2 + convex_h ** 2
    return iou - rho2 / torch.clamp(c2, min=1e-7)


def _cluster_diou_nms_pytorch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    max_iterations: int = 200,
) -> torch.Tensor:
    """Cluster-DIoU-NMS.

    Boxes are sorted by score, then an upper-triangular DIoU affinity matrix is
    iteratively pruned so suppressed boxes do not suppress lower-scored boxes.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.int64, device=boxes.device)
    order = scores.argsort(descending=True)
    if order.numel() == 1:
        return order

    sorted_boxes = boxes[order]
    diou = _pairwise_diou_pytorch(sorted_boxes).triu_(diagonal=1)
    clustered = diou
    max_overlap = clustered.max(dim=0).values

    for _ in range(int(max_iterations)):
        previous = max_overlap
        keep_mask = (max_overlap <= float(iou_threshold)).to(dtype=diou.dtype)
        clustered = diou * keep_mask[:, None]
        max_overlap = clustered.max(dim=0).values
        if torch.equal(previous <= float(iou_threshold), max_overlap <= float(iou_threshold)):
            break

    return order[max_overlap <= float(iou_threshold)]
