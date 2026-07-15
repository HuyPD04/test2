from __future__ import annotations

import numpy as np
import torch
import torchvision.ops

from rl_sahi.common.box_types import as_boxes

def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    scores_t = torch.as_tensor(scores, dtype=torch.float32, device=device)
    
    keep_t = torchvision.ops.nms(boxes_t, scores_t, iou_threshold)
    return keep_t.cpu().numpy().astype(np.int64)
