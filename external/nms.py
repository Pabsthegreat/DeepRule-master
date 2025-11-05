# external/nms.py  — Python 3.13+ friendly shim using torchvision
import torch
from torchvision.ops import nms as _tv_nms

def nms(boxes, scores, iou=0.5, method='union'):
    """
    boxes: Tensor[N,4] or array-like (x1,y1,x2,y2)
    scores: Tensor[N] or array-like
    iou: IoU threshold
    """
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    keep = _tv_nms(boxes, scores, iou)
    return keep

# Optional: soft_nms fallback; behave like vanilla NMS
def soft_nms(boxes, scores, iou=0.5, sigma=0.5, Nt=0.5, threshold=0.001, method=0):
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    keep = _tv_nms(boxes, scores, iou)
    return boxes, scores, keep
