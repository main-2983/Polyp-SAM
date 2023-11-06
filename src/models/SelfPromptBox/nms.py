import torch 
import time
import torchvision
from ultralytics.utils import LOGGER
import numpy as np
from torchvision.ops.boxes import box_iou



def non_max_suppression(bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    order = torch.argsort(-scores)
    indices = torch.arange(bboxes.shape[0])
    keep = torch.ones_like(indices, dtype=torch.bool).cuda()
    for i in indices:
        if keep[i]:
            bbox = bboxes[order[i]]
            iou = box_iou(bbox[None,...],(bboxes[order[i + 1:]]) * keep[i + 1:][...,None])
            overlapped = torch.nonzero(iou > iou_threshold)
            keep[overlapped + i + 1] = 0
    return order[keep]


# def non_max_suppression(
#         prediction,
#         boxes,
#         scores,
#         conf_thres=0.25,
#         iou_thres=0.45,
#         classes=None,
#         agnostic=False,
#         multi_label=False,
#         labels=(),
#         max_det=300,
#         nc=0,  # number of classes (optional)
#         max_time_img=0.05,
#         max_nms=30000,
#         max_wh=7680,
# ):
#     """
#     Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

#     Arguments:
#         prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
#             containing the predicted boxes, classes, and masks. The tensor should be in the format
#             output by a model, such as YOLO.
#         conf_thres (float): The confidence threshold below which boxes will be filtered out.
#             Valid values are between 0.0 and 1.0.
#         iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
#             Valid values are between 0.0 and 1.0.
#         classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
#         agnostic (bool): If True, the model is agnostic to the number of classes, and all
#             classes will be considered as one.
#         multi_label (bool): If True, each box may have multiple labels.
#         labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
#             list contains the apriori labels for a given image. The list should be in the format
#             output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
#         max_det (int): The maximum number of boxes to keep after NMS.
#         nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
#         max_time_img (float): The maximum time (seconds) for processing one image.
#         max_nms (int): The maximum number of boxes into torchvision.ops.nms().
#         max_wh (int): The maximum box width and height in pixels

#     Returns:
#         (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
#             shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
#             (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
#     """

#     # Checks
#     assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
#     assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
#     if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
#         prediction = prediction[0]  # select only inference output

#     device = prediction.device
#     mps = 'mps' in device.type  # Apple MPS
#     if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
#         prediction = prediction.cpu()
#     bs = prediction.shape[0]  # batch size
#     nc = nc or (prediction.shape[1] - 4)  # number of classes
#     nm = prediction.shape[1] - nc - 4
#     mi = 4 + nc  # mask start index
#     xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

#     # Settings
#     # min_wh = 2  # (pixels) minimum box width and height
#     time_limit = 0.5 + max_time_img * bs  # seconds to quit after
#     redundant = True  # require redundant detections
#     multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
#     merge = False  # use merge-NMS

#     prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
#     prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

#     t = time.time()
#     output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         x = x[xc[xi]]  # confidence

#         # Cat apriori labels if autolabelling
#         if labels and len(labels[xi]):
#             lb = labels[xi]
#             v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
#             v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
#             v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
#             x = torch.cat((x, v), 0)

#         # If none remain process next image
#         if not x.shape[0]:
#             continue

#         # Detections matrix nx6 (xyxy, conf, cls)
#         box, cls, mask = x.split((4, nc, nm), 1)

#         if multi_label:
#             i, j = torch.where(cls > conf_thres)
#             x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
#         else:  # best class only
#             conf, j = cls.max(1, keepdim=True)
#             x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

#         # Filter by class
#         if classes is not None:
#             x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

#         # Apply finite constraint
#         # if not torch.isfinite(x).all():
#         #     x = x[torch.isfinite(x).all(1)]

#         # Check shape
#         n = x.shape[0]  # number of boxes
#         if not n:  # no boxes
#             continue
#         if n > max_nms:  # excess boxes
#             x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

#         # Batched NMS
#         c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#         boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
#         i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
#         i = i[:max_det]  # limit detections
#         if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
#             # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#             iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#             weights = iou * scores[None]  # box weights
#             x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
#             if redundant:
#                 i = i[iou.sum(1) > 1]  # require redundancy

#         output[xi] = x[i]
#         if mps:
#             output[xi] = output[xi].to(device)
#         if (time.time() - t) > time_limit:
#             LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
#             break  # time limit exceeded

#     return output

# def xywh2xyxy(x):
#     """
#     Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
#     top-left corner and (x2, y2) is the bottom-right corner.

#     Args:
#         x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
#     Returns:
#         y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
#     """
#     y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
#     dw = x[..., 2] / 2  # half-width
#     dh = x[..., 3] / 2  # half-height
#     y[..., 0] = x[..., 0] - dw  # top left x
#     y[..., 1] = x[..., 1] - dh  # top left y
#     y[..., 2] = x[..., 0] + dw  # bottom right x
#     y[..., 3] = x[..., 1] + dh  # bottom right y
#     return y

# def box_iou(box1, box2, eps=1e-7):
#     """
#     Calculate intersection-over-union (IoU) of boxes.
#     Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#     Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

#     Args:
#         box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
#         box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
#         eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

#     Returns:
#         (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
#     """

#     # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
#     (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
#     inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

#     # IoU = inter / (area1 + area2 - inter)
#     return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)