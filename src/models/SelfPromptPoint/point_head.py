from typing import Union, Tuple, Sequence

import torch
import torch.nn as nn

from segment_anything.modeling.common import LayerNorm2d

from src.models.assigner.point_generator import PointGenerator


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 norm: nn.Module = LayerNorm2d,
                 activation: nn.Module = nn.ReLU):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.norm = norm
        self.act = activation

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)

        return out


class PointHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_convs: int = 3,
                 stride: int = 16,
                 top_k: int = 1,
                 center_radius: float = 1.5):
        super(PointHead, self).__init__()
        self.obj_convs = []
        for i in range(num_convs):
            in_chn = in_channels if i == 0 else feat_channels
            self.obj_convs.append(
                ConvModule(in_chn,
                           feat_channels,
                           kernel_size=3,
                           padding=1,
                           norm=LayerNorm2d(feat_channels),
                           activation=nn.ReLU(inplace=True)))

        self.obj_convs = nn.Sequential(*self.obj_convs)

        self.obj_pred = nn.Conv2d(feat_channels, 1, 1)

        self.prior_generator = PointGenerator()
        self.stride = stride
        self.top_k = top_k
        self.center_radius = center_radius

    def forward(self, x):
        obj_feats = self.obj_convs(x)

        obj_pred = self.obj_pred(obj_feats)

        return obj_pred

    def decode_prediction(self,
                          obj_pred: torch.Tensor,
                          threshold: float = 0.5):
        """ This is a single image prediction, do not use it on batch size > 1"""
        _, _, H, W = obj_pred.shape
        device = obj_pred.device
        flatten_pred = obj_pred.flatten()
        flatten_pred = flatten_pred.sigmoid()
        positive_mask = torch.where(flatten_pred >= threshold, True, False)
        priors = self.prior_generator.grid_points((H, W), stride=self.stride, device=device) # (H * W, 2)
        positive_priors = priors[positive_mask] # (positive_points, 2)

        point_labels = torch.ones((1, positive_priors.shape[0]), dtype=torch.long, device=device)

        return positive_priors.unsqueeze(0),\
               point_labels # (1, positive_points, 2) -> Expand num_box dim

    def prepare_for_loss(self,
                         obj_pred: torch.Tensor,
                         gt_bboxes: Sequence[torch.Tensor]):
        """
        Calculate the loss based on the features extracted by the detection
        head.

        Args:
            obj_pred (Tensor): Objectness prediction, has shape (B, 1, H, W)
            gt_bboxes (Sequence[Tensor]): Ground truth bounding boxes. Each image is a list of Tensor of shape (num_box, 4)
        """
        num_imgs = len(gt_bboxes)
        device = obj_pred.device

        featmap_size = obj_pred.shape[-2:]
        priors = self.prior_generator.grid_points(
            featmap_size, self.stride, device
        ) # [num_priors, 2]
        points_targets = []
        num_positives = 0
        for gt_bbox in gt_bboxes:
            points_target, num_positive = self._get_target_single(
                priors, gt_bbox, stride=self.stride
            )
            points_targets.append(points_target)
            num_positives += num_positive
        points_targets = torch.cat(points_targets, 0)
        flatten_obj_preds = obj_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1).contiguous()
        flatten_obj_preds = flatten_obj_preds.view(-1, 1).contiguous()
        points_targets = points_targets.unsqueeze(1)
        return flatten_obj_preds, points_targets, num_positives

    def _get_target_single(self,
                           priors: torch.Tensor,
                           gt_bbox: torch.Tensor,
                           stride: int):
        """
        Label Assignment for a single image

        Args:
            priors (Tensor): prior of shape (num_priors, 2)
            gt_bbox(Tensor): bbox of a single image of shape (num_gts, 4)
        """
        num_priors = priors.shape[0]
        num_gts = gt_bbox.shape[0]

        # Label Assignment
        # Simple LA strat: if the prior is inside a bounding box -> positive
        # Assign 0 by default
        assigned_gt_inds = priors.new_full((num_priors, ), 0, dtype=torch.long)
        # Get positive mask
        positive_mask = self._is_in_bbox_and_center(
            priors, gt_bbox, stride, self.center_radius
        ) # (num_priors, num_gts)
        # Get top-k prior of each gt
        positive_mask, positive_inds = self._top_k_points(positive_mask, num_gts) # (num_priors, )
        # Assign 1 if positive
        assigned_gt_inds[positive_mask] = 1
        num_positives = sum(positive_mask)

        return assigned_gt_inds, num_positives # (num_priors, num_gts, 4),

    def _is_in_bbox_and_center(self,
                               priors,
                               gt_bbox,
                               stride,
                               center_radius=1.5):
        num_priors = priors.shape[0]
        num_gts = gt_bbox.shape[0]

        # Condition 1: Inside bbox
        xs, ys = priors[:, 0], priors[:, 1]
        xs = xs[:, None].expand(num_priors, num_gts)
        ys = ys[:, None].expand(num_priors, num_gts)
        left = xs - gt_bbox[..., 0]
        right = gt_bbox[..., 2] - xs
        top = ys - gt_bbox[..., 1]
        bottom = gt_bbox[..., 3] - ys
        bbox_deltas = torch.stack((left, top, right, bottom), -1) # (num_priors, num_gts, 4)

        is_in_bbox = bbox_deltas.min(-1)[0] > 0 # (num_priors, num_gts)

        # Condition 2: Inside the center region of object
        # 2.1: Find the center of gt bbox
        cxs = (gt_bbox[..., 0] + gt_bbox[..., 2]) / 2.0
        cys = (gt_bbox[..., 1] + gt_bbox[..., 3]) / 2.0
        # 2.2: Create the center bbox
        c_box_left = cxs - center_radius * stride
        c_box_right = cxs + center_radius * stride
        c_box_top = cys - center_radius * stride
        c_box_bot = cys + center_radius * stride
        # 2.3: Condition checking
        c_left = xs - c_box_left
        c_right = c_box_right - xs
        c_top = ys - c_box_top
        c_bot = c_box_bot - ys
        center_deltas = torch.stack((c_left, c_top, c_right, c_bot), -1)
        is_in_centers = center_deltas.min(-1)[0] > 0

        is_in_bbox_and_center = is_in_bbox & is_in_centers

        return is_in_bbox_and_center

    def _top_k_points(self,
                      postive_mask,
                      num_gt):
        new_positive_mask = torch.zeros_like(postive_mask, dtype=torch.uint8) # (num_priors, num_gts)
        top_k_inds = []
        for i in range(num_gt):
            _, top_k_ind = torch.topk(postive_mask.long()[:, i], k=self.top_k)
            new_positive_mask[:, i][top_k_ind] = 1
            top_k_inds.append(top_k_ind)
        new_positive_mask = new_positive_mask.sum(dim=1) > 0 # (num_priors, )
        return new_positive_mask, top_k_inds