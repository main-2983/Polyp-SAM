from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn

from segment_anything.modeling.common import LayerNorm2d

from ..assigner.point_generator import PointGenerator
from .base_iterative_prompt import *
from .layers import ConvModule

__all__ = ["SingleIterativePromptSAM", "SingleIterativePrompt"]


class SingleIterativePrompt(BaseIterativePrompt):
    """
    This is Iterative Self Prompt module which supports only positive or negative point
    Args:
        positive (bool): Where to sample positive point
    """
    def __init__(self,
                 num_convs: int = 3,
                 in_channels: int = 256,
                 feat_channels: int = 128,
                 kernel_size: int = 3,
                 positive: bool = True,
                 strides: List[int] = [16]):
        super(SingleIterativePrompt, self).__init__(strides=strides)

        self.feat_channels = feat_channels
        convs = []
        for i in range(num_convs):
            in_chn = in_channels if i == 0 else feat_channels
            convs.append(
                ConvModule(
                    in_channels=in_chn,
                    out_channels=feat_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    norm=LayerNorm2d(feat_channels),
                    activation=nn.ReLU(inplace=True)
                )
            )
        self.convs = nn.Sequential(*convs)
        self.pred = nn.Conv2d(self.feat_channels, 1, kernel_size=1)
        self.positive = positive

    def forward(self,
                img_emb: torch.Tensor,
                dense_emb: torch.Tensor) -> torch.Tensor:
        """
        This forward function does not support batch forward
        Args:
            img_emb: of shape (1, 256, 64, 64)
            dense_emb: of shape (num_objs, 256, 64, 64)
        Returns:
            Prediction mask of shape (1, 2, 64, 64)
        """
        img_emb = img_emb.detach()
        dense_emb = dense_emb.detach()

        num_objs = dense_emb.shape[0]
        src = torch.repeat_interleave(img_emb, num_objs, dim=0)
        src = src + dense_emb

        feats = self.convs(src)
        pred = self.pred(feats)
        pred = torch.sum(pred, dim=0, keepdim=True)

        return pred

    def decode_prediction(self,
                          pred: torch.Tensor,
                          positive_threshold: float = 0.5,
                          negative_threshold: float = 0.5):
        """
        Convert prediction to point and label
        Single image prediction, don't use on batch size > 1
        Args:
            pred (Tensor): of shape (1, 1, H, W)
        """
        _, _, H, W = pred.shape
        device = pred.device
        priors_generator = PointGenerator()
        positive_priors = priors_generator.grid_points(
            (H, W), device=device) # (H * W, 2)
        negative_priors = priors_generator.grid_points(
            (H, W), device=device) # (H * W, 2)

        pred = pred.sigmoid()
        pred = pred.permute(0, 2, 3, 1).flatten() # (H * W,)
        if self.positive:
            selected_mask = torch.where(pred >= positive_threshold, True, False) # (H * W,)
            selected_priors = positive_priors[selected_mask] # (num_selected_pos, 2)
            labels = torch.ones((selected_priors.shape[0],),
                                dtype=torch.long, device=device)
        else:
            selected_mask = torch.where(pred >= negative_threshold, True, False)
            selected_priors = negative_priors[selected_mask] # (num_selected_neg, 2)
            labels = torch.zeros((selected_priors.shape[0],),
                                dtype=torch.long, device=device)

        return selected_priors.unsqueeze(0), labels.unsqueeze(0) # Expand num_box dim

    @torch.no_grad()
    def get_target_single(self,
                          point_prompt: Tuple[torch.Tensor, torch.Tensor],
                          img_embedding: torch.Tensor):
        """
        Convert the point prompt for the next iteration into label assignment mask
        Args:
            point_prompt: Tuple of point and label. Point has shape (num_box, point_per_box, 2),
                                                    Label has shape (num_box, point_per_box)
        """
        points = point_prompt[0]
        labels = point_prompt[1]
        device = img_embedding.device
        featmap_size = img_embedding.shape[-2:]
        num_priors = featmap_size[0] * featmap_size[1]
        # Assigned with 0
        assigned_gt_inds = torch.full((num_priors, 1), 0, device=device) # (num_priors, 1)

        for (point, label) in zip(points, labels):
            point = point / self.strides[0] # scale the point to featmap size
            point[:, 0] = torch.clamp(point[:, 0], min=0, max=featmap_size[0])
            point[:, 1] = torch.clamp(point[:, 1], min=0, max=featmap_size[1])
            # Map point to index
            index = [torch.floor((torch.ceil(p[1]) - 1) * featmap_size[1] + p[0]).to(torch.int) for p in point]
            assert len(index) == len(label)
            for i in range(len(index)):
                assigned_gt_inds[index[i]] = 1

        return assigned_gt_inds


class SingleIterativePromptSAM(BaseIterativePromptSAM):
    def __init__(self,
                 *args,
                 point_prompt_module: SingleIterativePrompt,
                 **kwargs):
        super(SingleIterativePromptSAM, self).__init__(
            *args, point_prompt_module=point_prompt_module, **kwargs)

    def get_target_single(self, gt_instances: Dict[str, Any]) -> torch.Tensor:
        return self.point_prompt_module.get_target_single(
            gt_instances["point_prompt"],
            gt_instances["image_embedding"]
        )
