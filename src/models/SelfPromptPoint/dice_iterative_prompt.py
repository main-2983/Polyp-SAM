from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling.common import LayerNorm2d

from ..assigner.point_generator import PointGenerator
from .base_iterative_prompt import *
from .layers import ConvModule

__all__ = ["DiceIterativePrompt", "DiceIterativePromptSAM", "SplitDiceIterativePrompt"]


class DiceIterativePrompt(BaseIterativePrompt):
    """
    This is Iterative Self Prompt module which supports both positive and negative point
    """
    def __init__(self,
                 num_convs: int = 3,
                 in_channels: int = 256,
                 feat_channels: int = 128,
                 kernel_size: int = 3,
                 strides: List[int] = [16]):
        super(DiceIterativePrompt, self).__init__(strides=strides)

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
        self.pred = nn.Conv2d(self.feat_channels, 2, kernel_size=1)

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

    def get_target_single(self):
        return


class SplitDiceIterativePrompt(DiceIterativePrompt):
    """
    This is Iterative Self Prompt module which supports both positive and negative point
    However, we split the prediction of positive and negative
    """
    def __init__(self,
                 num_convs: int = 3,
                 in_channels: int = 256,
                 feat_channels: int = 128,
                 kernel_size: int = 3,
                 strides: List[int] = [16]):
        super(DiceIterativePrompt, self).__init__(strides=strides)

        self.feat_channels = feat_channels
        convs = []
        for i in range(num_convs - 1):
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
        self.inter_convs = nn.Sequential(*convs)
        self.positive_decomp = ConvModule(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            padding=1,
            norm=LayerNorm2d(feat_channels),
            activation=nn.ReLU(inplace=True)
        )
        self.negative_decomp = ConvModule(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            padding=1,
            norm=LayerNorm2d(feat_channels),
            activation=nn.ReLU(inplace=True)
        )
        self.positive_pred = nn.Conv2d(self.feat_channels, 1, kernel_size=1)
        self.negative_pred = nn.Conv2d(self.feat_channels, 1, kernel_size=1)

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

        inter_feats = self.inter_convs(src)
        positive_feats = self.positive_decomp(inter_feats)
        negative_feats = self.negative_decomp(inter_feats)

        positive_pred = self.positive_pred(positive_feats)
        negative_pred = self.negative_pred(negative_feats)
        pred = torch.cat([negative_pred, positive_pred], dim=1)
        pred = torch.sum(pred, dim=0, keepdim=True)

        return pred

    def get_target_single(self):
        return


class DiceIterativePromptSAM(BaseIterativePromptSAM):
    def __init__(self,
                 *args,
                 point_prompt_module: BaseIterativePrompt,
                 **kwargs):
        assert isinstance(point_prompt_module, BaseIterativePrompt)
        super(DiceIterativePromptSAM, self).__init__(
            *args, point_prompt_module=point_prompt_module, **kwargs)

    @torch.no_grad()
    def get_target_single(self, gt_instance: Dict[str, Any]) -> torch.Tensor:
        """
        This function takes every point in mask_to_sample as point prompt then forward to the mask_decoder
        Every point will have its dice score, which then create the dice score map
        """
        stride = self.point_prompt_module.strides[0]
        negative_mask_to_sample = gt_instance["mask_to_sample"][0] # (1024, 1024)

        positive_mask_to_sample = gt_instance["mask_to_sample"][1] # (1024, 1024)
        image_embedding = gt_instance["image_embedding"] # (1, 256, H, W)
        logit_mask = gt_instance["logit_mask"] # (num_objects, 1, 256, 256)
        featmap_size = image_embedding.shape[-2:]
        device = image_embedding.device

        # Step 0: Create an all 0s target
        target = torch.zeros((featmap_size[0] * featmap_size[1], 2), device=device) # (H * W, 2)
        # Step 1: Shrink the mask_to_sample to (64, 64) so we have fewer points to sample
        scaled_negative_mask = F.interpolate(negative_mask_to_sample[None, None].float(), featmap_size, mode='bilinear')
        scaled_negative_mask = torch.where(scaled_negative_mask > 0.5, True, False) # Convert to boolean mask
        scaled_positive_mask = F.interpolate(positive_mask_to_sample[None, None].float(), featmap_size, mode='bilinear')
        scaled_positive_mask = torch.where(scaled_positive_mask > 0.5, True, False) # Convert to boolean mask
        # Step 2: Create the grid to sample points from
        prior_generator = PointGenerator(stride=stride)
        priors = prior_generator.grid_points(featmap_size, device=device) # (H * W, 2)
        # Step 3: Sample points
        # Step 3.1: Sample negative points
        flatten_mask_to_sample = scaled_negative_mask.flatten() # (H * W,)
        selected_indices = torch.nonzero(flatten_mask_to_sample, as_tuple=True)[0] # (H * W,)
        selected_points = priors[flatten_mask_to_sample] # (num_selected, 2)
        selected_labels = torch.zeros((selected_points.shape[0], 1), device=device, dtype=torch.long) # (num_selected, 1)
        # Step 3.2: Perform forward for every single point to get the dice score
        for (point, label, index) in zip(selected_points, selected_labels, selected_indices):
            point = point[None, None] # Expand to (1, 1, 2)
            label = label[None] # Expand to (1, 1)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point, label),
                boxes=None,
                masks=logit_mask,
            )
            # (num_objects, 1, 256, 256) and (num_objects, 1)
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            selected_iou = iou_predictions[torch.argmax(iou_predictions)]
            target[index, 0] = selected_iou
        # Step 3.3: Sample positive points
        flatten_mask_to_sample = scaled_positive_mask.flatten() # (H * W,)
        selected_indices = torch.nonzero(flatten_mask_to_sample, as_tuple=True)[0] # (H * W,)
        selected_points = priors[flatten_mask_to_sample] # (num_selected, 2)
        selected_labels = torch.ones((selected_points.shape[0], 1), device=device, dtype=torch.long) # (num_selected, 1)
        for (point, label, index) in zip(selected_points, selected_labels, selected_indices):
            point = point[None, None] # Expand to (1, 1, 2)
            label = label[None] # Expand to (1, 1)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point, label),
                boxes=None,
                masks=logit_mask,
            )
            # (num_objects, 1, 256, 256) and (num_objects, 1)
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            selected_iou = iou_predictions[torch.argmax(iou_predictions)]
            target[index, 1] = selected_iou

        return target