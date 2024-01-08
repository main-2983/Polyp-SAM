from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling.common import LayerNorm2d

from ..assigner.point_generator import PointGenerator
from .base_iterative_prompt import *
from .layers import ConvModule

__all__ = ["SingleDiceIterativePrompt", "SingleDiceIterativePromptSAM"]


class SingleDiceIterativePrompt(BaseIterativePrompt):
    """
    This is Iterative Self Prompt module which supports only positive or negative point
    The target assignment process uses
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
        super(SingleDiceIterativePrompt, self).__init__(strides=strides)

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
            Prediction mask of shape (1, 1, 64, 64)
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
        priors_generator = PointGenerator(stride=self.strides[0])
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


class SingleDiceIterativePromptSAM(BaseIterativePromptSAM):
    def __init__(self,
                 *args,
                 point_prompt_module: SingleDiceIterativePrompt,
                 **kwargs):
        assert isinstance(point_prompt_module, SingleDiceIterativePrompt)
        super(SingleDiceIterativePromptSAM, self).__init__(
            *args, point_prompt_module=point_prompt_module, **kwargs)

    @torch.no_grad()
    def get_target_single(self, gt_instance: Dict[str, Any]) -> torch.Tensor:
        """
        This function takes every point in mask_to_sample as point prompt then forward to the mask_decoder
        Every point will have its dice score, which then create the dice score map
        """
        mask_to_sample = gt_instance["mask_to_sample"] # (1024, 1024)
        image_embedding = gt_instance["image_embedding"] # (1, 256, H, W)
        logit_mask = gt_instance["logit_mask"] # (num_objects, 1, 256, 256)
        positive = self.point_prompt_module.positive
        stride = self.point_prompt_module.strides[0]
        device = image_embedding.device
        featmap_size = image_embedding.shape[-2:]

        # Step 0: Create an all 0s target
        target = torch.zeros(featmap_size[0] * featmap_size[1], device=device)
        # Step 1: Shrink the mask_to_sample to (64, 64) so we have fewer points to sample
        scaled_mask_to_sample = F.interpolate(mask_to_sample[None, None].float(), featmap_size, mode='bilinear')
        scaled_mask_to_sample = torch.where(scaled_mask_to_sample > 0.5, True, False) # Convert to boolean mask
        # Step 2: Create the grid to sample points from
        prior_generator = PointGenerator(stride=stride)
        priors = prior_generator.grid_points(featmap_size, device=device) # (H * W, 2)
        # Step 3: Sample points
        flatten_mask_to_sample = scaled_mask_to_sample.flatten() # (H * W,)
        selected_indices = torch.nonzero(flatten_mask_to_sample, as_tuple=True)[0] # (H * W,)
        selected_points = priors[flatten_mask_to_sample] # (num_selected, 2)
        selected_labels = torch.ones((selected_points.shape[0], 1), device=device, dtype=torch.long) # (num_selected, 1)
        if not positive:
            selected_labels = torch.zeros((selected_points.shape[0], 1), device=device, dtype=torch.long)
        # Step 4: Perform forward for every single point to get the dice score
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
            target[index] = selected_iou

        return target