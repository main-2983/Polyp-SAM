from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

from segment_anything import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling.common import LayerNorm2d

from src.models.polypSAM import PolypSAM
from src.models.assigner.point_generator import PointGenerator
from .layers import ConvModule


class IterativePointPrompt(nn.Module):
    def __init__(self,
                 num_convs: int = 3,
                 in_channels: int = 256,
                 feat_channels: int = 128,
                 kernel_size: int = 3,
                 stride: int = 16):
        super(IterativePointPrompt, self).__init__()

        self.stride = stride
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
        self.pred = nn.Conv2d(feat_channels, 2, kernel_size=1)

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
            pred (Tensor): of shape (1, 2, H, W)
        """
        _, _, H, W = pred.shape
        device = pred.device
        priors_generator = PointGenerator()
        positive_priors = priors_generator.grid_points(
            (H, W), device=device) # (H * W, 2)
        negative_priors = priors_generator.grid_points(
            (H, W), device=device) # (H * W, 2)

        pred = pred.sigmoid()
        pred = pred.permute(0, 2, 3, 1).view(-1, 2) # (H * W, 2)
        selected_mask = torch.where(pred >= positive_threshold, True, False) # (H * W, 2)
        selected_positives = positive_priors[selected_mask[:, 1]] # (num_selected_pos, 2)
        selected_mask = torch.where(pred >= negative_threshold, True, False)
        selected_negatives = negative_priors[selected_mask[:, 0]] # (num_selected_neg, 2)
        positive_labels = torch.ones((selected_positives.shape[0], ),
                                     dtype=torch.long, device=device) # (num_selected_pos, )
        negative_labels = torch.zeros((selected_negatives.shape[0], ),
                                      dtype=torch.long, device=device) # (num_selected_neg, )

        selected_points = torch.cat([selected_positives, selected_negatives],
                                    dim=0) # (num_selected, 2)
        selected_labels = torch.cat([positive_labels, negative_labels],
                                    dim=0) # (num_selected, )

        return selected_points.unsqueeze(0), selected_labels.unsqueeze(0) # Expand num_box dim

    def prepare_for_loss(self,
                         pred: torch.Tensor,
                         point_prompt: Tuple[torch.Tensor, torch.Tensor],
                         img_embedding: torch.Tensor):
        """
        Prepare the loss component based on the features extracted by the head
        and the next-iteration prompts
        Args:
            pred (Tensor): prediction of shape (1, 2, H, W)
            point_prompt (Tuple) contains:
                + point of shape (num_box, point_per_box, 2)
                + label of shape (num_box, point_per_box)
            img_embedding (Tensor): output embedding of shape (1, 256, H, W)
        """
        target = self.get_target_single(
            point_prompt, img_embedding
        ) # (H * W, 2)
        flatten_pred = pred.permute(0, 2, 3, 1).view(-1, 2).contiguous() # (H * W, 2)

        return target, flatten_pred

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
        assigned_gt_inds = torch.full((num_priors, 2), 0, device=device) # (num_priors, 2)

        for (point, label) in zip(points, labels):
            point = point / self.stride # scale the point to featmap size
            point[:, 0] = torch.clamp(point[:, 0], min=0, max=featmap_size[0])
            point[:, 1] = torch.clamp(point[:, 1], min=0, max=featmap_size[1])
            # Map point to index
            index = [torch.floor((torch.ceil(p[1]) - 1) * featmap_size[1] + p[0]).to(torch.int) for p in point]
            assert len(index) == len(label)
            for i in range(len(index)):
                assigned_gt_inds[index[i], label[i]] = 1

        return assigned_gt_inds


class IterativeSinglePointPrompt(IterativePointPrompt):
    def __init__(self,
                 *args,
                 positive: bool = True,
                 **kwargs):
        super(IterativeSinglePointPrompt, self).__init__(*args, **kwargs)
        self.pred = nn.Conv2d(self.feat_channels, 1, kernel_size=1)
        self.positive = positive

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

    def prepare_for_loss(self,
                         pred: torch.Tensor,
                         point_prompt: Tuple[torch.Tensor, torch.Tensor],
                         img_embedding: torch.Tensor):
        """
        Prepare the loss component based on the features extracted by the head
        and the next-iteration prompts
        Args:
            pred (Tensor): prediction of shape (1, 1, H, W)
            point_prompt (Tuple) contains:
                + point of shape (num_box, point_per_box, 2)
                + label of shape (num_box, point_per_box)
            img_embedding (Tensor): output embedding of shape (1, 256, H, W)
        """
        target = self.get_target_single(
            point_prompt, img_embedding
        ) # (H * W, 1)
        flatten_pred = pred.permute(0, 2, 3, 1).view(-1, 1).contiguous() # (H * W, 1)

        return target, flatten_pred

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
            point = point / self.stride # scale the point to featmap size
            point[:, 0] = torch.clamp(point[:, 0], min=0, max=featmap_size[0])
            point[:, 1] = torch.clamp(point[:, 1], min=0, max=featmap_size[1])
            # Map point to index
            index = [torch.floor((torch.ceil(p[1]) - 1) * featmap_size[1] + p[0]).to(torch.int) for p in point]
            assert len(index) == len(label)
            for i in range(len(index)):
                assigned_gt_inds[index[i]] = 1

        return assigned_gt_inds


class IterativeSelfPromptSAM(PolypSAM):
    def __init__(self,
                 *args,
                 point_prompt_module: IterativePointPrompt,
                 **kwargs):
        super(IterativeSelfPromptSAM, self).__init__(*args, **kwargs)

        self.point_prompt_module = point_prompt_module
        self.image_embedding = None

    def forward(self,
                input: Dict[str, Any],
                multimask_output: bool = False):
        """
        This forward function does not accept batch input
        """

        image = input.get("image")

        # Normal SAM forward
        if self.image_embedding is None:
            image_embedding = self.image_encoder(image[None])
        else:
            image_embedding = self.image_embedding

        points = input.get("point_prompt")
        mask_input = input.get("mask_input", None)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=input.get("box_prompt", None),
            masks=mask_input,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        # Self-prompt
        point_pred = None
        if mask_input is not None:
            point_pred = self.point_prompt_module(image_embedding.detach(), dense_embeddings.detach())

        return low_res_masks, iou_predictions, point_pred, image_embedding
    
    def forward_embedding(self, image):
        self.image_embedding = self.image_encoder(image[None])
