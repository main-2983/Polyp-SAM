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
                 stride: int = 16):
        super(IterativePointPrompt, self).__init__()

        self.stride = stride
        convs = []
        for i in range(num_convs):
            in_chn = in_channels if i == 0 else feat_channels
            convs.append(
                ConvModule(
                    in_channels=in_chn,
                    out_channels=feat_channels,
                    kernel_size=3,
                    padding=1,
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
            (H, W), stride=self.stride, device=device) # (H * W, 2)
        negative_priors = priors_generator.grid_points(
            (H, W), stride=self.stride, device=device) # (H * W, 2)

        pred = pred.sigmoid()
        pred = pred.permute(0, 2, 3, 1).view(-1, 2) # (H * W, 2)
        selected_mask = torch.where(pred >= positive_threshold, True, False) # (H * W, 2)
        selected_positives = positive_priors[selected_mask[:, 0]] # (num_selected_pos, 2)
        selected_mask = torch.where(pred >= negative_threshold, True, False)
        selected_negatives = negative_priors[selected_mask[:, 1]] # (num_selected_neg, 2)
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


class IterativeSelfPromptSAM(PolypSAM):
    def __init__(self,
                 *args,
                 point_prompt_module: IterativePointPrompt,
                 **kwargs):
        super(IterativeSelfPromptSAM, self).__init__(*args, **kwargs)

        self.point_prompt_module = point_prompt_module

    def forward(self,
                input: Dict[str, Any],
                multimask_output: bool = False):
        """
        This forward function does not accept batch input
        """

        image = input.get("image")

        # Normal SAM forward
        image_embedding = self.image_encoder(image[None])
        points = input.get("point_prompt")
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=input.get("box_prompt", None),
            masks=input.get("mask_input", None),
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        # Self-prompt
        point_pred = self.point_prompt_module(image_embedding.detach(), dense_embeddings.detach())

        return low_res_masks, iou_predictions, point_pred, image_embedding


class IterativeSelfPredictor(SamPredictor):
    def __init__(self,
                 *args,
                 model: IterativeSelfPromptSAM):
        super(IterativeSelfPredictor, self).__init__(*args)
        self.model = model
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.reset_image()

    def predict(
        self,
        threshold: Tuple[float, float] = [0.5, 0.5],
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        mask_input_torch = None

        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks, points, labels = self.predict_torch(
            threshold,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        points_np = points[0].detach().cpu().numpy()
        labels_np = labels[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np, points_np, labels_np

    @torch.no_grad()
    def predict_torch(
        self,
        threshold: Tuple[float, float] = [0.5, 0.5],
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            masks (Tensor): output masks of shape (1, C, H, W) where C is the number of
                masks, (H, W) is the original image size
            iou_predictions (Tensor): of shape (B, C) containing the model's
                predictions for the quality of each mask
            low_res_mask (Tensor): of shape (1, C, H, W) where C is the
                number of masks, H=W=256. These low res logits can be passed
                to a subsequent iteration as mask input
            point_coords (Tensor): of shape (1, num_points, 2)
            labels (Tensor): of shape (1, num_points)
        """

        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Get dense emb
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=None,
            masks=mask_input)

        # Predict points
        point_pred = self.model.point_prompt_module(
            self.features, dense_embeddings)
        point_coords, labels = self.model.point_prompt_module.decode_prediction(
            point_pred, threshold[0], threshold[1])
        points = (point_coords, labels)

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=None,
            masks=mask_input
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks, point_coords, labels