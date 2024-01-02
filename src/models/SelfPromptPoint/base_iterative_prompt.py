from typing import List, Dict, Any, Tuple
from abc import abstractmethod

import torch
import torch.nn as nn

from ..polypSAM import PolypSAM
from ..assigner.point_generator import PointGenerator, MlvlPointGenerator

__all__ = ["BaseIterativePromptSAM", "BaseIterativePrompt"]


class BaseIterativePrompt(nn.Module):
    def __init__(self,
                 strides: List[int] = [16]):
        super(BaseIterativePrompt, self).__init__()
        self.strides = strides

    @abstractmethod
    def forward(self, **kwargs) -> torch.Tensor:
        pass

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
        priors_generator = PointGenerator(self.strides[0])
        if len(self.strides) > 1:
            priors_generator = MlvlPointGenerator(self.strides)

        positive_priors = priors_generator.grid_points(
            (H, W), device=device) # (H * W, 2)
        negative_priors = priors_generator.grid_points(
            (H, W), device=device) # (H * W, 2)

        # Positive Priors will have index 1
        # Negative Priors will have index 0
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

    @abstractmethod
    @torch.no_grad()
    def get_target_single(self, **kwargs):
        pass


class BaseIterativePromptSAM(PolypSAM):
    def __init__(self,
                 *args,
                 point_prompt_module: BaseIterativePrompt,
                 **kwargs):
        super(BaseIterativePromptSAM, self).__init__(*args, **kwargs)

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

    def prepare_for_loss(self,
                         pred: torch.Tensor,
                         gt_instances: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the loss component based on the features extracted by the head
        and the next-iteration prompts
        Args:
            pred (Tensor): prediction of shape (1, 2, H, W)
            gt_instances (dict): Required fields for get_target_single
        """
        target = self.get_target_single(gt_instances) # (H * W, 2)
        flatten_pred = pred.permute(0, 2, 3, 1).view(-1, 2).contiguous() # (H * W, 2)

        return target, flatten_pred

    @abstractmethod
    def get_target_single(self, gt_instances: Dict[str, Any]) -> torch.Tensor:
        pass
