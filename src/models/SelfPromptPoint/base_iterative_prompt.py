from typing import List, Dict, Any, Tuple, Optional
from abc import abstractmethod, ABCMeta

import torch
import torch.nn as nn
import numpy as np

from segment_anything import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

from ..polypSAM import PolypSAM
from ..assigner.point_generator import PointGenerator, MlvlPointGenerator

__all__ = ["BaseIterativePromptSAM", "BaseIterativePrompt", "IterativeSelfPredictor",
           "AutoSAM"]


class BaseIterativePrompt(nn.Module, metaclass=ABCMeta):
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
        priors_generator = PointGenerator(stride=self.strides[0])
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


class BaseIterativePromptSAM(PolypSAM, metaclass=ABCMeta):
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

    def forward_embedding(self, image):
        self.image_embedding = self.image_encoder(image[None])

    def prepare_for_loss(self,
                         pred: torch.Tensor,
                         gt_instance: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the loss component based on the features extracted by the head
        and the next-iteration prompts
        Args:
            pred (Tensor): prediction of shape (1, 2, H, W)
            gt_instance (dict): Required fields for get_target_single
        """
        target = self.get_target_single(gt_instance) # (H * W, 2)
        flatten_pred = pred.permute(0, 2, 3, 1).view(-1, 2).contiguous() # (H * W, 2)

        return target, flatten_pred

    @abstractmethod
    def get_target_single(self, gt_instance: Dict[str, Any]) -> torch.Tensor:
        pass

    def save_module(self,
                    checkpoint: str,
                    save_path: str):
        """
        This method extracts the point prompt module from the base model and save it to `save_path`
        Args:
            checkpoint (str): Path to the base model
            save_path (str): Path to save
        """
        model_state_dict = torch.load(checkpoint, map_location="cpu")
        module_state_dict = self.point_prompt_module.state_dict()

        # Remove string in keys
        for k, v in model_state_dict.items():
            if "point_prompt_module" in k:
                key = k.split("point_prompt_module.")[1]
                module_state_dict[key] = v

        torch.save(module_state_dict, save_path)


class IterativeSelfPredictor(SamPredictor):
    def __init__(self,
                 *args,
                 model: BaseIterativePromptSAM):
        super(IterativeSelfPredictor, self).__init__(*args)
        self.model = model
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.reset_image()

    def predict(
        self,
        threshold: Tuple[float, float] = [0.5, 0.5],
        point_prompt: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        mask_input_torch, point_prompt_torch = None, None

        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        if point_prompt is not None:
            point_coords = self.transform.apply_coords(point_prompt[0], self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_prompt[1], dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            point_prompt_torch = (coords_torch, labels_torch)

        masks, iou_predictions, low_res_masks, points, labels = self.predict_torch(
            threshold,
            point_prompt_torch,
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
        point_prompt: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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

        if point_prompt is None:
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
        else:
            points = point_prompt
            point_coords = point_prompt[0]
            labels = point_prompt[1]

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


class AutoSAM(SamPredictor):
    def __init__(self,
                 *args,
                 first_prompt_module: nn.Module,
                 iterative_prompt_module: BaseIterativePrompt):
        super(AutoSAM, self).__init__(*args)
        self.first_prompt_module = first_prompt_module
        self.iterative_prompt_module = iterative_prompt_module
        self.model.pixel_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.model.pixel_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def inference_mode(self, device):
        self.model.to(device)
        self.model.eval()
        self.first_prompt_module.to(device)
        self.first_prompt_module.eval()
        self.iterative_prompt_module.to(device)
        self.iterative_prompt_module.eval()

    def load_weights(self,
                     sam_weight_path: str = None,
                     first_prompt_weight_path: str = None,
                     iterative_prompt_weight_path: str = None):
        if sam_weight_path is not None:
            sam_sate_dict = torch.load(sam_weight_path, map_location="cpu")
            self.model.load_state_dict(sam_sate_dict)
        if first_prompt_weight_path is not None:
            first_prompt_state_dict = torch.load(first_prompt_weight_path, map_location="cpu")
            self.first_prompt_module.load_state_dict(first_prompt_state_dict)
        if iterative_prompt_weight_path is not None:
            iterative_prompt_state_dict = torch.load(iterative_prompt_weight_path, map_location="cpu")
            self.iterative_prompt_module.load_state_dict(iterative_prompt_state_dict)

    def predict(
        self,
        iter: int,
        first_threshold: float = 0.4,
        iterative_threshold: Tuple[float, float] = [0.5, 0.5],
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        mask_input_torch, point_prompt_torch = None, None

        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks, points, labels = self.predict_torch(
            iter,
            first_threshold,
            iterative_threshold,
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
        iter: int,
        first_threshold: float = 0.4,
        iterative_threshold: Tuple[float, float] = [0.5, 0.5],
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

        if iter == 0:
            first_prompt_pred = self.first_prompt_module(self.features)
            point_coords, point_labels = self.first_prompt_module.decode_prediction(first_prompt_pred, first_threshold)
        elif iter > 0:
            # Get dense emb
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=None,
                masks=mask_input)
            iterative_prompt_pred = self.iterative_prompt_module(self.features, dense_embeddings)
            point_coords, point_labels = self.iterative_prompt_module.decode_prediction(
                iterative_prompt_pred, iterative_threshold[0], iterative_threshold[1]
            )
        else:
            raise AssertionError

        points = (point_coords, point_labels)

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

        return masks, iou_predictions, low_res_masks, point_coords, point_labels