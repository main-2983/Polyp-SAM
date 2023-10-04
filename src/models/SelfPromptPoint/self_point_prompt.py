import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Any

from segment_anything.modeling import MaskDecoder, ImageEncoderViT, PromptEncoder


class PointGenModel(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    def __init__(self,
                 image_encoder: ImageEncoderViT,
                 point_model: nn.Module,
                 pixel_mean: List[float] = [0.485, 0.456, 0.406],
                 pixel_std: List[float] = [0.229, 0.224, 0.225],
                 freeze: bool = True):
        super(PointGenModel, self).__init__()
        self.image_encoder = image_encoder
        self.point_model = point_model

        self.register_buffer('pixel_mean', torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.tensor(pixel_std).view(-1, 1, 1), False)

        if freeze:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def forward(self, x):
        image = torch.stack([self.preprocess(img) for img in x], dim=0)
        image_embedding = self.image_encoder(image)

        point_pred = self.point_model(image_embedding)

        return point_pred


class SelfPointPromptSAM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    def __init__(
            self,
            point_model: nn.Module,
            image_encoder: ImageEncoderViT,
            mask_decoder: MaskDecoder,
            prompt_encoder: PromptEncoder,
            pixel_mean: List[float] = [0.485, 0.456, 0.406],
            pixel_std: List[float] = [0.229, 0.224, 0.225],
            freeze: List[nn.Module] = None
    ):
        super(SelfPointPromptSAM, self).__init__()

        self.point_model = point_model
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        num_points = self.point_model.num_points
        self.register_buffer('labels', torch.ones((1, num_points), dtype=torch.long), False)
        self.labels[0, num_points // 2:] = 0

        self.register_buffer('pixel_mean', torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.tensor(pixel_std).view(-1, 1, 1), False)

        if freeze is not None:
            for module in freeze:
                for m in [self.image_encoder, self.mask_decoder, self.prompt_encoder]:
                    if isinstance(m, type(module)):
                        for param in m.parameters():
                            param.requires_grad = False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self,
                input: Dict[str, Any],
                multimask_output: bool = False):
        """
        This forward function does not accept batch input
        input = {
            'image': torch.Tensor,
            'image_embedding': torch.Tensor,
            'image_size': Tuple[int, int]
        }
        """

        image = input.get("image")  # [1, 1, 1024, 1024]
        image = torch.stack([self.preprocess(img) for img in image], dim=0)
        image_embedding = input.get("image_embedding", None)  # [1, 256, 64, 64]
        if image_embedding is None:
            image_embedding = self.image_encoder(image)

        points = self.point_model(image)

        point_prompt = (points, self.labels)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=point_prompt,
            boxes=None,
            masks=None
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        mask = self.postprocess_masks(
            low_res_masks,
            input_size=image.shape[-2:],
            original_size=input.get("image_size"),
        )

        return mask

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            models, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the models, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class SelfPointPromptSAMv2(SelfPointPromptSAM):
    def __init__(self,
                 point_model: nn.Module,
                 *args,
                 **kwargs):
        super(SelfPointPromptSAMv2, self).__init__(point_model, *args, **kwargs)
        self.point_model = point_model

    def forward(self,
                input: Dict[str, Any],
                multimask_output: bool = False):

        image = input.get("image")  # [1, 1, 1024, 1024]
        image = torch.stack([self.preprocess(img) for img in image], dim=0)
        image_embedding = input.get("image_embedding", None)  # [1, 256, 64, 64]
        if image_embedding is None:
            image_embedding = self.image_encoder(image)

        points = self.point_model(image_embedding)

        point_prompt = (points, self.labels)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=point_prompt,
            boxes=None,
            masks=None
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        mask = self.postprocess_masks(
            low_res_masks,
            input_size=image.shape[-2:],
            original_size=input.get("image_size"),
        )

        return mask
