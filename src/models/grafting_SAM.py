from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling import PromptEncoder, MaskDecoder
from segment_anything.modeling.common import LayerNorm2d


class GraftingSAM(nn.Module):
    def __init__(self,
                 image_encoder: nn.Module,
                 prompt_encoder: PromptEncoder,
                 mask_decoder: MaskDecoder,
                 img_enc_out_index: int,
                 img_enc_dim: int,
                 out_chans: int,
                 pixel_mean: List[float] = [0.485, 0.456, 0.406],
                 pixel_std: List[float] = [0.229, 0.224, 0.225],
                 freeze: List[nn.Module] = None):
        super(GraftingSAM, self).__init__()

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.out_index = img_enc_out_index

        self.neck = nn.Sequential(
            nn.Conv2d(
                img_enc_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        if freeze is not None:
            for module in freeze:
                for m in [self.image_encoder, self.mask_decoder, self.prompt_encoder]:
                    if isinstance(m, type(module)):
                        for param in m.parameters():
                            param.requires_grad = False

    def forward(self,
                input: Dict[str, Any],
                multimask_output: bool = False):
        image = input.get("image")

        image_embeddings = self.image_encoder(image[None])
        image_embedding = image_embeddings[self.out_index]
        image_embedding = self.neck(image_embedding)

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

        return low_res_masks, iou_predictions

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Upscale masks to the original image size.

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

        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values"""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        return x
