
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heatmap_model import PointModelHeatmap

from typing import List, Tuple, Dict, Any

from src.models.SelfPromptPointHeatmap.loss import BASE_LOSS_DICT

from segment_anything.modeling import MaskDecoder, ImageEncoderViT, PromptEncoder


class SelfPointPromptWithHeatmap(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    def __init__(
            self,
            heat_model: PointModelHeatmap,
            image_encoder: ImageEncoderViT,
            mask_decoder: MaskDecoder,
            prompt_encoder: PromptEncoder,
            pixel_mean: List[float] = [0.485, 0.456, 0.406],
            pixel_std: List[float] = [0.229, 0.224, 0.225],
            freeze: List[nn.Module] = None
    ):
        super(SelfPointPromptWithHeatmap, self).__init__()

        self.heat_model = heat_model
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # num_points = self.point_model.num_points
        self.register_buffer('labels', torch.tensor([[1, 0]], dtype=torch.long), False)
        # self.labels[0, num_points // 2:] = 0

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
                INPUT: Dict[str, Any],
                multimask_output: bool = False):

        image = INPUT.get("image")  # [1, 1, 1024, 1024]
        image = torch.stack([self.preprocess(img) for img in image], dim=0)

        image_embedding = self.image_encoder(image)
        
        heatmap_out = self.heat_model(image_embedding)
        loss_dict = dict()
        for i in range(len(heatmap_out)):
            loss_dict['heatmap_loss{}'.format(i)] = dict(
                params=[heatmap_out[i], INPUT['heatmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
                weight=torch.cuda.FloatTensor([BASE_LOSS_DICT['weight_dict']['heatmap_loss{}'.format(i)]])
            )

        return heatmap_out[-1], loss_dict
        # return heatmap_out, loss_dict



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

# class OpenPose(nn.Module):
#     def __init__(self, configer):
#         super(OpenPose, self).__init__()
#         self.configer = configer
#         self.backbone = ModuleHelper.get_backbone(
#             backbone=self.configer.get('network.backbone'),
#             pretrained=self.configer.get('network.pretrained')
#         )
#         self.pose_model = PoseModel(configer, self.backbone.get_num_features())
#         self.valid_loss_dict = configer.get('loss', 'loss_weights', configer.get('loss.loss_type'))

#     def forward(self, data_dict):
#         x = self.backbone(data_dict['img'])
#         paf_out, heatmap_out = self.pose_model(x)
#         out_dict = dict(paf=paf_out[-1], heatmap=heatmap_out[-1])
#         if self.configer.get('phase') == 'test':
#             return out_dict

#         loss_dict = dict()
#         for i in range(len(paf_out)):
#             if 'paf_loss{}'.format(i) in self.valid_loss_dict:
#                 loss_dict['paf_loss{}'.format(i)] = dict(
#                     params=[paf_out[i]*data_dict['maskmap'], data_dict['vecmap']*data_dict['maskmap']],
#                     type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
#                     weight=torch.cuda.FloatTensor([self.valid_loss_dict['paf_loss{}'.format(i)]])
#                 )

        # for i in range(len(heatmap_out)):
        #     if 'heatmap_loss{}'.format(i) in self.valid_loss_dict:
        #         loss_dict['heatmap_loss{}'.format(i)] = dict(
        #             params=[heatmap_out[i]*data_dict['maskmap'], data_dict['heatmap']*data_dict['maskmap']],
        #             type=torch.cuda.LongTensor([BASE_LOSS_DICT['mse_loss']]),
        #             weight=torch.cuda.FloatTensor([self.valid_loss_dict['heatmap_loss{}'.format(i)]])
        #         )

#         return out_dict, loss_dict