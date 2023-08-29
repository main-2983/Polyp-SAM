from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling import ImageEncoderViT


class PolypCounter(nn.Module):
    def __init__(self,
                 image_encoder: ImageEncoderViT,
                 pixel_mean: List[float] = [0.485, 0.456, 0.406],
                 pixel_std: List[float] = [0.229, 0.224, 0.225],
                 freeze: List[nn.Module] = None):
        super(PolypCounter, self).__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.image_encoder = image_encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(256, 128, 1)
        self.relu = nn.ReLU()
        self.pred = nn.Conv2d(128, 1, 1)

        if freeze is not None:
            for module in freeze:
                if isinstance(self.image_encoder, type(module)):
                    for param in self.image_encoder.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = self.image_encoder(x)
        out = self.pool(enc_out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.pred(out)

        return out

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
