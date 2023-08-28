from typing import Tuple

import torch

from segment_anything.modeling import ImageEncoderViT


class MultiFeatEncoder(ImageEncoderViT):
    def __init__(self,
                 *args,
                 **kwargs):
        super(MultiFeatEncoder, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        features = []
        for blk in self.blocks:
            x = blk(x) # (B, H, W, C)
            if blk.window_size == 0: # Global attention (end of one stage)
                features.append(x.permute(0, 3, 1, 2)) # (B, C, H, W)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x, features
