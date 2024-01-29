from typing import Type
from segment_anything.modeling import MaskDecoder, TwoWayTransformer
from torch.nn.modules import GELU, Module
from typing import Tuple,List
import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
class LogoMaskDecoder(MaskDecoder):
    def __init__(self):
        super().__init__(transformer_dim=256, 
                    transformer=TwoWayTransformer(
                                    depth=2,
                                    embedding_dim=256,
                                    mlp_dim=2048,
                                    num_heads=8,
                                ),
                    num_multimask_outputs=3, 
                    activation=GELU, 
                    iou_head_depth=3, 
                    iou_head_hidden_dim=256)


        # self.gl_token=nn.Embedding(1,transformer_dim)
        transformer_dim=256
        self.local_encoder=nn.Sequential(
                                    nn.Conv2d(768,transformer_dim,1,bias=False),
                                    LayerNorm2d(transformer_dim),
                                    nn.Conv2d(transformer_dim,transformer_dim,kernel_size=3,padding=1,bias=False),
                                    LayerNorm2d(transformer_dim)
                                )

        self.global_encoder = nn.Sequential(
                                    nn.Conv2d(transformer_dim, transformer_dim*2, 3, 1, 1), 
                                    LayerNorm2d(transformer_dim*2),
                                    nn.GELU(),
                                    nn.Conv2d(transformer_dim*2, transformer_dim, 3, 1, 1))
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        vit_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
            image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
            image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
            dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
            multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
            torch.Tensor: batched predicted final masks
        """
        local_features=vit_embeddings[0].permute(0,3,1,2)
        gl_features=self.local_encoder(local_features)+self.global_encoder(image_embeddings)
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.gl_token.weight], dim=0)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(gl_features, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

                # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
