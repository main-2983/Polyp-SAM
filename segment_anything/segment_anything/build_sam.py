# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_embedding_size=None,
    image_size=1024,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_embedding_size if image_embedding_size is not None else image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        if image_size != 1024:
            print("Interpolating Position Embedding!")
            state_dict = interpolate_state_dict(
                sam, state_dict, image_embedding_size, encoder_global_attn_indexes
            )
        sam.load_state_dict(state_dict)
    return sam


def interpolate_state_dict(sam: Sam,
                           pretrained_state_dict,
                           token_size,
                           global_attention_indexes: list):
    state_dict = sam.state_dict()
    pos_embed = pretrained_state_dict["image_encoder.pos_embed"]

    if pos_embed.shape[1] != token_size:
        # interpolate the pre-trained position embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        pretrained_state_dict["image_encoder.pos_embed"] = pos_embed # update pretrained SAM with interpolated pos_embed
        # interpolate the rel_pos of global attention (local attention has fixed rel_pos shape with window size)
        rel_pos_keys = [k for k in state_dict.keys() if "rel_pos" in k]
        # get all global attention keys
        global_rel_pos_keys = []
        for i in range(len(global_attention_indexes)):
            block_i = str(global_attention_indexes[i])
            check_string = f"blocks.{block_i}"
            for k in rel_pos_keys:
                if check_string in k:
                    print(k)
                    global_rel_pos_keys.append(k)

        for k in global_rel_pos_keys:
            target_h, target_w = state_dict[k].shape
            rel_pos_params = pretrained_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            if h != target_h or w != target_w:
                rel_pos_params = F.interpolate(rel_pos_params, (target_h, target_w), mode='bilinear', align_corners=False)

            pretrained_state_dict[k] = rel_pos_params[0, 0, ...]

    # update init SAM with pretrained
    state_dict.update(pretrained_state_dict)

    return state_dict