from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import SAMMed2DImageEncoder
from segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer, Sam


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


def build_sammed2D(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_size,
    checkpoint,
    encoder_adapter,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=SAMMed2DImageEncoder(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos = True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            adapter_train = encoder_adapter,
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
        if "model" in state_dict.keys():
            state_dict = state_dict["model"]
        if image_size != 256:
            print("Interpolating Position Embedding!")
            state_dict = interpolate_state_dict(
                sam, state_dict, image_embedding_size, encoder_global_attn_indexes
            )
        sam.load_state_dict(state_dict)
    return sam


def build_sammed2D_b(checkpoint=None):
    return build_sammed2D(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=256,
        checkpoint=checkpoint,
        encoder_adapter=True
    )