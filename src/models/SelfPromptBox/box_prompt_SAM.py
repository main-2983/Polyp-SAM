# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple
import sys

from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from src.models.SelfPromptBox.detection_head import DetectionHead
from src.models.SelfPromptBox.position_encoding import *
from tools.box_ops import box_cxcywh_to_xyxy
from src.models.iterative_polypSAM import IterativePolypSAM

class SelfBoxPromptSam(IterativePolypSAM):
    def __init__(
        self,
        box_decoder:DetectionHead,
                 *args,
                 **kwargs):
        super(SelfBoxPromptSam, self).__init__(*args, **kwargs)
        """
        """
        self.box_decoder = box_decoder
    def forward_mask(
        self,
        input: List[Dict[str, Any]],
        multimask_output: bool= True,
    ) -> List[Dict[str, torch.Tensor]]:

        image = input.get("image")  # [1, 1, 1024, 1024]

        image_embeddings = self.image_encoder(image[None])
        points = input.get("point_prompt")
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=input.get("mask_input", None),
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return low_res_masks,iou_predictions
    
    def forward_box(self,input: List[Dict[str, Any]]):
        image = input.get("image")  # [1, 1, 1024, 1024]
        # image = torch.stack([self.preprocess(img) for img in image], dim=0)
        image_embeddings = self.image_encoder(image[None])
        out=self.box_decoder(image_embeddings.detach())
        postprocessors = {'bbox': PostProcess()}
        orig_target_sizes= torch.stack([torch.tensor(input.get('image_size')).to(image.device)])
        results=postprocessors['bbox'](out, target_sizes=orig_target_sizes)
        return results 
    
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results
