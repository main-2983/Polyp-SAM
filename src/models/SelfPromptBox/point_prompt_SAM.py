# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Any, Dict, List, Tuple

# from src.models.SelfPromptBox.detection_head import DetectionHead
from src.models.SelfPromptBox.point_predictor import PointDetector
from src.models.SelfPromptBox.position_encoding import *
from src.models.iterative_polypSAM import IterativePolypSAM
from torch.nn import functional as F
from tools.box_ops import box_cxcywh_to_xyxy

class SelfPointPromptSam(IterativePolypSAM):
    def __init__(
        self,
        point_detector:PointDetector,
        pos_encoder,
                 *args,
                 **kwargs):
        super(SelfPointPromptSam, self).__init__(*args, **kwargs)
        """
        """
        self.point_detector = point_detector
        self.pos_encoder=pos_encoder
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
            boxes=input.get("box_prompt"),
            masks=input.get("mask_input", None),
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return low_res_masks, iou_predictions
    
    def forward_point(self,input_img):
        image = input_img  # [B, C, 1024, 1024]
        # image = torch.stack([self.preprocess(img) for img in image], dim=0)
        image_embeddings = self.image_encoder(image)
        pos_embedding=self.pos_encoder(image_embeddings)
        out=self.point_detector(image_embeddings.detach(),pos=pos_embedding)
        return out 
    
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