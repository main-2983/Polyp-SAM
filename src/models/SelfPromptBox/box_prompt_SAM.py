# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Any, Dict, List, Tuple

from src.models.SelfPromptBox.detection_head import DetectionHead
from src.models.SelfPromptBox.position_encoding import *
from src.models.iterative_polypSAM import IterativePolypSAM

class SelfBoxPromptSam(IterativePolypSAM):
    def __init__(
        self,
        box_decoder:DetectionHead,
        pos_encoder,
                 *args,
                 **kwargs):
        super(SelfBoxPromptSam, self).__init__(*args, **kwargs)
        """
        """
        self.box_decoder = box_decoder
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
    
    def forward_box(self,input_img):
        image = input_img  # [B, C, 1024, 1024]
        # image = torch.stack([self.preprocess(img) for img in image], dim=0)
        image_embeddings = self.image_encoder(image)
        pos_embedding=self.pos_encoder(image_embeddings)
        out=self.box_decoder(image_embeddings.detach(),pos=pos_embedding)
        return out 
    
