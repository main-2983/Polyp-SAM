from glob import glob

import torch
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import sys
sys.path.append('/home/trinh.quang.huy/polyp_for_sam/Polyp-SAM/segment-anything')
from segment_anything.modeling import Sam
from segment_anything import sam_model_registry
from torch.nn import BCEWithLogitsLoss, MSELoss

from src.models.SelfPromptBox.box_prompt_SAM import SelfBoxPromptSam
from src.models.SelfPromptPoint import SelfPointPromptSAM, PointGenModule

from src.scheduler import LinearWarmupCosineAnnealingLR
from src.losses import CombinedLoss
from src.datasets.polyp.polyp_dataset import PolypDataset
from src.datasets.polyp.Box_dataloader import PromptBaseDataset
from src.models.SelfPromptBox.detection_head import DetectionHead


class Config:
    def __init__(self):
        # Model init
        PRETRAINED_PATH = "ckpts/sam_vit_b_01ec64.pth"
        MODEL_SIZE = "vit_b"

        # Model
        sam: Sam = sam_model_registry[MODEL_SIZE](PRETRAINED_PATH)
        self.box_decoder=DetectionHead(        
                            hidden_dim=256,
                            nhead=8,
                            num_classes=1,
                            dim_feedforward=2048,
                            num_queries=100,)
        self.model = SelfBoxPromptSam(self.box_decoder,
                                    sam.image_encoder,
                                    sam.prompt_encoder,
                                    sam.mask_decoder)
        point_gen = PointGenModule()
        # self.model= SelfPointPromptSAM(point_gen,
        #                                 sam.image_encoder,
        #                                 sam.mask_decoder,
        #                                 sam.prompt_encoder,
        #                                 freeze=[sam.image_encoder, sam.mask_decoder, sam.prompt_encoder])

        # Dataset and Dataloader
        IMG_PATH = "/home/dang.hong.thanh/sun_sam_polyp/Dataset/TrainDataset/image/*"
        MASK_PATH = "/home/dang.hong.thanh/sun_sam_polyp/Dataset/TrainDataset/mask/*"
        self.IMAGE_SIZE = 1024
        self.EMBEDDING_PATHS = None
        self.dataset = PromptBaseDataset(
            glob(IMG_PATH),
            glob(MASK_PATH),
            image_size=self.IMAGE_SIZE
        )
        self.USE_BOX_PROMPT = False

        self.BATCH_SIZE = 1
        self.NUM_WORKERS = 8

        # Training
        self.MAX_EPOCHS = 200
        self.ROUND_PER_EPOCH = 6

        # Optimizer
        self.ACCUMULATE_STEPS = 1
        self.OPTIMIZER = torch.optim.AdamW
        self.OPTIMIZER_KWARGS = dict(
            lr=1e-4,
            weight_decay=0.001
        )
        self.SCHEDULER = LinearWarmupCosineAnnealingLR
        self.SCHEDULER_KWARGS = dict(
            warmup_epochs=5,
            max_epochs=self.MAX_EPOCHS,
            warmup_start_lr=5e-7,
            eta_min=1e-6
        )

        # Loss
        self.IOU_LOSS = MSELoss()
        loss1 = DiceLoss(mode="binary", from_logits=True)
        loss2 = FocalLoss(mode="binary")
        self.LOSS_FN = CombinedLoss(
            [loss1, loss2]
        )

        # Save
        self.SAVE_PATH = "workdir/train/Self-Prompt-Box"
        self.RATE = 0.5
