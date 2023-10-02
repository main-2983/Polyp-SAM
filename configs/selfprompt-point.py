from glob import glob

import torch
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import sys
# sys.path.append('/home/dang.hong.thanh/Polyp-SAM/segment-anything/')
from segment_anything.modeling import Sam
from segment_anything import sam_model_registry

from src.models.SelfPromptPoint import SelfPointPromptSAM, PointGenModule
from src.scheduler import LinearWarmupCosineAnnealingLR
from src.losses import CombinedLoss
from src.datasets.polyp.polyp_dataset import PolypDataset


class Config:
    def __init__(self):
        # Model init
        PRETRAINED_PATH = "ckpts/sam_vit_b_01ec64.pth"
        MODEL_SIZE = "vit_b"

        # Model
        sam: Sam = sam_model_registry[MODEL_SIZE](PRETRAINED_PATH)
        point_gen = PointGenModule()
        self.model = SelfPointPromptSAM(point_gen,
                                        sam.image_encoder,
                                        sam.mask_decoder,
                                        sam.prompt_encoder,
                                        freeze=[sam.image_encoder, sam.mask_decoder, sam.prompt_encoder])

        # Dataset and Dataloader
        IMG_PATH = "/home/dang.hong.thanh/sun_sam_polyp/Dataset/TrainDataset/image/*"
        MASK_PATH = "/home/dang.hong.thanh/sun_sam_polyp/Dataset/TrainDataset/mask/*"
        self.IMAGE_SIZE = 1024
        self.EMBEDDING_PATHS = None
        self.dataset = PolypDataset(
            glob(IMG_PATH),
            glob(MASK_PATH),
            embedding_paths=self.EMBEDDING_PATHS,
            image_size=self.IMAGE_SIZE
        )

        self.BATCH_SIZE = 1
        self.NUM_WORKERS = 8

        # Training
        self.MAX_EPOCHS = 20

        # Optimizer
        self.ACCUMULATE_STEPS = 8
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
        loss1 = DiceLoss(mode="binary", from_logits=True)
        loss2 = FocalLoss(mode="binary")
        self.LOSS_FN = CombinedLoss(
            [loss1, loss2]
        )

        # Save
        self.SAVE_PATH = "workdir/train/Self-Prompt-Point"