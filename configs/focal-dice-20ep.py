from glob import glob

import torch
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

from src.losses import CombinedLoss
from src.base_config import Config as BaseConfig
from src.datasets import PromptPolypDataset
from src.scheduler import LinearWarmupCosineAnnealingLR


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        IMG_PATH = "/home/nguyen.mai/Dataset/Polyp/TrainDataset/image/*"
        MASK_PATH = "/home/nguyen.mai/Dataset/Polyp/TrainDataset/mask/*"
        self.USE_BOX_PROMPT = False
        USE_CENTER_POINT = True
        self.dataset = PromptPolypDataset(
            glob(IMG_PATH),
            glob(MASK_PATH),
            image_size=self.IMAGE_SIZE,
            use_box_prompt=self.USE_BOX_PROMPT,
            use_center_points=USE_CENTER_POINT
        )
        
        self.NUM_WORKERS = 8
        self.BATCH_SIZE = 16

        # Training
        self.MAX_EPOCHS = 20
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
            warmup_start_lr=5e-5,
            eta_min=1e-5
        )

        # Save
        self.EPOCH_TO_SAVE = 4
        self.SAVE_FREQUENCY = 2

        loss1 = DiceLoss(mode="binary", from_logits=True)
        loss2 = FocalLoss(mode="binary")
        self.LOSS_FN = CombinedLoss(
            [loss1, loss2]
        )