import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch.nn import BCEWithLogitsLoss

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry
from src.model.iterative_polypSAM import IterativePolypSAM
from src.scheduler import LinearWarmupCosineAnnealingLR
from src.losses import CombinedLoss


class Config:
    def __init__(self):
        # Model init
        self.PRETRAINED_PATH = "ckpts/sam_vit_b_01ec64.pth"
        self.MODEL_SIZE = "vit_b"

        # Dataset and Dataloader
        self.IMG_PATH = "/home/nguyen.mai/Workplace/sun-polyp/Dataset/TrainDataset/image/*"
        self.MASK_PATH = "/home/nguyen.mai/Workplace/sun-polyp/Dataset/TrainDataset/mask/*"
        self.NUM_WORKERS = 0
        self.USE_BOX_PROMPT = False
        self.USE_CENTER_POINT = True
        self.BATCH_SIZE = 2

        # Training
        self.MAX_EPOCHS = 200
        self.ROUND_PER_EPOCH = 6

        # Model
        self.sam: Sam = sam_model_registry[self.MODEL_SIZE](self.PRETRAINED_PATH)
        self.model = IterativePolypSAM(self.sam.image_encoder,
                                       self.sam.mask_decoder,
                                       self.sam.prompt_encoder)

        # Optimizer
        self.OPTIMIZER = torch.optim.AdamW
        self.OPTIMIZER_KWARGS = dict(
            lr=1e-4,
            weight_decay=0.001
        )
        self.SCHEDULER = LinearWarmupCosineAnnealingLR
        self.SCHEDULER_KWARGS = dict(
            warmup_epochs=30,
            max_epochs=self.MAX_EPOCHS,
            warmup_start_lr=5e-7,
            eta_min=1e-6
        )

        # Loss
        loss1 = DiceLoss(mode="binary", from_logits=True)
        loss2 = BCEWithLogitsLoss(reduction='mean')
        self.LOSS_FN = CombinedLoss(
            [loss1, loss2]
        )

        # Sampling
        self.RATE = 0.5

        # Save
        self.SAVE_PATH = "workdir/train/"
        self.EPOCH_TO_SAVE = 100
        self.SAVE_FREQUENCY = 10