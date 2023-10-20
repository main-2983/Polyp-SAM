from glob import glob
from torch.optim import lr_scheduler
import torch
from src.scheduler import LinearWarmupCosineAnnealingLR
from segment_anything.modeling import Sam
from segment_anything import sam_model_registry
from src.models.SelfPromptPointHeatmap.heatmap_model import PointModelHeatmap
from src.models.SelfPromptPointHeatmap import SelfPointPromptWithHeatmap
from src.datasets.polyp.dataloader_heatmap import PromptDatasetHeatmap
from src.models.SelfPromptPointHeatmap.loss import Loss

class Config:
    def __init__(self):
        # Model init
        # PRETRAINED_PATH = "/home/trinh.quang.huy/polyp_for_sam/Polyp-SAM/ckpts/sam_vit_b_01ec64.pth"
        PRETRAINED_PATH = None
        MODEL_SIZE = "vit_b"

        # Model
        sam: Sam = sam_model_registry[MODEL_SIZE](PRETRAINED_PATH)
        heat_model = PointModelHeatmap()
        self.model = SelfPointPromptWithHeatmap(heat_model,
                                        sam.image_encoder,
                                        sam.mask_decoder,
                                        sam.prompt_encoder,
                                        freeze=[sam.image_encoder, sam.mask_decoder, sam.prompt_encoder])

        # Dataset and Dataloader
        IMG_PATH = "/home/trinh.quang.huy/sun_polyp_dataset/Dataset/TrainDataset/image/*"
        MASK_PATH = "/home/trinh.quang.huy/sun_polyp_dataset/Dataset/TrainDataset/mask/*"
        self.IMAGE_SIZE = 1024
        self.MASK_SIZE = 64
        self.EMBEDDING_PATHS = None
        self.dataset = PromptDatasetHeatmap(
            glob(IMG_PATH),
            glob(MASK_PATH),
            image_size=self.IMAGE_SIZE,
            mask_size=self.MASK_SIZE
        )

        self.BATCH_SIZE = 1
        self.NUM_WORKERS = 8

        # Training
        self.MAX_EPOCHS = 50

        # Optimizer
        self.ACCUMULATE_STEPS = 8
        self.OPTIMIZER = torch.optim.AdamW
        self.OPTIMIZER_KWARGS = dict(
            lr=5e-4,
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
        self.LOSS_FN = Loss()

        # Save
        self.SAVE_PATH = "workdir/train/Self-Prompt-Point"