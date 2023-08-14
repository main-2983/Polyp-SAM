import os
import datetime
import logging
logging.basicConfig(level=logging.INFO)
from glob import glob
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import torch
import torch.nn as nn
from torch.optim import *

from segmentation_models_pytorch.losses import DiceLoss

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from src.dataset import create_dataloader
from src.losses import StructureLoss, CombinedLoss
from src.scheduler import LinearWarmupCosineAnnealingLR
from src.models import IterativePolypSAM

try:
    import wandb
except ImportError:
    wandb = None


def main():
    # Wandb logging

    # Const
    # TODO: Make config
    PRETRAINED_PATH = "ckpts/sam_vit_b_01ec64.pth"
    MODEL_SIZE = "vit_b"

    IMG_PATH = "/home/nguyen.mai/Workplace/sun-polyp/Dataset/TrainDataset/image/*"
    MASK_PATH = "/home/nguyen.mai/Workplace/sun-polyp/Dataset/TrainDataset/mask/*"
    NUM_WORKERS = 0
    USE_BOX_PROMPT = True
    USE_CENTER_POINT = True

    MAX_EPOCHS = 200
    ITERATION_PER_EPOCH = 6
    LR = 4e-6
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 2

    SAVE_PATH = "workdir/train/"
    EPOCH_TO_SAVE = 100
    SAVE_FREQUENCY = 10

    # Save
    date = datetime.date.today().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H%M%S")
    time_str = date + "_" + time
    save_folder = f"{SAVE_PATH}/{time_str}"
    os.makedirs(f"{save_folder}/ckpts", exist_ok=True)

    # Init Accelerate
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Model
    sam: Sam = sam_model_registry[MODEL_SIZE](PRETRAINED_PATH)
    model = IterativePolypSAM(sam.image_encoder,
                              sam.mask_decoder,
                              sam.prompt_encoder)

    # Dataloader
    train_dataset, train_loader = create_dataloader(
        glob(IMG_PATH),
        glob(MASK_PATH),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_box_prompt=USE_BOX_PROMPT,
        use_center_points=USE_CENTER_POINT
    )

    # Loss
    loss_fn = CombinedLoss([
        DiceLoss(mode="binary", from_logits=True),
        nn.BCEWithLogitsLoss(reduction='mean')
    ])

    # Optimizer
    optimizer = AdamW(
        sam.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=30, max_epochs=MAX_EPOCHS,
                                              warmup_start_lr=5e-7, eta_min=1e-6)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop
    for epoch in range(MAX_EPOCHS):
        epoch_losses = []

        # One epoch
        for batch in tqdm(train_loader, desc=f"epoch: {epoch}"):
            images        = batch[0] # (B, C, H, W)
            masks         = batch[1] # (B, C, H, W)
            point_prompts = batch[2] # (B, num_boxes, points_per_box, 2)
            point_labels  = batch[3] # (B, num_boxes, points_per_box)
            box_prompts   = batch[4] # (B, num_boxes, 4)
            image_size    = (train_dataset.image_size, train_dataset.image_size)
            batch_size    = images.shape[0] # Batch size

            input_images = torch.stack([model.module.preprocess(img) for img in images], dim=0)

            # Forward batch, process per-image
            batch_loss = 0
            for image, gt_mask, point_prompt, point_label, box_prompt in zip(
                input_images, masks, point_prompts, point_labels, box_prompts
            ):
                # Prepare iteration 0 inputs
                mask_input = None
                point = (point_prompt, point_label)
                for iter in range(ITERATION_PER_EPOCH):
                    model_input = {
                        "image": image,
                        "point_prompt": point,
                        "box_prompt": box_prompt if USE_BOX_PROMPT else None,
                        "mask_input": mask_input,
                        "image_size": image_size
                    }
                    low_res_masks, iou_predictions = model(model_input)

                    # Select the mask with highest IoU
