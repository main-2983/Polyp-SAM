import os
import datetime
import logging
logging.basicConfig(level=logging.INFO)
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import *

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from src.dataset import create_dataloader
from src.losses import StructureLoss
from src.scheduler import LinearWarmupCosineAnnealingLR

try:
    import wandb
except ImportError:
    wandb = None


def main():
    # Wandb logging

    # Const
    # TODO: Make config
    MAX_EPOCHS = 200
    LR = 4e-6
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 2

    SAVE_PATH = "workdir/train/"

    # Save
    date = datetime.date.today().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H%M%S")
    time_str = date + "_" + time
    save_folder = f"{SAVE_PATH}/{time_str}"
    os.makedirs(save_folder, exist_ok=True)

    # Model
    sam: Sam = sam_model_registry["vit_b"]
    device = "cuda:0"
    sam.to(device)

    # Dataloader
    train_loader = create_dataloader(glob("/home/nguyen.mai/Workplace/sun-polyp/Dataset/TrainDataset/image/*"),
                                     glob("/home/nguyen.mai/Workplace/sun-polyp/Dataset/TrainDataset/mask/*"),
                                     batch_size=BATCH_SIZE,
                                     num_workers=0)
    # Loss
    loss = StructureLoss()

    # Optimizer
    optimizer = AdamW(
        sam.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=30, max_epochs=MAX_EPOCHS,
                                              warmup_start_lr=5e-7, eta_min=1e-6)

    # Training loop
    for epoch in range(MAX_EPOCHS):
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"epoch: {epoch}"):
            image         = batch["image"]
            mask          = batch["mask"]
            point_prompts = batch["point_prompts"]
            point_labels  = batch["point_labels"]
            box_prompts   = batch["box_prompts"]

            # Image encoder
            input_images = torch.stack([sam.preprocess(img) for img in image], dim=0)
            image_embeddings = sam.image_encoder(input_images)

            outputs = []



