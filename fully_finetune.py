import os
import time
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
from src.models import PolypSAM

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
    LR = 4e-6
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 2

    SAVE_PATH = "workdir/train/"
    EPOCH_TO_SAVE = 100
    SAVE_FREQUENCY = 10

    # Save
    date = datetime.date.today().strftime("%Y-%m-%d")
    _time = datetime.datetime.now().strftime("%H%M%S")
    time_str = date + "_" + _time
    save_folder = f"{SAVE_PATH}/{time_str}"
    os.makedirs(f"{save_folder}/ckpts", exist_ok=True)

    # Init Accelerate
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Model
    sam: Sam = sam_model_registry[MODEL_SIZE](PRETRAINED_PATH)
    model = PolypSAM(sam.image_encoder,
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

    start_time = time.time()

    # Training loop
    for epoch in range(1, MAX_EPOCHS):
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
                # Prepare input
                point = (point_prompt, point_label)
                model_input = {
                    "image": image,
                    "point_prompt": point,
                    "box_prompt": box_prompt if USE_BOX_PROMPT else None,
                    "image_size": image_size
                }
                pred_masks = model(model_input)

                loss = loss_fn(pred_masks, gt_mask[:, None, :, :]) # expand channel dim
                accelerator.backward(loss)
                batch_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(batch_loss / batch_size)

        scheduler.step()
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logging.info(f"Epoch: {epoch} \t Loss: {epoch_loss}")

        # Saving
        if epoch >= EPOCH_TO_SAVE and epoch % SAVE_FREQUENCY == 0:
            torch.save(sam.state_dict(), f"{save_folder}/ckpts/{epoch}.pt")

    end_time = time.time()
    logging.info(f"Training time: {(end_time - start_time) / 3600:.2f}")


if __name__ == '__main__':
    main()