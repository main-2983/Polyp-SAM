import os
import datetime
import logging
logging.basicConfig(level=logging.INFO)
from glob import glob
from tqdm import tqdm

import torch
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
    IMG_PATH = "/home/nguyen.mai/Workplace/sun-polyp/Dataset/TrainDataset/image/*"
    MASK_PATH = "/home/nguyen.mai/Workplace/sun-polyp/Dataset/TrainDataset/mask/*"

    MAX_EPOCHS = 200
    LR = 4e-6
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 2

    SAVE_PATH = "workdir/train/"
    USE_BOX_PROMPT = False

    # Save
    date = datetime.date.today().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H%M%S")
    time_str = date + "_" + time
    save_folder = f"{SAVE_PATH}/{time_str}"
    os.makedirs(save_folder, exist_ok=True)

    # Model
    sam: Sam = sam_model_registry["vit_b"]()
    sam.pixel_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    sam.pixel_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    device = "cpu"
    sam.to(device)

    # Dataloader
    train_dataset, train_loader = create_dataloader(
        glob(IMG_PATH),
        glob(MASK_PATH),
        batch_size=BATCH_SIZE,
        num_workers=0,
        use_box_prompt=USE_BOX_PROMPT
    )
    # Loss
    loss_fn = StructureLoss()

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

        # One epoch
        for batch in tqdm(train_loader, desc=f"epoch: {epoch}"):
            images        = batch[0] # (B, C, H, W)
            masks         = batch[1] # (B, C, H, W)
            point_prompts = batch[2] # (B, num_points, 2)
            point_labels  = batch[3] # (B, num_points)
            box_prompts   = batch[4] # (B, num_boxes, 4)
            image_size    = (train_dataset.image_size, train_dataset.image_size)
            batch_size    = images.shape[0] # Batch size

            # Put to device
            images = images.to(device)
            masks  = masks.to(device)
            point_prompts = point_prompts.to(device)
            point_labels = point_labels.to(device)
            box_prompts = box_prompts.to(device)

            input_images = torch.stack([sam.preprocess(img) for img in images], dim=0)

            # Forward batch, process per-image
            batch_loss = 0
            for image, gt_mask, point_prompt, point_label, box_prompt in zip(
                input_images, masks, point_prompts, point_labels, box_prompts
            ):
                image_embedding = sam.image_encoder(image[None])
                point = (point_prompt[None, :, :], point_label[None, :]) # Expand batch dim
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=point,
                    boxes=box_prompt if USE_BOX_PROMPT else None,
                    masks=None
                )
                low_res_masks, iou_predictions = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_masks = sam.postprocess_masks(
                    low_res_masks,
                    input_size=image.shape[-2:],
                    original_size=image_size,
                )
                binary_masks = torch.sigmoid(pred_masks)

                loss = loss_fn(binary_masks, gt_mask[None]) # expand batch dim
                loss.backward()
                batch_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(batch_loss / batch_size)

        scheduler.step()
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logging.info(f"Epoch: {epoch} \t Loss: {epoch_loss}")

if __name__ == '__main__':
    main()