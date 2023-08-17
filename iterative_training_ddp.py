import os
import time
import datetime
import logging
logging.basicConfig(level=logging.INFO)
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import torch
import torch.nn as nn
from torch.optim import *

from segmentation_models_pytorch.losses import DiceLoss

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from src.dataset import create_dataloader, sample_box, filter_box, sample_center_point, uniform_sample_points
from src.losses import CombinedLoss
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
    USE_BOX_PROMPT = False
    USE_CENTER_POINT = True

    MAX_EPOCHS = 200
    ROUND_PER_EPOCH = 6
    LR = 1e-4
    WEIGHT_DECAY = 0.001
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

    device = model.device

    # Training loop

    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        epoch_losses = []

        # One epoch
        for batch in tqdm(train_loader, desc=f"epoch: {epoch}"):
            images = batch[0]  # (B, C, H, W)
            masks = batch[1]  # (B, C, H, W)
            point_prompts = batch[2]  # (B, num_boxes, points_per_box, 2)
            point_labels = batch[3]  # (B, num_boxes, points_per_box)
            box_prompts = batch[4]  # (B, num_boxes, 4)
            image_size = (train_dataset.image_size, train_dataset.image_size)
            batch_size = images.shape[0]  # Batch size

            input_images = torch.stack([model.module.preprocess(img) for img in images], dim=0)

            # Forward batch, process per-image
            batch_loss = []
            for image, gt_mask, point_prompt, point_label, box_prompt in zip(
                    input_images, masks, point_prompts, point_labels, box_prompts
            ):
                # Prepare round 0 inputs
                round_loss = 0
                mask_input = None
                point = (point_prompt, point_label)
                for round in range(ROUND_PER_EPOCH):
                    model_input = {
                        "image": image,
                        "point_prompt": point,
                        "box_prompt": box_prompt if USE_BOX_PROMPT else None,
                        "mask_input": mask_input,
                        "image_size": image_size
                    }
                    # low_res_mask (num_objects, num_preds, 256, 256)
                    # iou_predictions (num_objects, num_preds)
                    low_res_masks, iou_predictions = model(model_input)

                    # Select the mask with highest IoU for each object
                    max_idx = torch.argmax(iou_predictions, dim=1)
                    _selected_masks = low_res_masks[torch.arange(low_res_masks.shape[0]), max_idx] # (num_objects, 256, 256)
                    selected_masks = _selected_masks.unsqueeze(1)  # (num_objects, 1, 256, 256)

                    # Calculate loss with the selected_masks
                    upscaled_masks = model.module.postprocess_masks(
                        selected_masks, image.shape[-2:], image_size
                    )
                    loss = loss_fn(upscaled_masks, gt_mask[:, None, :, :])  # expand channel dim
                    accelerator.backward(loss)
                    round_loss += loss.item()
                    # TODO: check on this after, is this correct?
                    # selected_masks = selected_masks.detach() # Detach from the computation grad of next round

                    # Find the error region mask between selected_masks and ground truth, then sample points
                    with torch.no_grad():
                        # Step 1: convert mask to binary
                        upscaled_masks = upscaled_masks > model.module.mask_threshold
                        gt = gt_mask > 0.5  # Cuz resizing image sucks
                        # Step 2: OR all mask to reduce to C=1
                        single_upscale_mask = upscaled_masks[0, 0, ...].clone()  # (1024, 1024)
                        single_gt_mask = gt[0, ...].clone()  # (1024, 1024)
                        for i in range(1, upscaled_masks.shape[0]):
                            single_upscale_mask = torch.logical_or(single_upscale_mask,
                                                                   upscaled_masks[i, 0, ...])  # (1024, 1024)
                            single_gt_mask = torch.logical_or(single_gt_mask, gt[i, ...])  # (1024, 1024)
                        single_upscale_mask = single_upscale_mask.long()
                        single_gt_mask = single_gt_mask.long()
                        # Step 2: Find the error region
                        # Error_mask will have value of:
                        # -1: On false positive pixel (predict mask but wrong)
                        # 0: On true positive and true negative pixel
                        # 1: On false negative pixel (predict none but has mask)
                        error_mask = single_gt_mask - single_upscale_mask  # (1024, 1024)
                        # Step 4: sample points
                        # Step 4.1: Separate the error mask into 2 part: The false positive and the false negative ones
                        false_positive_mask = torch.where(error_mask == -1, error_mask, 0)
                        false_positive_mask = -false_positive_mask
                        false_negative_mask = torch.where(error_mask == 1, error_mask, 0)
                        # Step 4.2: Choose a mask to sample from
                        if (np.random.rand() >= 0.5):  # sample from false negative mask
                            mask_to_sample = false_negative_mask
                            rand = 1
                        else:
                            mask_to_sample = false_positive_mask
                            rand = -1
                        # Step 4.3: RANDOMLY Sample point from mask
                        height_point_prompt, width_point_prompt = uniform_sample_points(mask_to_sample,
                                                                                        num_points=1)
                        _point_prompt = torch.hstack([height_point_prompt, width_point_prompt])  # (1, 2)
                        if _point_prompt.shape[0] <= 0: # can't sample any points
                            # Resample with different mask
                            if rand == 1:
                                mask_to_sample = false_positive_mask
                                rand = -1
                            else:
                                mask_to_sample = false_negative_mask
                                rand = 1
                            height_point_prompt, width_point_prompt = uniform_sample_points(mask_to_sample,
                                                                                            num_points=1)
                            _point_prompt = torch.hstack([height_point_prompt, width_point_prompt])  # (1, 2)
                            if _point_prompt.shape[0] <= 0: # If still no points -> 100% correct mask
                                break # Exit the Loop early
                        _point_prompt = _point_prompt.unsqueeze(0)  # (1, 1, 2)
                        if rand == 1:  # If sampled from false negative, insert label 1
                            _point_label = torch.ones((1,))
                        else:
                            _point_label = torch.zeros((1,))
                        _point_label = _point_label.unsqueeze(0)  # (1, 1)

                        new_point_prompts = torch.as_tensor(_point_prompt, device=device, dtype=torch.float)
                        new_point_labels = torch.as_tensor(_point_label, device=device, dtype=torch.int)

                        # Step 5: Update the input for next round
                        point = (new_point_prompts, new_point_labels)
                        mask_input = selected_masks
                # End of all round
                batch_loss.append(round_loss)

            # After batch
            optimizer.step()
            optimizer.zero_grad()
            batch_loss = sum(batch_loss) / batch_size
            epoch_losses.append(batch_loss)

        # After epoch
        scheduler.step()
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logging.info(f"Epoch: {epoch} \t Loss: {epoch_loss}")

        # Saving
        if epoch >= EPOCH_TO_SAVE and epoch % SAVE_FREQUENCY == 0:
            torch.save(sam.state_dict(), f"{save_folder}/ckpts/{epoch}.pt")

    end_time = time.time()
    logging.info(f"Training time: {(end_time - start_time)/3600:.2f}")


if __name__ == '__main__':
    main()