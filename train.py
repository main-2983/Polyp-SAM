import argparse
import os
import time
import datetime
import importlib
import shutil
from glob import glob
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")
from accelerate.utils import DistributedDataParallelKwargs

import torch

from src.dataset import create_dataloader

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default="src.base_config", help="where to get config module")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    module = importlib.import_module(args.config)
    config = module.Config()

    # Save
    date = datetime.date.today().strftime("%Y-%m-%d")
    _time = datetime.datetime.now().strftime("%H%M%S")
    time_str = date + "_" + _time
    save_folder = f"{config.SAVE_PATH}/{time_str}"
    os.makedirs(f"{save_folder}/ckpts", exist_ok=True)

    # Write config
    shutil.copy(module.__file__, save_folder)

    # Init Accelerate
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Model
    model = config.model

    # Dataloader
    train_dataset, train_loader = create_dataloader(
        config.dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # Loss
    loss_fn = config.LOSS_FN

    # Optimizer
    optimizer = config.OPTIMIZER(model.parameters(), **config.OPTIMIZER_KWARGS)
    scheduler = config.SCHEDULER(optimizer, **config.SCHEDULER_KWARGS)

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    # Training loop

    start_time = time.time()

    # Training loop
    for epoch in range(1, config.MAX_EPOCHS + 1):
        epoch_losses = []

        # One epoch
        for batch in tqdm(train_loader, desc=f"epoch: {epoch}", disable=not accelerator.is_main_process):
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
                    "box_prompt": box_prompt if config.USE_BOX_PROMPT else None,
                    "image_size": image_size
                }
                pred_masks = model(model_input)

                loss = loss_fn(pred_masks, gt_mask[:, None, :, :]) # expand channel dim
                accelerator.backward(loss)
                batch_loss += loss.item()

            # After batch
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(batch_loss / batch_size)

        # After epoch
        scheduler.step()
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch: {epoch} \t Loss: {epoch_loss}", main_process_only=True)
        if accelerator.is_main_process:
            with open(f"{save_folder}/exp.log", 'a') as f:
                f.write(f"Epoch: {epoch} \t Loss: {epoch_loss} \n")

        # Saving
        if accelerator.is_main_process:
            if epoch >= config.EPOCH_TO_SAVE and epoch % config.SAVE_FREQUENCY == 0:
                accelerator.wait_for_everyone()
                model_state_dict = accelerator.get_state_dict(model)
                accelerator.save(model_state_dict, f"{save_folder}/ckpts/{epoch}.pt")

    end_time = time.time()
    logger.info(f"Training time: {(end_time - start_time) / 3600:.2f}", main_process_only=True)


if __name__ == '__main__':
    main()