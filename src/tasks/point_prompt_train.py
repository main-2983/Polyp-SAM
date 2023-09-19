import argparse
import os
import time
import datetime
import pandas as pd
import importlib  # for import module
import shutil  # for copy files
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)

from accelerate import Accelerator
from accelerate.logging import get_logger
logger = get_logger(__name__)

from src.datasets import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='data', help='data directory')

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
    accelerator = Accelerator(
        gradient_accumulation_steps=config.ACCUMULATE_STEPS
    )

    # Model
    model = config.model

    # Dataloader
    train_dataset, train_loader = create_dataloader(
        config.dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        collate_fn=None
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

    for epoch in range(1, config.MAX_EPOCHS + 1):
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"epoch: {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                if config.EMBEDDING_PATHS:
                    image_embedding = batch['image_embedding']
                else:
                    image_embedding = None
                image = batch['image']
                mask = batch['mask']

                _input = {
                    'image_embedding': image_embedding,
                    'image': image,
                    'image_size': (config.IMAGE_SIZE, config.IMAGE_SIZE)
                }

                pred = model(_input)

                loss = loss_fn(pred, mask)

                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                epoch_losses.append(loss.item())

        # After epoch
        scheduler.step()
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch: {epoch} \t Loss: {epoch_loss}", main_process_only=True)
        if accelerator.is_main_process:
            with open(f"{save_folder}/exp.log", 'a') as f:
                f.write(f"Epoch: {epoch} \t Loss: {epoch_loss} \n")

        # Saving
        accelerator.wait_for_everyone()
        model_state_dict = accelerator.get_state_dict(model)
        accelerator.save(model_state_dict, f"{save_folder}/ckpts/{epoch}.pt")

    end_time = time.time()
    logger.info(f"Training time: {(end_time - start_time)/3600:.2f}", main_process_only=True)
    if accelerator.is_main_process:
        with open(f"{save_folder}/exp.log", 'a') as f:
            f.write(f"Training time: {(end_time - start_time)/3600:.2f}")


if __name__ == '__main__':
    main()