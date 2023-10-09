import argparse
import os
import time
import datetime
import pandas as pd
import torch
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import importlib  # for import module
import shutil  # for copy files
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)

from accelerate import Accelerator
from accelerate.logging import get_logger
logger = get_logger(__name__)

import sys
package = os.path.join(os.path.dirname(sys.path[0]), "src")
sys.path.append(os.path.dirname(package))
from src.datasets import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='data', help='data directory')

    args = parser.parse_args()

    return args

class MLoss(torch.nn.Module):
    def __init__(self):
        super(MLoss, self).__init__()

        self.dice_loss = DiceLoss(mode="binary", from_logits=True)

    def forward(self, pred_point, center_point, pred_mask, mask):
        distance_loss = self.euclidean_distance(pred_point, center_point)
        dice_loss = self.dice_loss(pred_mask, mask)
        total_loss = self.euclidean_distance(pred_point, center_point) + 100 * self.dice_loss(pred_mask, mask)
        return total_loss, distance_loss, dice_loss

    def euclidean_distance(self, pred_point, center_point):
        return torch.sqrt(torch.sum((pred_point - center_point) ** 2, dim=1)).mean()


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
    # loss_fn = config.LOSS_FN
    loss_fn = MLoss()

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
        dis_loss_epoch = []
        dice_loss_epch = []
        for batch in tqdm(train_loader, desc=f"epoch: {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                if config.EMBEDDING_PATHS:
                    image_embedding = batch['image_embedding']
                else:
                    image_embedding = None
                image = batch['image']
                mask = batch['mask']
                points = batch['points']
                
                _input = {
                    'image_embedding': image_embedding.squeeze(0),
                    'image': image,
                    'image_size': (config.IMAGE_SIZE, config.IMAGE_SIZE)
                }

                pred_mask, pred_point = model(_input)
                # pred = model(_input)

                # loss = loss_fn(pred, mask)
                loss, dis_loss, dice_loss = loss_fn(pred_point, points, pred_mask, mask)

                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                epoch_losses.append(loss.item())
                dis_loss_epoch.append(dis_loss.item())
                dice_loss_epch.append(dice_loss.item())
                

        # After epoch
        scheduler.step()
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        dis_loss_epoch = sum(dis_loss_epoch) / len(dis_loss_epoch)
        dice_loss_epch = sum(dice_loss_epch) / len(dice_loss_epch)
        logger.info(f"Epoch: {epoch} \t Loss: {epoch_loss} \t dis_loss: {dis_loss} \t dice_loss: {dice_loss}", main_process_only=True)
        if accelerator.is_main_process:
            with open(f"{save_folder}/exp.log", 'a') as f:
                f.write(f"Epoch: {epoch} \t Loss: {epoch_loss} \n")

        # Saving
        accelerator.wait_for_everyone()
        model_state_dict = accelerator.get_state_dict(model)
        print(f"Saved model to {save_folder}")
        accelerator.save(model_state_dict, f"{save_folder}/ckpts/{epoch}.pt")

    end_time = time.time()
    logger.info(f"Training time: {(end_time - start_time)/3600:.2f}", main_process_only=True)
    if accelerator.is_main_process:
        with open(f"{save_folder}/exp.log", 'a') as f:
            f.write(f"Training time: {(end_time - start_time)/3600:.2f}")


if __name__ == '__main__':
    main()