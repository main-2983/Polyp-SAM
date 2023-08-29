import argparse
import os
import time
import datetime
import importlib
import shutil
from glob import glob
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)

from accelerate import Accelerator
from accelerate.logging import get_logger
logger = get_logger(__name__)
from accelerate.utils import DistributedDataParallelKwargs

import torch
from torch.utils.data import DataLoader

from src.dataset import CountPolypDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default="src.base_config", help="where to get config module")
    parser.add_argument('--val-path', type=str, help="Path to test folder")
    args = parser.parse_args()
    return args


@torch.no_grad()
def val(accelerator_obj: Accelerator,
        model,
        test_folder):

    model.eval()
    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    for dataset_name in dataset_names:
        data_path = f'{test_folder}/{dataset_name}'
        X_test = glob('{}/images/*'.format(data_path))
        X_test.sort()
        y_test = glob('{}/masks/*'.format(data_path))
        y_test.sort()

        val_dataset = CountPolypDataset(X_test, y_test)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
        val_loader = accelerator_obj.prepare(val_loader)

        correct = 0
        for batch in tqdm(val_loader, disable=not accelerator_obj.is_main_process):
            images = batch[0]
            targets = batch[1]

            input_images = torch.stack([model.module.preprocess(img) for img in images], dim=0)

            preds = model(input_images)
            preds = torch.round(preds)
            all_preds, all_targets = accelerator_obj.gather_for_metrics((preds, targets))
            correct += (all_preds == all_targets).float().sum()
        acc = 100 * correct / len(val_dataset)
        if accelerator_obj.is_main_process:
            logger.info(f"Acc on {dataset_name}: {acc}")



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
    train_dataset = config.dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )

    # Loss
    loss_fn = config.LOSS_FN

    # Optimizer
    optimizer = config.OPTIMIZER(model.parameters(), **config.OPTIMIZER_KWARGS)
    scheduler = config.SCHEDULER(optimizer, **config.SCHEDULER_KWARGS)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    start_time = time.time()

    # Training loop
    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []

        # One epoch
        for batch in tqdm(train_loader, desc=f"epoch: {epoch}", disable=not accelerator.is_main_process):
            images = batch[0]
            targets = batch[1]

            input_images = torch.stack([model.module.preprocess(img) for img in images], dim=0)

            preds = model(input_images)
            loss = loss_fn(preds, targets)
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()

            epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch: {epoch} \t Loss: {epoch_loss}", main_process_only=True)

        val(accelerator, model, args.val_path)

        # Saving
        if accelerator.is_main_process:
            if epoch >= config.EPOCH_TO_SAVE and epoch % config.SAVE_FREQUENCY == 0:
                accelerator.wait_for_everyone()
                model_state_dict = accelerator.get_state_dict(model)
                accelerator.save(model_state_dict, f"{save_folder}/ckpts/{epoch}.pt")

    end_time = time.time()
    logger.info(f"Training time: {(end_time - start_time) / 3600:.2f}", main_process_only=True)
