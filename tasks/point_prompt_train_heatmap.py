import argparse
import os
import datetime
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import importlib  # for import module
import shutil  # for copy files
from tqdm import tqdm
import logging
import torch
import numpy as np
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from accelerate import Accelerator
from accelerate.logging import get_logger
logger = get_logger(__name__)

import sys
package = os.path.join(os.path.dirname(sys.path[0]), "src")
sys.path.append(os.path.dirname(package))
from src.datasets import create_dataloader
from point_prompt_val_heatmap import test_prompt

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

    model = config.model

    loss_fn = config.LOSS_FN

    optimizer = config.OPTIMIZER(model.parameters(), **config.OPTIMIZER_KWARGS)
    scheduler = config.SCHEDULER(optimizer, **config.SCHEDULER_KWARGS)

    train_dataset, train_loader = create_dataloader(
        config.dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        collate_fn=None
    )

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    for epoch in range(1, config.MAX_EPOCHS + 1):

        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"epoch: {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                image = batch[0]
                mask = batch[1]
                heatmap_label = batch[-1]
                _input = {
                    'image': image,
                    'heatmap': heatmap_label,
                    'maskmap': mask
                }

                out = model(_input)

                loss_dict = loss_fn(out)

                loss = loss_dict["loss"]
                
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
        if epoch >= 40 and epoch % 5 == 0:

            accelerator.wait_for_everyone()
            model_state_dict = accelerator.get_state_dict(model)
            print(f"Saved model to {save_folder}")
            accelerator.save(model_state_dict, f"{save_folder}/ckpts/{epoch}.pt")       
            ckpts_path = f"{save_folder}/ckpts/{epoch}.pt"
            DatasetTest = "/home/trinh.quang.huy/sun-polyp/Dataset/TestDataset"
            test_prompt(ckpts_path, args.config, DatasetTest)

    # for i, batch in enumerate(tqdm(train_loader)):

    #     heatmap = batch[-1]
    #     mask = batch[1][0][0]
        # mask = np.array(mask.cpu())
        # heatmap = np.array(heatmap.cpu())
        # maps = heatmap + mask
        # plt.imshow(maps)
        # if not os.path.exists("fig"):
        #     os.makedirs("fig")
        # plt.savefig("fig/mask{0}.png".format(i))

if __name__ == "__main__":
    main()