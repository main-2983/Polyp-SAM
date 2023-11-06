import argparse
import os
import time
import datetime
import importlib
import shutil
from tqdm import tqdm
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from accelerate import Accelerator
from accelerate.logging import get_logger
logger = get_logger(__name__)
from accelerate.utils import DistributedDataParallelKwargs

import torch
import sys
package = os.path.join(os.path.dirname(sys.path[0]), "src")
sys.path.append(os.path.dirname(package))
from src.datasets.polyp.Box_dataloader import create_dataloader
from src.models.SelfPromptBox.criterion import build_criterion

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default="src.base_config", help="where to get config module")
    args = parser.parse_args()
    return args
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
    accelerator = Accelerator(
        gradient_accumulation_steps=config.ACCUMULATE_STEPS,
        kwargs_handlers=[ddp_kwargs]
    )
    is_distributed = accelerator.distributed_type != accelerator.distributed_type.NO

    # Model
    model = config.model
    state_dict=torch.load("ckpts/polyp_box.pt")
    model.load_state_dict(state_dict,strict=False)

    # freeze image encoder and mask decoder
    for param in model.image_encoder.parameters():
        param.requires_grad=False
    for param in model.mask_decoder.parameters():
        param.requires_grad=False
    # Dataloader
    train_dataset, train_loader = create_dataloader(
        config.dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    val_dataset, val_loader = create_dataloader(
        config.val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    # Loss
    optimizer_detection = config.OPTIMIZER(list(model.box_decoder.parameters()), **config.OPTIMIZER_KWARGS_DETECTION)
    scheduler_detection = config.SCHEDULER_DETECTION(optimizer_detection, 50)

    criterion = build_criterion()

    # model, optimizer, train_loader, optimizer_detection, criterion = accelerator.prepare(
    #     model, optimizer, train_loader, optimizer_detection, criterion
    # )
    model, train_loader,val_loader, optimizer_detection, criterion = accelerator.prepare(
        model, train_loader,val_loader, optimizer_detection, criterion
    )
    device = model.device

    # load checkpoint to detector head
    checkpoint = torch.load("/home/dang.hong.thanh/Polyp-SAM/ckpts/detr-r50-dc5-f0fb7ef5.pth")
    model_state_dict=checkpoint['model']
    head_dict=dict()
    for key in model_state_dict.keys():
        if('decoder' in key):
            new_key=key.replace('transformer.decoder','box_decoder')
            head_dict[new_key]=model_state_dict[key]
    model.box_decoder.load_state_dict(head_dict,strict=False)


    # Training loop
    start_time = time.time()

    for epoch in range(1, config.MAX_EPOCHS + 1):
        epoch_dect_losses = []
        # One epoch
        train_loss = AverageMeter()
        val_loss = AverageMeter()

        # Training
        for iter,batch in enumerate(tqdm(train_loader, desc=f"epoch: {epoch}", disable=not accelerator.is_main_process)):
            with accelerator.accumulate(model):
                images = batch[0]  # (B, C, H, W)
                target_detection = batch[6]

                if is_distributed:
                    input_images = torch.stack([model.module.preprocess(img) for img in images], dim=0)
                else:
                    input_images = torch.stack([model.preprocess(img) for img in images], dim=0)


                target_detection=[{k: v.to(device) for k, v in t.items()} for t in target_detection]
                outputs = model.forward_box(input_images)
                loss_dict = criterion(outputs, target_detection)
                # if(iter%100==0):
                #    print("CE loss: ",loss_dict["loss_ce"].cpu().item())
                #    print("Box loss: ",loss_dict['loss_bbox'].cpu().item())
                #    print("GIOU loss: ",loss_dict['loss_giou'].cpu().item())
                #    print("Number of error class samples: ",loss_dict["class_error"].cpu().item())
                #    print("OJ query: ",torch.sum(torch.argmax(outputs['pred_logits'],dim=-1)).cpu().item())
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                train_loss.update(losses.item(), images.shape[0])   

                accelerator.backward(losses)
                optimizer_detection.step()
                optimizer_detection.zero_grad()
                
                epoch_dect_losses.append(losses)

        # Evaluating
        for iter,batch in enumerate(tqdm(val_loader, desc=f"epoch: {epoch}", disable=not accelerator.is_main_process)):
            with accelerator.accumulate(model):
                images = batch[0]  # (B, C, H, W)
                target_detection = batch[6]
                if is_distributed:
                    input_images = torch.stack([model.module.preprocess(img) for img in images], dim=0)
                else:
                    input_images = torch.stack([model.preprocess(img) for img in images], dim=0)
                target_detection=[{k: v.to(device) for k, v in t.items()} for t in target_detection]
                outputs = model.forward_box(input_images)
                loss_dict = criterion(outputs, target_detection)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                val_loss.update(losses.item(), images.shape[0])   
        print(
            f'Epoch : {epoch+1} - loss : {train_loss.avg:.4f} - val_loss : {val_loss.avg:.4f}\n'
        )
        # After epoch
        # scheduler.step()
        scheduler_detection.step()
        epoch_dect_losses = sum(epoch_dect_losses) / len(epoch_dect_losses)

        # Saving
        if epoch >= config.EPOCH_TO_SAVE and epoch % config.SAVE_FREQUENCY == 0:
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count