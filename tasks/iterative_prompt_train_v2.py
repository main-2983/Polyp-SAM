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
from src.datasets import create_dataloader, uniform_sample_points
from src.models.SelfPromptPoint.base_iterative_prompt import BaseIterativePromptSAM

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
    accelerator = Accelerator(
        gradient_accumulation_steps=config.ACCUMULATE_STEPS,
        kwargs_handlers=[ddp_kwargs]
    )
    is_distributed = accelerator.distributed_type != accelerator.distributed_type.NO

    # Model
    model: BaseIterativePromptSAM = config.model
    mask_threshold = model.mask_threshold

    # Dataloader
    train_dataset, train_loader = create_dataloader(
        config.dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # Loss
    self_prompt_loss = config.SELF_PROMPT_LOSS

    # Optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.OPTIMIZER(params, **config.OPTIMIZER_KWARGS)
    scheduler = config.SCHEDULER(optimizer, **config.SCHEDULER_KWARGS)

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    device = model.device

    # Training loop

    start_time = time.time()

    for epoch in range(1, config.MAX_EPOCHS + 1):
        epoch_losses = []

        # One epoch
        for batch in tqdm(train_loader, desc=f"epoch: {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                images = batch[0]  # (B, C, H, W)
                masks = batch[1]  # (B, C, H, W)
                point_prompts = batch[2]  # (B, num_boxes, points_per_box, 2)
                point_labels = batch[3]  # (B, num_boxes, points_per_box)
                box_prompts = batch[4]  # (B, num_boxes, 4)
                image_size = (train_dataset.image_size, train_dataset.image_size)
                batch_size = images.shape[0]  # Batch size

                if is_distributed:
                    input_images = torch.stack([model.module.preprocess(img) for img in images], dim=0)
                else:
                    input_images = torch.stack([model.preprocess(img) for img in images], dim=0)

                # Forward batch, process per-image
                batch_loss, sam_losses, prompt_losses = [], [], []
                for image, gt_mask, point_prompt, point_label, box_prompt in zip(
                        input_images, masks, point_prompts, point_labels, box_prompts
                ):
                    # Prepare round 0 inputs
                    round_loss = 0
                    mask_input, mask_to_sample, rand = None, None, None
                    point = (point_prompt, point_label)
                    for round in range(config.ROUND_PER_EPOCH):
                        model_input = {
                            "image": image,
                            "point_prompt": point if round < config.ROUND_PER_EPOCH - 1 else None,
                            "box_prompt": box_prompt if config.USE_BOX_PROMPT and round == 0 else None,
                            "mask_input": mask_input,
                            "image_size": image_size
                        }
                        # low_res_mask (num_objects, num_preds, 256, 256)
                        # iou_predictions (num_objects, num_preds)
                        # point_pred (1, 2, 64, 64)
                        # img_emb (1, 256, 64, 64)
                        low_res_masks, iou_predictions, point_pred, img_emb = model(model_input)

                        # Select the mask with highest IoU for each object
                        max_idx = torch.argmax(iou_predictions, dim=1)
                        selected_masks = low_res_masks[0:1, max_idx[0]:max_idx[0] + 1, ...]  # (num_objects, 1, 256, 256)
                        selected_ious = iou_predictions[0:1, max_idx[0]:max_idx[0] + 1] # (num_objects, 1)
                        for i in range(1, low_res_masks.shape[0]):
                            selected_masks = torch.concatenate([selected_masks,
                                                                low_res_masks[i:i + 1, max_idx[i]:max_idx[i] + 1, ...]],
                                                               dim=0)
                            selected_ious = torch.concatenate([selected_ious,
                                                               iou_predictions[i:i+1, max_idx[i]:max_idx[i]+1]])

                        # Calculate loss with the selected_masks
                        gt_instance = dict(
                            gt_mask=gt_mask,
                            image_embedding=img_emb,
                            point_prompt=point,
                            box_prompt=box_prompt,
                            logit_mask=mask_input,
                            mask_to_sample=mask_to_sample,
                            rand=rand
                        )
                        if is_distributed:
                            upscaled_masks = model.module.postprocess_masks(
                                selected_masks, image.shape[-2:], image_size
                            )
                            if round > 0:
                                point_target, flatten_point_pred = model.module.prepare_for_loss(
                                    point_pred, gt_instance
                                )
                        else:
                            upscaled_masks = model.postprocess_masks(
                                selected_masks, image.shape[-2:], image_size
                            )
                            if round > 0:
                                point_target, flatten_point_pred = model.prepare_for_loss(
                                    point_pred, gt_instance
                                )

                        if round > 0:
                            with accelerator.autocast():
                                prompt_loss = self_prompt_loss(flatten_point_pred, point_target)
                            accelerator.backward(prompt_loss)
                            round_loss += prompt_loss.item()

                        selected_masks = selected_masks.detach() # Detach from the computation grad of next round

                        # Find the error region mask between selected_masks and ground truth, then sample points
                        with torch.no_grad():
                            # Step 1: convert mask to binary
                            upscaled_masks = upscaled_masks > mask_threshold
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
                            if (np.random.rand() >= config.RATE):  # sample from false negative mask
                                mask_to_sample = false_negative_mask
                                rand = 1
                            else:
                                mask_to_sample = false_positive_mask
                                rand = -1
                            # Step 4.3: RANDOMLY Sample point from mask
                            width_point_prompt, height_point_prompt = uniform_sample_points(mask_to_sample,
                                                                                            num_points=1)
                            _point_prompt = torch.hstack([width_point_prompt, height_point_prompt])  # (1, 2)
                            if _point_prompt.shape[0] <= 0: # can't sample any points
                                # Resample with different mask
                                if rand == 1:
                                    mask_to_sample = false_positive_mask
                                    rand = -1
                                else:
                                    mask_to_sample = false_negative_mask
                                    rand = 1
                                width_point_prompt, height_point_prompt = uniform_sample_points(mask_to_sample,
                                                                                                num_points=1)
                                _point_prompt = torch.hstack([width_point_prompt, height_point_prompt])  # (1, 2)
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
        logger.info(f"Epoch: {epoch} \t Loss: {epoch_loss}",
                    main_process_only=True)
        if accelerator.is_main_process:
            with open(f"{save_folder}/exp.log", 'a') as f:
                f.write(f"Epoch: {epoch} \t Loss: {epoch_loss} \n")

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
