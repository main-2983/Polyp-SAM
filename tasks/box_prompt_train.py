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
from src.datasets import uniform_sample_points
from src.datasets.polyp.Box_dataloader import create_dataloader
from src.metrics import iou_torch
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

    # Loss
    loss_fn = config.LOSS_FN
    iou_loss = config.IOU_LOSS

    # Optimizer
    # optimizer = config.OPTIMIZER(list(model.image_encoder.parameters()) + list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()), **config.OPTIMIZER_KWARGS)
    # scheduler = config.SCHEDULER(optimizer, **config.SCHEDULER_KWARGS)

    optimizer_detection = config.OPTIMIZER(list(model.box_decoder.parameters()), **config.OPTIMIZER_KWARGS_DETECTION)
    scheduler_detection = config.SCHEDULER_DETECTION(optimizer_detection, 50)

    criterion = build_criterion()

    # model, optimizer, train_loader, optimizer_detection, criterion = accelerator.prepare(
    #     model, optimizer, train_loader, optimizer_detection, criterion
    # )
    model, train_loader, optimizer_detection, criterion = accelerator.prepare(
        model, train_loader, optimizer_detection, criterion
    )
    device = model.device
    # load weight head
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
        epoch_losses = []
        epoch_dect_losses = []
        # One epoch
        i=0
        for iter,batch in enumerate(tqdm(train_loader, desc=f"epoch: {epoch}", disable=not accelerator.is_main_process)):
            with accelerator.accumulate(model):
                images = batch[0]  # (B, C, H, W)
                # masks = batch[1]  # (B, C, H, W)
                # point_prompts = batch[2]  # (B, num_boxes, points_per_box, 2)
                # point_labels = batch[3]  # (B, num_boxes, points_per_box)
                # box_prompts = batch[4]  # (B, num_boxes, 4)
                target_detection = batch[6]
                # image_size = (train_dataset.image_size, train_dataset.image_size)
                # batch_size = images.shape[0]  # Batch size

                if is_distributed:
                    input_images = torch.stack([model.module.preprocess(img) for img in images], dim=0)
                else:
                    input_images = torch.stack([model.preprocess(img) for img in images], dim=0)

                # # Forward batch, process per-image
                # batch_loss = []
                # for image, gt_mask, point_prompt, point_label, box_prompt in zip(
                #         input_images, masks, point_prompts, point_labels, box_prompts
                # ):
                #     # Prepare round 0 inputs
                #     round_loss = 0
                #     mask_input = None
                #     point = (point_prompt, point_label)
                #     for round in range(config.ROUND_PER_EPOCH):
                #         model_input = {
                #             "image": image,
                #             "point_prompt": point if round < config.ROUND_PER_EPOCH - 1 else None,
                #             "box_prompt": box_prompt if config.USE_BOX_PROMPT and round == 0 else None,
                #             "mask_input": mask_input,
                #             "image_size": image_size
                #         }
                #         # low_res_mask (num_objects, num_preds, 256, 256)
                #         # iou_predictions (num_objects, num_preds)
                #         low_res_masks, iou_predictions = model.forward_mask(model_input)
                #         # print('low res:',low_res_masks.shape)
                #         # Select the mask with highest IoU for each object
                #         max_idx = torch.argmax(iou_predictions, dim=1)
                #         selected_masks = low_res_masks[0:1, max_idx[0]:max_idx[0] + 1, ...]  # (num_objects, 1, 256, 256)
                #         selected_ious = iou_predictions[0:1, max_idx[0]:max_idx[0] + 1] # (num_objects, 1)
                #         for i in range(1, low_res_masks.shape[0]):
                #             selected_masks = torch.concatenate([selected_masks,
                #                                                 low_res_masks[i:i + 1, max_idx[i]:max_idx[i] + 1, ...]],
                #                                                dim=0)
                #             selected_ious = torch.concatenate([selected_ious,
                #                                                iou_predictions[i:i+1, max_idx[i]:max_idx[i]+1]])

                #         # Calculate loss with the selected_masks
                #         if is_distributed:
                #             upscaled_masks = model.module.postprocess_masks(
                #                 selected_masks, image.shape[-2:], image_size
                #             )
                #         else:
                #             upscaled_masks = model.postprocess_masks(
                #                 selected_masks, image.shape[-2:], image_size
                #             )
                #         # Calculate ious with the selected_masks
                #         gt_ious = []
                #         for i in range(upscaled_masks.shape[0]):
                #             # transform logit mask to binary mask
                #             m_pred = upscaled_masks[i].clone().detach() > mask_threshold # (1, 256, 256)
                #             gt_ious.append(iou_torch(gt_mask[i] > 0.5, m_pred[0]))
                #         gt_ious = torch.stack(gt_ious, dim=0)
                #         gt_ious = torch.unsqueeze(gt_ious, dim=1) # (num_objects, 1)

                #         with accelerator.autocast():
                #             loss = loss_fn(upscaled_masks, gt_mask[:, None, :, :])  # expand channel dim
                #             loss += iou_loss(selected_ious, gt_ious)
                #         accelerator.backward(loss)
                #         round_loss += loss.item()

                #         selected_masks = selected_masks.detach() # Detach from the computation grad of next round

                #         # Find the error region mask between selected_masks and ground truth, then sample points
                #         with torch.no_grad():
                #             # Step 1: convert mask to binary
                #             upscaled_masks = upscaled_masks > mask_threshold
                #             gt = gt_mask > 0.5  # Cuz resizing image sucks
                #             # Step 2: OR all mask to reduce to C=1
                #             single_upscale_mask = upscaled_masks[0, 0, ...].clone()  # (1024, 1024)
                #             single_gt_mask = gt[0, ...].clone()  # (1024, 1024)
                #             for i in range(1, upscaled_masks.shape[0]):
                #                 single_upscale_mask = torch.logical_or(single_upscale_mask,
                #                                                        upscaled_masks[i, 0, ...])  # (1024, 1024)
                #                 single_gt_mask = torch.logical_or(single_gt_mask, gt[i, ...])  # (1024, 1024)
                #             single_upscale_mask = single_upscale_mask.long()
                #             single_gt_mask = single_gt_mask.long()
                #             # Step 2: Find the error region
                #             # Error_mask will have value of:
                #             # -1: On false positive pixel (predict mask but wrong)
                #             # 0: On true positive and true negative pixel
                #             # 1: On false negative pixel (predict none but has mask)
                #             error_mask = single_gt_mask - single_upscale_mask  # (1024, 1024)
                #             # Step 4: sample points
                #             # Step 4.1: Separate the error mask into 2 part: The false positive and the false negative ones
                #             false_positive_mask = torch.where(error_mask == -1, error_mask, 0)
                #             false_positive_mask = -false_positive_mask
                #             false_negative_mask = torch.where(error_mask == 1, error_mask, 0)
                #             # Step 4.2: Choose a mask to sample from
                #             if (np.random.rand() >= config.RATE):  # sample from false negative mask
                #                 mask_to_sample = false_negative_mask
                #                 rand = 1
                #             else:
                #                 mask_to_sample = false_positive_mask
                #                 rand = -1
                #             # Step 4.3: RANDOMLY Sample point from mask
                #             height_point_prompt, width_point_prompt = uniform_sample_points(mask_to_sample,
                #                                                                             num_points=1)
                #             _point_prompt = torch.hstack([height_point_prompt, width_point_prompt])  # (1, 2)
                #             if _point_prompt.shape[0] <= 0: # can't sample any points
                #                 # Resample with different mask
                #                 if rand == 1:
                #                     mask_to_sample = false_positive_mask
                #                     rand = -1
                #                 else:
                #                     mask_to_sample = false_negative_mask
                #                     rand = 1
                #                 height_point_prompt, width_point_prompt = uniform_sample_points(mask_to_sample,
                #                                                                                 num_points=1)
                #                 _point_prompt = torch.hstack([height_point_prompt, width_point_prompt])  # (1, 2)
                #                 if _point_prompt.shape[0] <= 0: # If still no points -> 100% correct mask
                #                     break # Exit the Loop early
                #             _point_prompt = _point_prompt.unsqueeze(0)  # (1, 1, 2)
                #             if rand == 1:  # If sampled from false negative, insert label 1
                #                 _point_label = torch.ones((1,))
                #             else:
                #                 _point_label = torch.zeros((1,))
                #             _point_label = _point_label.unsqueeze(0)  # (1, 1)

                #             new_point_prompts = torch.as_tensor(_point_prompt, device=device, dtype=torch.float)
                #             new_point_labels = torch.as_tensor(_point_label, device=device, dtype=torch.int)

                #             # Step 5: Update the input for next round
                #             point = (new_point_prompts, new_point_labels)
                #             mask_input = selected_masks
                #     # End of all round
                #     batch_loss.append(round_loss)
                
                # # After batch
                # optimizer.step()
                # optimizer.zero_grad()
                # batch_loss = sum(batch_loss) / batch_size
                # epoch_losses.append(batch_loss)

                target_detection=[{k: v.to(device) for k, v in t.items()} for t in target_detection]
                outputs = model.forward_box(input_images)
                loss_dict = criterion(outputs, target_detection)
                if(iter%100==0):
                   print("CE loss: ",loss_dict["loss_ce"].cpu().item())
                   print("Box loss: ",loss_dict['loss_bbox'].cpu().item())
                   print("GIOU loss: ",loss_dict['loss_giou'].cpu().item())
                   print("Number of error class samples: ",loss_dict["class_error"].cpu().item())
                   print("OJ query: ",torch.sum(torch.argmax(outputs['pred_logits'],dim=-1)).cpu().item())
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
                accelerator.backward(losses)
                optimizer_detection.step()
                optimizer_detection.zero_grad()
                
                epoch_dect_losses.append(losses)

        # After epoch
        # scheduler.step()
        scheduler_detection.step()
        epoch_dect_losses = sum(epoch_dect_losses) / len(epoch_dect_losses)
        # epoch_loss = sum(epoch_losses) / len(epoch_losses)
        # logger.info(f"Epoch: {epoch} \t Loss_mask: {epoch_loss} \t Loss_detection: {epoch_dect_losses}", main_process_only=True)
        # if accelerator.is_main_process:
        #     with open(f"{save_folder}/exp.log", 'a') as f:
        #         f.write(f"Epoch: {epoch} \t Loss: {epoch_loss} \t {epoch_dect_losses}\n")

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