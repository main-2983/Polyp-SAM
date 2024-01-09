import argparse
import importlib
import os
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
package = os.path.join(os.path.dirname(sys.path[0]), "src")
sys.path.append(os.path.dirname(package))
from src.models.SelfPromptPoint import BaseIterativePromptSAM
from src.plot_utils import show_box
from src.datasets import UnNormalize, uniform_sample_points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--ckpt', required=False, type=str)
    parser.add_argument('--round', type=int, default=6)
    parser.add_argument('--store_path', type=str, default=None, required=False)

    args = parser.parse_args()

    return args


@torch.no_grad()
def main():
    args = parse_args()

    module = importlib.import_module(args.config)
    config = module.Config()
    # Model
    model: BaseIterativePromptSAM = config.model
    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    device = "cuda"
    model = model.to(device)
    mask_threshold = model.mask_threshold

    os.makedirs(f"{args.store_path}", exist_ok=True)

    dataset = config.dataset
    for i in tqdm(range(len(dataset))):
        image_path = dataset.image_paths[i]
        name = os.path.basename(image_path)
        name = os.path.splitext(name)[0]
        sample = dataset[i]
        image = sample[0].to(device)  # (C, H, W)
        gt_mask = sample[1].to(device)  # (C, H, W)
        point_prompt = sample[2].to(device)  # (num_boxes, points_per_box, 2)
        point_label = sample[3].to(device)  # (num_boxes, points_per_box)
        image_size = (dataset.image_size, dataset.image_size)
        gt = gt_mask > 0.5  # Cuz resizing image sucks
        single_gt_mask = gt[0, ...].clone()  # (1024, 1024)
        for i in range(1, gt_mask.shape[0]):
            single_gt_mask = torch.logical_or(single_gt_mask, gt[i, ...])  # (1024, 1024)
        single_gt_mask = single_gt_mask.long() # (1024, 1024)
        mask_np = single_gt_mask.cpu().numpy()

        # Round 0 input
        model.forward_embedding(image)

        mask_input, false_positive_mask, false_negative_mask, rand = None, None, None, None
        point = (point_prompt, point_label)
        for round_i in range(args.round):
            model_input = {
                "image": image,
                "point_prompt": point,
                "box_prompt": None,
                "mask_input": mask_input,
                "image_size": image_size
            }
            low_res_masks, iou_predictions, point_pred, img_emb = model(model_input)

            gt_instance = dict(
                gt_mask=gt_mask,
                image_embedding=img_emb,
                point_prompt=point,
                box_prompt=None,
                logit_mask=mask_input,
                mask_to_sample=(false_positive_mask, false_negative_mask),
                rand=rand
            )
            if round_i > 0:
                point_target, flatten_point_pred = model.prepare_for_loss(
                    point_pred, gt_instance
                )
            # Select the mask with highest IoU for each object
            max_idx = torch.argmax(iou_predictions, dim=1)
            selected_masks = low_res_masks[0:1, max_idx[0]:max_idx[0] + 1, ...]  # (num_objects, 1, 256, 256)
            for i in range(1, low_res_masks.shape[0]):
                selected_masks = torch.concatenate([selected_masks,
                                                    low_res_masks[i:i + 1, max_idx[i]:max_idx[i] + 1, ...]],
                                                   dim=0)

            # Prepare next round target
            upscaled_masks = model.postprocess_masks(
                selected_masks, image.shape[-2:], image_size
            )

            # Step 1: convert mask to binary
            upscaled_masks = upscaled_masks > mask_threshold
            # Step 2: OR all mask to reduce to C=1
            single_upscale_mask = upscaled_masks[0, 0, ...].clone()  # (1024, 1024)
            for i in range(1, upscaled_masks.shape[0]):
                single_upscale_mask = torch.logical_or(single_upscale_mask,
                                                       upscaled_masks[i, 0, ...])  # (1024, 1024)
            single_upscale_mask = single_upscale_mask.long()
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
            if (np.random.rand() >= config.RATE):  # sample from false negative mask (select positive point)
                mask_to_sample = false_negative_mask
                rand = 1
            else:
                mask_to_sample = false_positive_mask # sample from false positive mask (select negative point)
                rand = -1
            # Step 4.3: RANDOMLY Sample point from mask
            width_point_prompt, height_point_prompt = uniform_sample_points(mask_to_sample,
                                                                            num_points=1)
            _point_prompt = torch.hstack([width_point_prompt, height_point_prompt])  # (1, 2)
            if _point_prompt.shape[0] <= 0:  # can't sample any points
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
                if _point_prompt.shape[0] <= 0:  # If still no points -> 100% correct mask
                    break  # Exit the Loop early
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

            # Plot
            if round_i > 0:
                save_path = f"{args.store_path}/{name}"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                fig, axis = plt.subplots(1, 5)
                axis[0].imshow(mask_np)
                axis[0].axis('off')
                point_target = point_target.view(64, 64, -1)
                if point_target.shape[-1] == 1: # Single label
                    pad_val = torch.zeros_like(point_target[:, :, 0])
                    point_target = torch.repeat_interleave(point_target, 2, dim=-1) # Expand to (64, 64, 2)
                    point_target[:, :, 1] = pad_val # Set the expanded axis to all 0s
                    point_pred = torch.repeat_interleave(point_pred, 2, dim=1) # Expand to (1, 2, 64, 64)
                    point_pred[0, 1, ...] = pad_val
                positive_target = point_target[..., 1].cpu().numpy() # (64, 64)
                negative_target = point_target[..., 0].cpu().numpy() # (64, 64)
                positive_pred = point_pred[0, 1, ...].cpu().numpy()
                negative_pred = point_pred[0, 0, ...].cpu().numpy()
                plots = (positive_target, negative_target, positive_pred, negative_pred)

                for i in range(1, 5):
                    axis[i].imshow(plots[i - 1])
                    axis[i].axis('off')
                plt.savefig(f"{save_path}/targets_iter_{round_i}.png")
                plt.close()
                fig, axis = plt.subplots(1, selected_masks.shape[0])
                if selected_masks.shape[0] > 1:
                    for i in range(selected_masks.shape[0]):
                        axis[i].imshow(selected_masks[i, 0, :, :].cpu().numpy())
                        axis[i].axis('off')
                else:
                    axis.imshow(selected_masks[0, 0, :, :].cpu().numpy())
                    axis.axis('off')
                plt.savefig(f"{save_path}/mask_iter_{round_i}.png")
                plt.close()


if __name__ == '__main__':
    main()