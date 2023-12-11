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
from src.models.SelfPromptPoint import IterativeSelfPromptSAM
from src.plot_utils import show_box
from src.datasets import UnNormalize


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
    model: IterativeSelfPromptSAM = config.model
    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    device = "cpu"
    model = model.to(device)

    os.makedirs(f"{args.store_path}", exist_ok=True)

    dataset = config.dataset
    for i in tqdm(range(len(dataset))):
        image_path = dataset.image_paths[i]
        name = os.path.basename(image_path)
        name = os.path.splitext(name)[0]
        sample = dataset[i]
        image = sample[0].to(device)  # (C, H, W)
        mask = sample[1].to(device)  # (C, H, W)
        point_prompt = sample[2].to(device)  # (num_boxes, points_per_box, 2)
        point_label = sample[3].to(device)  # (num_boxes, points_per_box)
        image_size = (dataset.image_size, dataset.image_size)
        image_np = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        image_np = image_np.cpu().numpy().transpose(1, 2, 0)

        # Round 0 input
        mask_input = None
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
            point_target, flatten_point_pred = model.point_prompt_module.prepare_for_loss(
                point_pred, point, img_emb
            )
            # Select the mask with highest IoU for each object
            max_idx = torch.argmax(iou_predictions, dim=1)
            selected_masks = low_res_masks[0:1, max_idx[0]:max_idx[0] + 1, ...]  # (num_objects, 1, 256, 256)
            for i in range(1, low_res_masks.shape[0]):
                selected_masks = torch.concatenate([selected_masks,
                                                    low_res_masks[i:i + 1, max_idx[i]:max_idx[i] + 1, ...]],
                                                   dim=0)
            mask_input = selected_masks

            # Plot
            save_path = f"{args.store_path}/{name}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig, axis = plt.subplots(1, 5)
            axis[0].imshow(mask[0].cpu().numpy())
            point_target = point_target.view(64, 64, -1)
            positive_target = point_target[..., 1].cpu().numpy() # (64, 64)
            negative_target = point_target[..., 0].cpu().numpy() # (64, 64)
            positive_pred = point_pred[0, 1, ...].cpu().numpy()
            negative_pred = point_pred[0, 0, ...].cpu().numpy()
            plots = (positive_target, negative_target, positive_pred, negative_pred)

            for i in range(1, 5):
                axis[i].imshow(plots[i - 1])
            plt.axis('off')
            plt.savefig(f"{save_path}/targets_iter_{round_i}.png")
            plt.close()
            fig, axis = plt.subplots(1, selected_masks.shape[0])
            for i in range(selected_masks.shape[0]):
                axis[i].imshow(selected_masks[i, 0, :, :].cpu().numpy())
            plt.axis('off')
            plt.savefig(f"{save_path}/mask_iter_{round_i}.png")
            plt.close()


if __name__ == '__main__':
    main()