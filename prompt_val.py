import os
import argparse
from glob import glob
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

import torch

from segment_anything import sam_model_registry, SamPredictor

from src.dataset import PromptPolypDataset
from src.metrics import get_scores, weighted_score


def test_prompt(checkpoint,
                model_size,
                test_folder,
                use_box: bool = False,
                store: bool = False,
                store_path: str = None):
    sam = sam_model_registry[model_size](checkpoint)
    sam.pixel_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    sam.pixel_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    sam = sam.to("cuda")
    predictor = SamPredictor(sam)

    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    table = []
    headers = ['Dataset', 'Dice']
    all_dices = []
    metric_weights = [0.1253, 0.0777, 0.4762, 0.0752, 0.2456]

    if (store):
        os.makedirs(store_path, exist_ok=True)

    for dataset_name in dataset_names:
        data_path = f'{test_folder}/{dataset_name}'
        test_images = glob('{}/images/*'.format(data_path))
        test_images.sort()
        test_masks = glob('{}/masks/*'.format(data_path))
        test_masks.sort()

        test_dataset = PromptPolypDataset(
            test_images, test_masks, 1024, 1, use_box, True
        )

        gts = []
        prs = []
        for i in tqdm(range(len(test_dataset)), desc=dataset_name):
            image_path = test_dataset.image_paths[i]
            name = os.path.basename(image_path)
            name = os.path.splitext(name)[0]
            sample = test_dataset[i]
            images = sample[0]  # (B, C, H, W)
            masks = sample[1]  # (B, C, H, W)
            point_prompts = sample[2]  # (B, num_boxes, points_per_box, 2)
            point_labels = sample[3]  # (B, num_boxes, points_per_box)
            box_prompts = sample[4]  # (B, num_boxes, 4)
            image_size = (test_dataset.image_size, test_dataset.image_size)

            predictor.set_torch_image(images, image_size)

            pred_masks, scores, logits = predictor.predict_torch(
                point_coords=point_prompts[0],
                point_labels=point_labels[0],
                boxes=box_prompts if use_box else None,
                multimask_output=False
            )

            pred_masks = pred_masks[0].detach().cpu().numpy() # (num_masks, H, W)
            final_mask = pred_masks[0]
            for i in range(1, len(pred_masks)):
                final_mask = np.logical_or(final_mask, pred_masks[i])
            gt_mask = masks[0, 0].cpu().numpy()

            gts.append(gt_mask)
            prs.append(final_mask)

            if (store):
                plt.figure(figsize=(10, 10))
                plt.imshow(final_mask)
                plt.axis("off")
                plt.savefig(f"{store_path}/autoSAM/{dataset_name}/{name}.png")
                plt.close()

        mean_iou, mean_dice, _, _ = get_scores(gts, prs)
        all_dices.append(mean_dice)
        table.append([dataset_name, mean_iou, mean_dice])

    wdice = weighted_score(
        scores=all_dices,
        weights=metric_weights
    )
    table.append(['wDice', 0, wdice])
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    # Write result to file
    if store:
        with open(f"{store_path}/autoSAM/results.txt", 'w') as f:
            f.write(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str, help="Model checkpoint")
    parser.add_argument('--size', type=str, default="vit_b")
    parser.add_argument('--path', type=str, help="Path to test folder")
    parser.add_argument('--use-box', action="store_true")
    parser.add_argument('--store', action="store_true")
    parser.add_argument('--store_path', type=str, default=None, required=False)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    test_prompt(args.ckpt,
                args.size,
                args.path,
                args.use_box,
                args.store,
                args.store_path)