import os
import argparse
import importlib
from glob import glob
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

import torch

import sys
package = os.path.join(os.path.dirname(sys.path[0]), "src")
sys.path.append(os.path.dirname(package))
from src.datasets.polyp.polyp_dataset import PolypDataset
from src.datasets.utils import UnNormalize
from src.metrics import get_scores, weighted_score, dice_np
from src.plot_utils import show_points, show_mask
from src.models.SelfPromptPoint import AutoSAM


@torch.no_grad()
def test_prompt(config,
                test_folder,
                first_threshold: float = 0.4,
                pos_threshold: float = 0.3,
                neg_threshold: float = 0.3,
                iters: int = 5,
                store: bool = False,
                store_path: str = None):
    module = importlib.import_module(config)
    config = module.Config()

    predictor: AutoSAM = config.autoSAM
    predictor.inference_mode("cuda")
    device = predictor.device

    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    table = []
    headers = ['Dataset', 'IoU', 'Dice', 'Recall', 'Precision']
    all_ious, all_dices, all_precisions, all_recalls = [], [], [], []
    metric_weights = [0.1253, 0.0777, 0.4762, 0.0752, 0.2456]

    if store:
        for dataset_name in dataset_names:
            os.makedirs(f"{store_path}/{dataset_name}", exist_ok=True)

    for dataset_name in dataset_names:
        data_path = f'{test_folder}/{dataset_name}'
        test_images = glob('{}/images/*'.format(data_path))
        test_images.sort()
        test_masks = glob('{}/masks/*'.format(data_path))
        test_masks.sort()

        test_dataset = PolypDataset(
            test_images, test_masks, image_size=config.IMAGE_SIZE)

        gts = []
        prs = []
        for i in tqdm(range(len(test_dataset)), desc=dataset_name):
            image_path = test_dataset.image_paths[i]
            name = os.path.basename(image_path)
            name = os.path.splitext(name)[0]
            sample = test_dataset[i]
            images = sample["image"].to(device) # (C, H, W)
            gt_masks = sample["mask"].to(device) # (num_boxes, 1024, 1024)
            image_size = (test_dataset.image_size, test_dataset.image_size)

            predictor.set_torch_image(images[None], image_size)
            image_np = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(images)
            image_np = image_np.cpu().numpy().transpose(1, 2, 0)

            gt_mask = gt_masks[0].cpu().numpy()
            for i in range(1, len(gt_masks)):
                gt_mask = np.logical_or(gt_mask, gt_masks[i].cpu().numpy())

            # Prepare round 0 input
            mask_input = None
            final_mask = np.zeros_like(gt_mask)
            best_dice = 0
            for iter in range(iters):
                pred_masks, iou_predictions, low_res_masks, points, labels = predictor.predict_torch(
                    iter=iter,
                    first_threshold=first_threshold,
                    iterative_threshold=(pos_threshold, neg_threshold),
                    mask_input=mask_input,
                    multimask_output=False,
                )

                points = points.cpu().numpy() # (num_boxes, points_per_box, 2)
                labels = labels.cpu().numpy() # (num_boxes, points_per_box)
                pred_masks = pred_masks[:, 0].detach().cpu().numpy() # (num_masks, H, W)
                pred_mask = pred_masks[0] # (H, W)
                for i in range(1, len(pred_masks)):
                    pred_mask = np.logical_or(pred_mask, pred_masks[i])

                temp_dice = dice_np(pred_mask, gt_mask)
                if temp_dice < best_dice:
                    break

                # Prepare next round input
                final_mask = pred_mask
                mask_input = low_res_masks
                best_dice = temp_dice

                if (store):
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image_np)
                    show_mask(final_mask, plt.gca())
                    for (point, label) in zip(points, labels):
                        show_points(point, label, plt.gca())
                    plt.axis("off")
                    save_path = f"{store_path}/{dataset_name}/{name}"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    plt.savefig(f"{save_path}/iter_{iter}.png")
                    plt.close()

            gts.append(gt_mask)
            prs.append(final_mask)

        mean_iou, mean_dice, mean_precision, mean_recall = get_scores(gts, prs)
        all_ious.append(mean_iou)
        all_dices.append(mean_dice)
        all_recalls.append(mean_recall)
        all_precisions.append(mean_precision)
        table.append([dataset_name, mean_iou, mean_dice, mean_recall, mean_precision])
    if len(dataset_names) > 1:
        wiou = weighted_score(
            scores=all_ious,
            weights=metric_weights
        )
        wdice = weighted_score(
            scores=all_dices,
            weights=metric_weights
        )
        wrecall = weighted_score(
            scores=all_recalls,
            weights=metric_weights
        )
        wprecision = weighted_score(
            scores=all_precisions,
            weights=metric_weights
        )
        table.append(['Weighted', wiou, wdice, wrecall, wprecision])
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    # Write result to file
    if store:
        with open(f"{store_path}/results_polyp.txt", 'w') as f:
            f.write(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Model config")
    parser.add_argument('--path', type=str, help="Path to test folder")
    parser.add_argument('--first-threshold', type=float, default=0.4)
    parser.add_argument('--pos-threshold', type=float, default=0.3)
    parser.add_argument('--neg-threshold', type=float, default=0.3)
    parser.add_argument('--iters', type=int, default=5, help="Number of prediction iteration")
    parser.add_argument('--store', action="store_true")
    parser.add_argument('--store_path', type=str, default=None, required=False)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    test_prompt(args.config,
                args.path,
                args.first_threshold,
                args.pos_threshold,
                args.neg_threshold,
                args.iters,
                args.store,
                args.store_path)