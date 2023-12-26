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
from src.datasets.polyp.polyp_dataset import PromptPolypDataset
from src.datasets.utils import UnNormalize
from src.metrics import get_scores, weighted_score
from src.plot_utils import show_points, show_mask
from src.models.SelfPromptPoint import IterativeSelfPredictor, IterativeSelfPromptSAM


@torch.no_grad()
def test_prompt(checkpoint,
                config,
                test_folder,
                positive_threshold: float = 0.1,
                negative_threshold: float = 0.1,
                iters: int = 5,
                store: bool = False,
                store_path: str = None):
    module = importlib.import_module(config)
    config = module.Config()

    model: IterativeSelfPromptSAM = config.model
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to("cuda")
    predictor = IterativeSelfPredictor(config.sam, model=model)
    device = model.device

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

        test_dataset = PromptPolypDataset(
            test_images, test_masks, image_size=config.IMAGE_SIZE,
            num_points=1, use_box_prompt=False, use_center_points=True
        )

        gts = []
        prs = []
        for i in tqdm(range(len(test_dataset)), desc=dataset_name):
            image_path = test_dataset.image_paths[i]
            name = os.path.basename(image_path)
            name = os.path.splitext(name)[0]
            sample = test_dataset[i]
            images = sample[0].to(device) # (C, H, W)
            gt_masks = sample[1].to(device) # (num_boxes, 1024, 1024)
            point_prompts = sample[2].to(device)  # (num_boxes, points_per_box, 2)
            point_labels = sample[3].to(device)  # (num_boxes, points_per_box)
            image_size = (test_dataset.image_size, test_dataset.image_size)

            predictor.set_torch_image(images[None], image_size)
            image_np = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(images)
            image_np = image_np.cpu().numpy().transpose(1, 2, 0)

            # Prepare round 0 input
            mask_input = None
            final_mask = np.zeros_like(gt_masks.cpu().numpy())
            point = (point_prompts, point_labels)
            for iter in range(iters):
                pred_masks, iou_predictions, low_res_masks, points, labels = predictor.predict_torch(
                    threshold=(positive_threshold, negative_threshold),
                    point_prompt=point if iter == 0 else None,
                    mask_input=mask_input,
                    multimask_output=False,
                )

                points = points.cpu().numpy() # (num_boxes, points_per_box, 2)
                labels = labels.cpu().numpy() # (num_boxes, points_per_box)
                pred_masks = pred_masks[0].detach().cpu().numpy() # (num_masks, H, W)
                final_mask = pred_masks[0] # (H, W)
                for i in range(1, len(pred_masks)):
                    final_mask = np.logical_or(final_mask, pred_masks[i])

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

                # Prepare next round input
                mask_input = low_res_masks

            gt_mask = gt_masks[0].cpu().numpy()
            for i in range(1, len(gt_masks)):
                gt_mask = np.logical_or(gt_mask, gt_masks[i])

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
    parser.add_argument('ckpt', type=str, help="Model checkpoint")
    parser.add_argument('config', type=str, help="Model config")
    parser.add_argument('--path', type=str, help="Path to test folder")
    parser.add_argument('--positive', type=float, default=0.1)
    parser.add_argument('--negative', type=float, default=0.1)
    parser.add_argument('--iters', type=int, default=5, help="Number of prediction iteration")
    parser.add_argument('--store', action="store_true")
    parser.add_argument('--store_path', type=str, default=None, required=False)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    test_prompt(args.ckpt,
                args.config,
                args.path,
                args.positive,
                args.negative,
                args.iters,
                args.store,
                args.store_path)