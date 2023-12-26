import os
import argparse
import importlib
from glob import glob
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

import torch

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam

import sys
package = os.path.join(os.path.dirname(sys.path[0]), "src")
sys.path.append(os.path.dirname(package))
from src.datasets.polyp.polyp_dataset import PolypDataset
from src.datasets import UnNormalize
from src.metrics import get_scores, weighted_score
from src.plot_utils import show_points, show_mask


@torch.no_grad()
def test_prompt(checkpoint,
                config,
                test_folder,
                threshold: float = 0.1,
                store: bool = False,
                store_path: str = None):
    module = importlib.import_module(config)
    config = module.Config()
    # Point Model
    point_gen = config.point_gen
    trained_state_dict = torch.load(checkpoint, map_location="cpu")
    state_dict = point_gen.state_dict()
    # Remove string in keys
    for k, v in trained_state_dict.items():
        key = k.split("point_model.")[1]
        state_dict[key] = v
    point_gen.load_state_dict(state_dict)
    point_gen.eval()
    point_gen = point_gen.to("cuda")

    # SAM
    sam = config.sam
    sam.pixel_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    sam.pixel_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    sam.to("cuda")
    predictor = SamPredictor(sam)
    device = sam.device

    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    table = []
    headers = ['Dataset', 'IoU', 'Dice', 'Recall', 'Precision']
    all_ious, all_dices, all_precisions, all_recalls = [], [], [], []
    metric_weights = [0.1253, 0.0777, 0.4762, 0.0752, 0.2456]

    if store:
        for dataset_name in dataset_names:
            os.makedirs(f"{store_path}/Self-Prompt-Point/{dataset_name}", exist_ok=True)

    for dataset_name in dataset_names:
        data_path = f'{test_folder}/{dataset_name}'
        test_images = glob('{}/images/*'.format(data_path))
        test_images.sort()
        test_masks = glob('{}/masks/*'.format(data_path))
        test_masks.sort()

        test_dataset = PolypDataset(
            test_images, test_masks, image_size=config.IMAGE_SIZE
        )

        gts = []
        prs = []
        for i in tqdm(range(len(test_dataset)), desc=dataset_name):
            image_path = test_dataset.image_paths[i]
            name = os.path.basename(image_path)
            name = os.path.splitext(name)[0]
            sample = test_dataset[i]
            image = sample["image"].to(device)
            gt_masks = sample["mask"].to(device)
            image_size = (test_dataset.image_size, test_dataset.image_size)

            predictor.set_torch_image(image[None], image_size)

            point_pred = point_gen(predictor.features)
            point_prompt, point_label = point_gen.decode_prediction(point_pred, threshold=threshold) # (1, num_points, 2), (1, num_points)

            pred_masks, scores, logits = predictor.predict_torch(
                point_coords=point_prompt,
                point_labels=point_label,
                boxes=None,
                multimask_output=False
            )

            pred_masks = pred_masks[0].detach().cpu().numpy() # (num_masks, H, W)
            final_mask = pred_masks[0]
            for i in range(1, len(pred_masks)):
                final_mask = np.logical_or(final_mask, pred_masks[i])
            gt_mask = gt_masks[0].cpu().numpy()
            for i in range(1, len(gt_masks)):
                gt_mask = np.logical_or(gt_mask, gt_masks[i].cpu().numpy())

            gts.append(gt_mask)
            prs.append(final_mask)

            point_np, label_np = point_prompt[0].cpu().numpy(), point_label[0].cpu().numpy()

            if (store):
                plt.figure(figsize=(10, 10))
                image = UnNormalize(sam.pixel_mean, sam.pixel_std)(image)
                image_np = image.cpu().numpy().transpose(1, 2, 0)
                plt.imshow(image_np)
                show_mask(final_mask, plt.gca())
                for p, l in zip(point_np, label_np):
                    show_points(p, l, plt.gca())
                plt.axis("off")
                plt.savefig(f"{store_path}/Self-Prompt-Point/{dataset_name}/{name}.png")
                plt.close()
                fig, axis = plt.subplots(1, len(point_pred))
                for i in range(len(point_pred)):
                    axis[i].imshow(point_pred[i][0, 0].cpu().numpy())
                plt.savefig(f"{store_path}/Self-Prompt-Point/{dataset_name}/{name}_featmap.png")
                plt.close()

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
        with open(f"{store_path}/Self-Prompt-Point/results_polyp.txt", 'w') as f:
            f.write(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str, help="Model checkpoint")
    parser.add_argument('config', type=str, help="Model config")
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--path', type=str, help="Path to test folder")
    parser.add_argument('--store', action="store_true")
    parser.add_argument('--store_path', type=str, default=None, required=False)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    test_prompt(args.ckpt,
                args.config,
                args.path,
                args.threshold,
                args.store,
                args.store_path)