import os
import argparse
import numpy as np
from glob import glob

import torch
from PIL import Image
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from src.metrics import weighted_score, get_scores


@torch.no_grad()
def auto_test(checkpoint,
              model_size,
              test_folder,
              store: bool = False,
              store_path: str = None):
    sam = sam_model_registry[model_size](checkpoint)
    sam = sam.to("cuda")
    autoSAM = SamAutomaticMaskGenerator(sam)

    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    table = []
    headers = ['Dataset', 'Dice', 'Recall', 'Precision']
    all_dices, all_precisions, all_recalls = [], [], []
    metric_weights = [0.1253, 0.0777, 0.4762, 0.0752, 0.2456]

    if store:
        for dataset_name in dataset_names:
            os.makedirs(f"{store_path}/autoSAM/{dataset_name}", exist_ok=True)

    for dataset_name in dataset_names:
        data_path = f'{test_folder}/{dataset_name}'
        X_test = glob('{}/images/*'.format(data_path))
        X_test.sort()
        y_test = glob('{}/masks/*'.format(data_path))
        y_test.sort()

        gts = []
        prs = []
        for image_path, mask_path in tqdm(zip(X_test, y_test)):
            name = os.path.basename(image_path)
            name = os.path.splitext(name)[0]
            image = Image.open(image_path)
            mask = Image.open(mask_path)

            image = np.asarray(image)
            mask = np.asarray(mask)

            anns = autoSAM.generate(image)
            shape = mask.shape[:2] if mask.ndim == 3 else mask.shape

            if (len(anns) > 0):
                final_mask = anns[0]["segmentation"]
                for i in range(1, len(anns)):
                    final_mask = np.logical_or(final_mask, anns[i]["segmentation"])
            else:
                final_mask = np.zeros(shape)
            gts.append(mask[:, :, 0] if mask.ndim == 3 else mask)
            prs.append(final_mask)
            if (store):
                plt.figure(figsize=(10, 10))
                plt.imshow(final_mask)
                plt.axis("off")
                plt.savefig(f"{store_path}/autoSAM/{dataset_name}/{name}.png")
                plt.close()

        _, mean_dice, mean_precision, mean_recall = get_scores(gts, prs)
        all_dices.append(mean_dice)
        all_recalls.append(mean_recall)
        all_precisions.append(mean_precision)
        table.append([dataset_name, mean_dice, mean_recall, mean_precision])

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
    table.append(['Weighted', wdice, wrecall, wprecision])
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
    parser.add_argument('--store', action="store_true")
    parser.add_argument('--store_path', type=str, default=None, required=False)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    auto_test(args.ckpt,
              args.size,
              args.path,
              args.store,
              args.store_path)
