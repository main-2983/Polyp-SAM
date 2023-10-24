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


import sys
package = os.path.join(os.path.dirname(sys.path[0]), "src")
sys.path.append(os.path.dirname(package))
from src.datasets.polyp.polyp_dataset import PolypDataset
from src.metrics import get_scores, weighted_score


@torch.no_grad()
def test_prompt(checkpoint,
                config,
                test_folder,
                store: bool = False,
                store_path: str = None):
    module = importlib.import_module(config)
    config = module.Config()
    model: torch.nn.Module = config.model
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to("cuda")
    predictor = SamPredictor(model)
    device = model.device

    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    # dataset_names = ['Kvasir']
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
            gt_mask = sample["mask"].to(device)
            image_size = (test_dataset.image_size, test_dataset.image_size)

            predictor.set_torch_image(image[None], image_size)

            img_embedding = model.image_encoder(model.preprocess(image[None]))
            
            point_prompt = model.point_model(img_embedding)
            point_label = model.labels
            

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
            gt_mask = gt_mask[0].cpu().numpy()

            gts.append(gt_mask)
            prs.append(final_mask)

            if (store):
                plt.figure(figsize=(10, 10))
                plt.imshow(final_mask)
                plt.axis("off")
                plt.savefig(f"{store_path}/Self-Prompt-Point/{dataset_name}/{name}.png")
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
    parser.add_argument('--path', type=str, help="Path to test folder")
    parser.add_argument('--store', action="store_true")
    parser.add_argument('--store_path', type=str, default=None, required=False)
    args = parser.parse_args()

    return args

# python /mnt/nvme1n1/intern2023/nguyen.xuan.hoa-b/Polyp-SAM/tasks/point_prompt_val.py '/mnt/nvme1n1/intern2023/nguyen.xuan.hoa-b/Polyp-SAM/workdir/train/Self-Prompt-Point/2023-09-20_095355/ckpts/599.pt' configs.selfprompt-point --path '/mnt/nvme1n1/intern2023/nguyen.xuan.hoa-b/Polyp-SAM/data/TestDataset'

# python /mnt/nvme1n1/intern2023/nguyen.xuan.hoa-b/Polyp-SAM/tasks/point_prompt_val.py '/mnt/nvme1n1/intern2023/nguyen.xuan.hoa-b/Polyp-SAM/workdir/train/Self-Prompt-Point/2023-10-02_022633/ckpts/20.pt' configs.selfprompt-point --path '/mnt/nvme1n1/intern2023/nguyen.xuan.hoa-b/Polyp-SAM/data/TestDataset' --store

if __name__ == '__main__': 
    args = parse_args()
    test_prompt(args.ckpt,
                args.config,
                args.path,
                args.store,
                args.store_path)