import argparse
import importlib
import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import sys
package = os.path.join(os.path.dirname(sys.path[0]), "src")
sys.path.append(os.path.dirname(package))
from src.models.SelfPromptPoint.point_head import MlvlPointHead, MlvlPointGenerator
from src.plot_utils import show_box
from src.datasets import UnNormalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--store_path', type=str, default=None, required=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    module = importlib.import_module(args.config)
    config = module.Config()
    # Point Model
    point_gen: MlvlPointHead = config.point_gen
    point_gen.strides = [16, 32, 64]
    point_gen.prior_generator = MlvlPointGenerator(point_gen.strides)

    device = "cpu"

    os.makedirs(f"{args.store_path}", exist_ok=True)

    dataset = config.dataset
    for i in tqdm(range(len(dataset))):
        image_path = dataset.image_paths[i]
        name = os.path.basename(image_path)
        name = os.path.splitext(name)[0]
        sample = dataset[i]
        image = sample[0].to(device)
        gt_bbox = sample[4].to(device)

        num_priors_per_lvl = [64*64, 32*32, 16*16]
        featmap_size_per_lvl = [(64, 64), (32, 32), (16, 16)]
        mlvl_priors = point_gen.prior_generator.grid_priors(
            featmap_size_per_lvl,
            device=device
        )
        assigned_gt_inds, _ = point_gen._get_target_single(
            mlvl_priors, gt_bbox
        )
        targets = []
        for i in range(len(num_priors_per_lvl)):
            start = sum(num_priors_per_lvl[0:i])
            # start = num_priors_per_lvl[i - 1] if i != 0 else 0
            end = num_priors_per_lvl[i]
            featmap_size_lvli = featmap_size_per_lvl[i]
            target_lvli = assigned_gt_inds[start : (start + end)]
            target_lvli = target_lvli.view(featmap_size_lvli)
            targets.append(target_lvli.cpu().numpy())

        plt.figure()
        image = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        plt.imshow(image_np)
        show_box(gt_bbox[0], plt.gca())
        plt.axis('off')
        plt.savefig(f"{args.store_path}/{name}.png")
        plt.close()
        fig, axis = plt.subplots(1, len(targets))
        for i in range(len(targets)):
            axis[i].imshow(targets[i])
        plt.savefig(f"{args.store_path}/{name}_targets.png")
        plt.close()


if __name__ == '__main__':
    main()