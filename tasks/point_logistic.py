import os
import argparse
import pickle
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tabulate import tabulate
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from sklearn.linear_model import LogisticRegression

from segment_anything.modeling import Sam
from segment_anything import build_sam_vit_b, SamPredictor

from src.plot_utils import show_points, show_mask
from src.metrics import get_scores, weighted_score


def rgb_loader(image_size, path) -> np.ndarray:
    with open(path, 'rb') as f:
        img = Image.open(f).resize((image_size, image_size), Image.BILINEAR)
        img = np.array(img.convert('RGB'))
        return img


def binary_loader(image_size, path) -> np.ndarray:
    with open(path, 'rb') as f:
        img = Image.open(f).resize((image_size, image_size), Image.NEAREST)
        img = np.array(img.convert('L'))
        return img


def get_max_dist_point(mask):
  # Compute the distance transform of the binary mask
  dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

  # Find the location of the point with maximum distance value
  max_dist = np.max(dist_transform)
  max_dist_idx = np.where(dist_transform == max_dist)
  point = (max_dist_idx[1][0], max_dist_idx[0][0])  # (x, y) coordinates

  return point


def train(args):
    sam: Sam = build_sam_vit_b(args.sam_ckpt)
    sam.to("cuda")
    sam.eval()
    predictor = SamPredictor(sam)

    image_paths = glob(args.image_path)
    mask_paths = glob(args.mask_path)

    assert len(image_paths) == len(mask_paths)

    embeddings, labels = [], []
    with torch.no_grad():
        for i in tqdm(range(len(image_paths))):
            image = rgb_loader(args.image_size, image_paths[i])
            mask = binary_loader(args.mask_size, mask_paths[i])
            predictor.set_image(image)
            embedding = predictor.get_image_embedding()
            embedding = embedding.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256)).reshape(-1, 256)
            embeddings.append(embedding)
            labels.append(mask.flatten())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    model = LogisticRegression(max_iter=1000)
    model.fit(embeddings, labels)

    pickle.dump(model, open("regression.pkl", 'rb'))

    return model


def val(args, model = None):
    sam: Sam = build_sam_vit_b(args.sam_ckpt)
    sam.to("cuda")
    sam.eval()
    predictor = SamPredictor(sam)

    # classifier
    if model is None:
        model = pickle.load(args.classifier)

    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    table = []
    headers = ['Dataset', 'IoU', 'Dice', 'Recall', 'Precision']
    all_ious, all_dices, all_precisions, all_recalls = [], [], [], []
    metric_weights = [0.1253, 0.0777, 0.4762, 0.0752, 0.2456]

    for dataset_name in dataset_names:
        os.makedirs(f"workdir/inference/LogisticRegression/{dataset_name}", exist_ok=True)

    for dataset_name in dataset_names:
        data_path = f'{args.test_folder}/{dataset_name}'
        test_images = glob('{}/images/*'.format(data_path))
        test_images.sort()
        test_masks = glob('{}/masks/*'.format(data_path))
        test_masks.sort()

        gts = []
        prs = []
        for i in tqdm(range(len(test_images))):
            image_path = test_images[i]
            mask_path = test_masks[i]
            name = os.path.basename(image_path)
            name = os.path.splitext(name)[0]
            image = rgb_loader(args.image_size, image_path)
            gt_mask = binary_loader(args.image_size, mask_path)

            predictor.set_image(image)
            embedding = predictor.get_image_embedding()
            embedding = embedding.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256)).reshape(-1, 256)

            # get the mask predicted by the linear classifier
            y_pred = model.predict(embedding)
            y_pred = y_pred.reshape((64, 64))
            # mask predicted by the linear classifier
            mask_pred_l = cv2.resize(y_pred, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            # use distance transform to find a point inside the mask
            fg_point = get_max_dist_point(mask_pred_l)
            # prompt the sam with the point
            input_point = np.array([[fg_point[0], fg_point[1]]])
            input_label = np.array([1])
            pred_masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=None,
                multimask_output=False,
            )
            final_mask = pred_masks[0]
            for i in range(1, len(pred_masks)):
                final_mask = np.logical_or(final_mask, pred_masks[i])

            gts.append(gt_mask)
            prs.append(final_mask)

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(final_mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.axis("off")
            plt.savefig(f"workdir/inference/LogisticRegression/{dataset_name}/{name}.png")
            plt.close()
            plt.imshow(mask_pred_l)
            plt.savefig(f"workdir/inference/LogisticRegression/{dataset_name}/{name}_predmask.png")
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
    with open(f"workdir/inference/LogisticRegression/Self-Prompt-Point/results_polyp.txt", 'w') as f:
            f.write(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('sam-ckpt', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--classifier', type=str, required=False)
    parser.add_argument('--test-folder', type=str, required=False)
    parser.add_argument('--image-path', type=str, required=False)
    parser.add_argument('--mask-path', type=str, required=False)
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--mask-size', type=int, default=64)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.mode == train:
        model = train(args)
        val(args, model)
    else:
        val(args)
