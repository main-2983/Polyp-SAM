import os
import argparse
import importlib
from glob import glob
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from segment_anything import sam_model_registry, SamPredictor


import sys
package = os.path.join(os.path.dirname(sys.path[0]), "src")
sys.path.append(os.path.dirname(package))
from src.datasets.polyp.dataloader_heatmap import PromptDatasetHeatmap
from src.metrics import get_scores, weighted_score
from src.datasets import UnNormalize
from src.plot_utils import show_points, show_mask

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
    emty_point = dict()
    for dataset_name in dataset_names:
        data_path = f'{test_folder}/{dataset_name}'
        test_images = glob('{}/images/*'.format(data_path))
        test_images.sort()
        test_masks = glob('{}/masks/*'.format(data_path))
        test_masks.sort()

        test_dataset = PromptDatasetHeatmap(
            test_images, test_masks, image_size=config.IMAGE_SIZE, mask_size=config.MASK_SIZE
        )

        gts = []
        prs = []
        count_data = 0
        for idx, i in enumerate(tqdm(range(len(test_dataset)), desc=dataset_name)):
            image_path = test_dataset.image_paths[i]
            name = os.path.basename(image_path)
            name = os.path.splitext(name)[0]
            sample = test_dataset[i]

            image = sample[0].cuda()
            masks = sample[1].cuda()
            img_embedding = model.image_encoder(model.preprocess(image[None]))

            # ViT_feature_map = torch.sum(img_embedding, dim=1, keepdim=True)

            heatmap = np.array(model.heat_model.forward_infer(img_embedding)[0][-1][0][0].cpu())
            feature_map = np.array(model.heat_model.forward_infer(img_embedding)[-1][0][0].cpu())

            heatmap = cv2.resize(heatmap, (1024, 1024))
            mapSmooth = cv2.GaussianBlur(heatmap,(3,3),0,0)
            
            mapSmooth = np.uint8(mapSmooth > 0) * 255

            contours, _ = cv2.findContours(mapSmooth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
            keypoints = list()
            # # if contours == ():
            # #     continue
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    keypoints.append((cX, cY))
            if len(keypoints) != 0:
                point_label = np.ones((len(keypoints), ))
                point_prompt = torch.as_tensor(keypoints, dtype=torch.float, device=torch.device('cuda'))
                point_label = torch.as_tensor(point_label, dtype=torch.int, device=torch.device('cuda'))
                point_prompt, point_label = point_prompt[None, :, :], point_label[None, :]
            else:
                count_data += 1
                point_prompt = None
                point_label = None

            predictor.set_torch_image(image[None], config.IMAGE_SIZE)
            pred_masks, scores, logits = predictor.predict_torch(
                point_coords=point_prompt,
                point_labels=point_label,
                boxes=None,
                multimask_output=False
            )

            # feature map have shape 1, 128, 64, 64
            # visualize feature map with 128 channel in 1 figure
            # for i in range(128):
            #     plt.subplot(8, 16, i+1)
            #     plt.imshow(feature_map[0][i])
            #     plt.axis("off")
            # if point_prompt is not None:
            #     if not os.path.exists(f"{store_path}/feature_map/Self-Prompt-Point/{dataset_name}"):
            #         os.makedirs(f"{store_path}/feature_map/Self-Prompt-Point/{dataset_name}")
            #     plt.savefig(f"{store_path}/feature_map/Self-Prompt-Point/{dataset_name}/{name}.png")
            # else:
            #     if not os.path.exists(f"{store_path}/feature_map/Self-Prompt-Point-empty/{dataset_name}"):
            #         os.makedirs(f"{store_path}/feature_map/Self-Prompt-Point-empty/{dataset_name}")
            #     plt.savefig(f"{store_path}/feature_map/Self-Prompt-Point-empty/{dataset_name}/{name}.png")
            # plt.close()
            pred_masks = pred_masks[:, 0].detach().cpu().numpy()
            final_mask = pred_masks[0]
            for i in range(1, len(pred_masks)):
                final_mask = np.logical_or(final_mask, pred_masks[i])
            gt_mask = masks[0].cpu().numpy()

            gts.append(gt_mask)
            prs.append(final_mask)

            plt.subplot(2, 2, 1)
            image = UnNormalize(model.pixel_mean, model.pixel_std)(image)
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            plt.imshow(image_np)
            plt.imshow(final_mask, alpha=0.2)
            plt.scatter([x[0] for x in keypoints], [x[1] for x in keypoints], color='red', marker='o', s=50 , edgecolor='white', linewidth=1.1)
            plt.subplot(2, 2, 2)
            plt.imshow(gt_mask)
            plt.scatter([x[0] for x in keypoints], [x[1] for x in keypoints], color='red', marker='o', s=50 , edgecolor='white', linewidth=1.1)
            plt.subplot(2, 2, 3)
            plt.imshow(image_np)
            plt.imshow(heatmap)
            plt.scatter([x[0] for x in keypoints], [x[1] for x in keypoints], color='red', marker='o', s=50 , edgecolor='white', linewidth=1.1)
            plt.subplot(2, 2, 4)
            plt.imshow(feature_map)
            if point_prompt is not None:
                # print("feature map have point: ", "max: ", np.max(feature_map), "min: ", np.min(feature_map))
                plt.savefig(f"{store_path}/Self-Prompt-Point/{dataset_name}/{name}.png")
            else:
                # print("=====> feature map no have point: ", "max: ", np.max(feature_map), "min: ", np.min(feature_map))
                plt.savefig(f"{store_path}/Self-Prompt-Point/{dataset_name}/{name}.png")
                if not os.path.exists(f"{store_path}/Self-Prompt-Point-empty/{dataset_name}"):
                    os.makedirs(f"{store_path}/Self-Prompt-Point-empty/{dataset_name}")
                plt.savefig(f"{store_path}/Self-Prompt-Point-empty/{dataset_name}/{name}.png")
            plt.gca().clear()
            plt.close()
 

            # if (store):
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(final_mask)
            #     plt.axis("off")
            #     plt.savefig(f"{store_path}/Self-Prompt-Point/{dataset_name}/{name}.png")
            #     plt.close()

        mean_iou, mean_dice, mean_precision, mean_recall = get_scores(gts, prs)
        all_ious.append(mean_iou)
        all_dices.append(mean_dice)
        all_recalls.append(mean_recall)
        all_precisions.append(mean_precision)
        table.append([dataset_name, mean_iou, mean_dice, mean_recall, mean_precision])
    
        emty_point[dataset_name] = count_data
    emty_point['Total'] = sum(emty_point.values())
    print("Emty point: ", emty_point)

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