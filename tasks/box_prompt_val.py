import os
import argparse
import importlib
from glob import glob
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor

import sys
package = os.path.join(os.path.dirname(sys.path[0]), "src")
sys.path.append(os.path.dirname(package))
from src.models.SelfPromptBox.nms import non_max_suppression
# from src.datasets.polyp.polyp_dataset import PolypDataset
from src.metrics import get_scores, weighted_score
from src.models.SelfPromptBox.box_prompt_SAM import PostProcess
from src.datasets.polyp.Box_dataloader import PromptBaseDataset



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

        # test_dataset = PolypDataset(
        #     test_images, test_masks, image_size=config.IMAGE_SIZE
        # )
        test_dataset = PromptBaseDataset(
            test_images, test_masks, image_size=config.IMAGE_SIZE
        )
        gts = []
        prs = []
        for i in tqdm(range(len(test_dataset)), desc=dataset_name):
            image_path = test_dataset.image_paths[i]
            name = os.path.basename(image_path)
            name = os.path.splitext(name)[0]
            sample = test_dataset[i]
            image, mask, point_prompts, point_labels, box_prompts, task_prompts, box_labels=sample
            image = image.to(device)
            gt_mask = mask.to(device)
            image_size = (test_dataset.image_size, test_dataset.image_size)

            predictor.set_torch_image(image[None], image_size)

            outputs=model.forward_box(image[None])
            postprocessors=PostProcess()
            orig_target_sizes= torch.stack([torch.tensor([1024,1024]).to(image.device)])
            results = postprocessors(outputs, orig_target_sizes)
            # point_prompt = model.point_model(model.preprocess(image[None]))
            # point_label = model.labels
            thresh_hold=0.5
            valid_mask=results[0]['scores']>thresh_hold
            valid_scores=results[0]['scores'][valid_mask]
            valid_boxes=results[0]['boxes'][valid_mask]

            outputs = non_max_suppression(valid_boxes, valid_scores)
            valid_boxes = valid_boxes[outputs]
            valid_scores = valid_scores[outputs]
            valid_mask = valid_mask[outputs]
            if valid_boxes.shape[0] == 0:
                valid_boxes = torch.from_numpy(np.array([[0,0,1023,1023]])).cuda()
                valid_scores = torch.ones((1,)).cuda()
            num_pred_boxes=valid_boxes.shape[0]
            img=Image.open(image_path,mode='r')
            img=img.resize((1024,1024),Image.BILINEAR)
            plt.gca().invert_yaxis()
            fig, ax = plt.subplots() 
            ax.imshow(img) # plot image

            for i in range(num_pred_boxes):
                score=valid_scores[i].cpu().item()
                x1,y1,x2,y2=valid_boxes[i].cpu().numpy()
                rect = Rectangle((x1,y1),(x2-x1), (y2-y1), linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            for i in range(box_prompts.size(0)):
                x1,y1,x2,y2=box_prompts[i].cpu().numpy()
                rect = Rectangle((x1,y1),(x2-x1), (y2-y1), linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
            # plt.savefig('save_figs2/'+name+"_box.png",)
            if not os.path.isdir('save_figs2/'+dataset_name):
                os.mkdir('save_figs2/'+dataset_name)
            plt.savefig('save_figs2/'+dataset_name+'/'+name+"_box.png",)
            pred_masks, scores, logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes= None if valid_boxes.shape[0] == 0 else valid_boxes,
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


if __name__ == '__main__':
    args = parse_args()
    test_prompt(args.ckpt,
                args.config,
                args.path,
                args.store,
                args.store_path)