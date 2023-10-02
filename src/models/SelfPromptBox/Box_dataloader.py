from typing import Iterable

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import ToTensor

from .src.polyp.utils import sample_box, filter_box, uniform_sample_points, sample_center_point

class PromptBaseDataset(Dataset):
    def __init__(self,
                 image_paths: list,
                 mask_paths: list,
                 task_number: int = 0,
                 image_size: int = 1024,
                 num_points: int = 1,
                 use_box_prompt: bool = True,
                 use_center_points: bool = False,
                 transform = None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.task_number = task_number
        self.image_size = image_size
        self.num_points = num_points
        self.use_box_prompt = use_box_prompt
        self.use_center_points = use_center_points
        self.box_threshold = image_size // 100
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).resize((self.image_size, self.image_size), Image.BILINEAR)
            img = np.array(img.convert('RGB'))
            return img

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).resize((self.image_size, self.image_size), Image.NEAREST)
            img = np.array(img.convert('L'))
            return img

    def __getitem__(self, index):
        image = self.rgb_loader(self.image_paths[index])
        mask = self.binary_loader(self.mask_paths[index])

        median = np.unique(gt)[len(np.unique(gt))//2]
        gt = np.where(gt >= median, 255, gt)

        point_prompts = []
        point_labels = []

        # Extract Boxes
        boxes = sample_box(mask)
        boxes = filter_box(boxes, self.box_threshold)


        if self.transform is not None:
            transformed = self.transform(image=image,
                                         mask=mask,
                                         bboxes=boxes,
                                         label=['positive'] * boxes.shape[0])
            image = transformed["image"]
            mask = transformed["mask"]
            boxes = np.asarray(transformed["bboxes"], dtype=np.int32)
        
        box_labels = boxes
        # Extract Points and Masks wrt box
        num_box = boxes.shape[0]
        masks = []
        for i in range(num_box):
            # Get the box region
            box = boxes[i]
            # Extract the mask within the box region
            region = mask[box[1]: box[3], box[0]: box[2]]
            # Create the fake original mask with the extracted mask above
            _mask = np.zeros(mask.shape, dtype=np.uint8)
            _mask[box[1]: box[3], box[0]: box[2]] = region
            if not self.use_center_points:
                rand_height, rand_width = uniform_sample_points(_mask, num_points=self.num_points)
                point_prompt = np.hstack([rand_height, rand_width])
            else:
                point_prompt = sample_center_point(_mask, num_points=self.num_points)
            point_label = np.ones((self.num_points,))
            point_prompts.append(point_prompt)
            point_labels.append(point_label)
            masks.append(_mask)
        try:
            point_prompts = np.asarray(point_prompts)
            point_labels = np.asarray(point_labels)
            masks = np.asarray(masks).transpose((1, 2, 0))
        except ValueError:
            point_prompts = np.asarray(point_prompts)
            point_labels = np.asarray(point_labels)
            masks = np.asarray(masks).transpose((1, 2, 0))
        if self.use_box_prompt:
            box_prompts = boxes
        else:
            box_prompts = np.zeros(boxes.shape)

        # To Tensor
        image = ToTensor()(image)
        mask = ToTensor()(masks) # (B, num_box, H, W)
        task_prompts = torch.as_tensor([self.task_number], dtype=torch.int)
        point_prompts = torch.as_tensor(point_prompts, dtype=torch.float) # (num_box, points_per_box, 2)
        point_labels = torch.as_tensor(point_labels, dtype=torch.int) # (num_box, points_per_box)
        box_prompts = torch.as_tensor(box_prompts, dtype=torch.float) # (num_box, 4)
        box_labels = torch.as_tensor(box_labels, dtype=torch.float)
        return image, mask, point_prompts, point_labels, box_prompts, task_prompts, box_labels


class ConcatPromptDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[PromptBaseDataset]):
        super(ConcatPromptDataset, self).__init__(datasets)

        self.image_size = self.datasets[0].image_size

def collate_fn(batch):
    images, masks, point_prompts, point_labels, box_prompts, task_prompts, box_labels = zip(*batch)

    images = torch.stack(images, dim=0)

    # Process Box
    # Find max length
    max_num_box = 0
    for box_prompt in box_prompts:
        if box_prompt.shape[0] > max_num_box:
            max_num_box = box_prompt.shape[0]
    # Pad other with box [0, 0, 0, 0]
    new_box_prompts = []
    for box_prompt in box_prompts:
        num_to_pad = max_num_box - box_prompt.shape[0]
        pad_prompt = torch.tensor([[0, 0, 0, 0]], dtype=torch.float)
        for i in range(num_to_pad):
            box_prompt = torch.concatenate([box_prompt, pad_prompt], dim=0)
        new_box_prompts.append(box_prompt)
    box_prompts = torch.stack(new_box_prompts, dim=0)

    new_point_prompts = []
    # Process Points: Pad in negative point at (0, 0)
    for point_prompt in point_prompts:
        num_to_pad = max_num_box - point_prompt.shape[0]
        pad_prompt = torch.zeros((1, *point_prompt.shape[1:]), dtype=torch.float)
        for i in range(num_to_pad):
            point_prompt = torch.concatenate([point_prompt, pad_prompt], dim=0)
        new_point_prompts.append(point_prompt)
    point_prompts = torch.stack(new_point_prompts, dim=0)

    new_point_labels = []
    # Process Labels: Pad in negative label
    for point_label in point_labels:
        num_to_pad = max_num_box - point_label.shape[0]
        pad_prompt = torch.zeros((1, *point_label.shape[1:]), dtype=torch.int)
        for i in range(num_to_pad):
            point_label = torch.concatenate([point_label, pad_prompt], dim=0)
        new_point_labels.append(point_label)
    point_labels = torch.stack(new_point_labels, dim=0)

    # Process Masks
    new_masks = []
    for mask in masks:
        num_to_pad = max_num_box - mask.shape[0]
        pad_mask = torch.zeros((1, *mask.shape[1:]))
        for i in range(num_to_pad):
            mask = torch.concatenate([mask, pad_mask], dim=0)
        new_masks.append(mask)
    masks = torch.stack(new_masks, dim=0)

    # Process Task Prompt
    new_task_prompts = []
    for task_prompt in task_prompts:
        num_to_pad = max_num_box - task_prompt.shape[0]
        for i in range(num_to_pad):
            task_prompt = torch.concatenate([task_prompt, task_prompt[0:1]], dim=0)
        new_task_prompts.append(task_prompt)
    task_prompts = torch.stack(new_task_prompts, dim=0)

    new_box_labels = []
    label_class = []
    for box in box_labels:
        number_object = box.shape[0]
        label_class.append(torch.zeros(number_object,))
        center_x = ((box[:, 0] + box[:, 2])/2)/1024
        center_y = ((box[:, 1] + box[:, 3])/2)/1024
        W = (box[:, 2] - box[:, 0])/1024
        H = (box[:, 3] - box[:, 1])/1024
        new_box = torch.stack((center_x, center_y, W, H), dim = 1)
        new_box_labels.append(new_box)
    new_box_labels = torch.stack(new_box_labels, dim = 0)
    label_class = torch.stack(label_class, dim=0)
    return images, masks, point_prompts, point_labels, box_prompts, task_prompts, new_box_labels, label_class

