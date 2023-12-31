import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from ..utils import sample_box, filter_box, uniform_sample_points, sample_center_point


class PromptNucleiDataset(Dataset):
    def __init__(self,
                 image_paths: list,
                 mask_paths: list,
                 task_number: int = 1,
                 image_size: int = 1024,
                 num_points: int = 1,
                 use_box_prompt: bool = True,
                 use_center_points: bool = False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.task_number = task_number
        self.image_size = image_size
        self.num_points = num_points
        self.use_box_prompt = use_box_prompt
        self.use_center_points = use_center_points

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
        # TODO: Support Augmentations

        image = self.rgb_loader(self.image_paths[index])
        mask = self.binary_loader(self.mask_paths[index])

        point_prompts = []
        point_labels = []

        # Extract Boxes
        boxes = sample_box(mask)
        boxes = filter_box(boxes)

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
        task_prompt = torch.as_tensor([self.task_number] * num_box, dtype=torch.int) # (num_box, )
        point_prompts = torch.as_tensor(point_prompts, dtype=torch.float) # (num_box, points_per_box, 2)
        point_labels = torch.as_tensor(point_labels, dtype=torch.int) # (num_box, points_per_box)
        box_prompts = torch.as_tensor(box_prompts, dtype=torch.float) # (num_box, 4)

        return image, mask, point_prompts, point_labels, box_prompts, task_prompt