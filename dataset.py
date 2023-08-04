from typing import Optional
import os
import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


class PromptPolypDataset(Dataset):
    def __init__(self,
                 image_paths: list,
                 mask_paths: list,
                 transform: Optional,
                 image_size: int = 1024):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def _uniform_sample_points(self, mask: np.ndarray, num_points:int = 1):
        """
        mask (np.ndarray): ground truth mask to sample points prompt from
        num_points (int): number of points to sample
        """
        # If the mask is not yet normalized
        norm_mask = mask
        if (max(mask.flatten()) > 1):
            norm_mask = mask / 255
        # Extract points of the mask
        width_non0, height_non0 = np.where(norm_mask == 1)
        # Randomly take a point
        rand_widths, rand_heights = [], []
        for i in range(num_points):
            index = np.random.choice(width_non0.shape[0], 1)
            rand_width, rand_height = width_non0[index], height_non0[index]
            rand_widths.append(rand_width)
            rand_heights.append(rand_height)
        rand_widths, rand_heights = np.array(rand_widths), np.array(rand_heights)

        return rand_heights, rand_widths # (Y-axis, X-axis)

    def _sample_bbox(self, mask: np.ndarray, num_bboxes:int = 1):
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        boxes = np.zeros([num_bboxes, 4], dtype=np.int32)

        for i in range(num_bboxes):
            y_indices = np.where(np.any(mask, axis=0))[0]



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

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask / 255

