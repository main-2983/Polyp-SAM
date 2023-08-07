from typing import Optional
import os
import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from skimage.measure import label, regionprops, find_contours

from segment_anything.utils.transforms import ResizeLongestSide


class PromptPolypDataset(Dataset):
    def __init__(self,
                 image_paths: list,
                 mask_paths: list,
                 image_size: int = 1024,
                 num_points: int = 1):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.num_points = num_points

    def __len__(self):
        return len(self.image_paths)

    def uniform_sample_points(self, mask: np.ndarray, num_points:int = 1):
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

        return rand_heights, rand_widths # Y-coord, X-coord

    def sample_box(self, mask: np.ndarray):
        """
        TODO: Allow multiple Bboxes extraction
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        boxes = []

        mask = self._mask_to_border(mask)
        lbl = label(mask)
        props = regionprops(lbl)

        for prop in props:
            x1 = prop.bbox[1]
            y1 = prop.bbox[0]

            x2 = prop.bbox[3]
            y2 = prop.bbox[2]

            boxes.append(np.asarray([x1, y1, x2, y2]))

        boxes = np.asarray(boxes)
        
        return boxes.astype(np.int32)

    def _mask_to_border(self, mask: np.ndarray):
        h, w = mask.shape[:2]
        border = np.zeros((h, w))

        contours = find_contours(mask, 128)
        for contour in contours:
            for c in contour:
                x = int(c[0])
                y = int(c[1])
                border[x, y] = 255

        return border

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
        original_size = image.shape[:2]
        mask = self.binary_loader(self.mask_paths[index])

        # Extract Points
        rand_heights, rand_widths = self.uniform_sample_points(mask, num_points=self.num_points)
        point_prompts = np.hstack([rand_heights, rand_widths])

        # Extract Boxes
        box_prompts = self.sample_box(mask)

        return dict(
            image=image,
            mask=mask,
            point_prompts=point_prompts,
            box_prompts=box_prompts
        )
