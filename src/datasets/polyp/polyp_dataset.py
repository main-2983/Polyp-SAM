from typing import List, Dict, Any

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset

from ..base import PromptBaseDataset, ToTensor
from ..utils import sample_center_point


class PromptPolypDataset(PromptBaseDataset):
    def __init__(self, *args, task_number=0, **kwargs):
        super(PromptPolypDataset, self).__init__(*args, task_number=task_number, **kwargs)


class PolypDataset(Dataset):
    def __init__(
            self,
            image_paths: List[str],
            mask_paths: List[str],
            embedding_paths: List[str] = None,
            image_size: int = 1024
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.embedding_paths = embedding_paths

        self.image_size = image_size

        # Sort
        self.image_paths.sort()
        self.mask_paths.sort()
        if self.embedding_paths is not None:
            self.embedding_paths.sort()

    def __len__(self) -> int:
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

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image = self.rgb_loader(self.image_paths[index])
        mask = self.binary_loader(self.mask_paths[index])
        
        # Sample center point
        points = sample_center_point(mask, 1)

        if self.embedding_paths is not None:
            embedding = np.load(self.embedding_paths[index])

        # To Tensor
        image = ToTensor()(image)
        mask = ToTensor()(mask)
        points = torch.from_numpy(points).float()
        
        if self.embedding_paths is not None:
            embedding = torch.from_numpy(embedding).float()

        if self.embedding_paths is not None:
            return {
                'image_embedding': embedding,  # (1, 256, 64, 64)
                'image': image,  # (3, 1024, 1024)
                'mask': mask,  # (1, 1024, 1024)
                'points': points
            }
        else:
            return {
                'image': image,
                'mask': mask,
                'points': points
            }