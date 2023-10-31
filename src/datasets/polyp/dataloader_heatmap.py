from typing import Iterable
from glob import glob
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
import torch
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from src.datasets.polyp.heatmap_generator import HeatmapGenerator, HeatmapGeneratorVer2
from src.datasets.utils import sample_box, filter_box, uniform_sample_points, sample_center_point


class PromptDatasetHeatmap(Dataset):
    def __init__(self,
                 image_paths: list,
                 mask_paths: list,
                 task_number: int = 0,
                 image_size: int = 1024,
                 mask_size: int = 64,
                 num_points: int = 1,
                 use_box_prompt: bool = True,
                 use_center_points: bool = True,
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
        self.mask_size = mask_size
        self.heatmap_gen = HeatmapGenerator()
        self.heatmap_gen_ver2 = HeatmapGeneratorVer2()

        # Sort
        self.image_paths.sort()
        self.mask_paths.sort()

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
            img1 = np.array(img.convert('L'))
            img = Image.open(f).resize((self.mask_size, self.mask_size), Image.NEAREST)
            img2 = np.array(img.convert('L'))
            return img1, img2

    def __getitem__(self, index):
        image = self.rgb_loader(self.image_paths[index])
        mask, mask2 = self.binary_loader(self.mask_paths[index])
        # mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        mask2 = np.where(mask2 > 127, 255, 0)
        

        heat_map = self.heatmap_gen_ver2(mask2, with_mask=False)

        # To Tensor
        image = ToTensor()(image)
        mask = ToTensor()(mask)
        heat_map = ToTensor()(heat_map)
        
        return image, mask, heat_map
