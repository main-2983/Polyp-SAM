from typing import Union
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from skimage.measure import label, regionprops, find_contours


class PromptPolypDataset(Dataset):
    def __init__(self,
                 image_paths: list,
                 mask_paths: list,
                 image_size: int = 1024,
                 num_points: int = 1,
                 use_box_prompt: bool = True,
                 use_center_points: bool = False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
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
        point_prompts = torch.as_tensor(point_prompts, dtype=torch.float) # (num_box, points_per_box, 2)
        point_labels = torch.as_tensor(point_labels, dtype=torch.int) # (num_box, points_per_box)
        box_prompts = torch.as_tensor(box_prompts, dtype=torch.float) # (num_box, 4)

        return image, mask, point_prompts, point_labels, box_prompts


def collate_fn(batch):
    images, masks, point_prompts, point_labels, box_prompts = zip(*batch)

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

    return images, masks, point_prompts, point_labels, box_prompts


def create_dataloader(image_paths: list,
                      mask_paths: list,
                      use_box_prompt: bool = True,
                      use_center_points: bool = False,
                      image_size: int = 1024,
                      num_points: int = 1,
                      batch_size: int = 16,
                      num_workers: int = 4,
                      shuffle: bool = True):
    dataset = PromptPolypDataset(image_paths, mask_paths, image_size, num_points, use_box_prompt, use_center_points)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn)

    return dataset, dataloader


def sample_box(mask: Union[np.ndarray, torch.Tensor]):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    boxes = []

    mask = _mask_to_border(mask)
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


def _mask_to_border(mask: np.ndarray):
    h, w = mask.shape[:2]
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x, y] = 255

    return border


def filter_box(boxes: np.ndarray):
    """
    Filter out box that does not meet the standard
    """
    new_boxes = []
    for box in boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        if (w * h) > 200:  # smaller than 200 pixel square
            new_boxes.append(box)
    return np.asarray(new_boxes)


def uniform_sample_points(mask: Union[torch.Tensor, np.ndarray], num_points: int = 1):
    """
    mask (torch Tensor or numpy array): ground truth mask to sample points prompt from
    num_points (int): number of points to sample
    """
    def _uniform_sample_points_numpy(mask: np.ndarray, num_points: int = 1):
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
        if width_non0.shape[0] > 0:  # if we have point in the mask
            for i in range(num_points):
                index = np.random.choice(width_non0.shape[0], 1)
                rand_width, rand_height = width_non0[index], height_non0[index]
                rand_widths.append(rand_width)
                rand_heights.append(rand_height)
            rand_widths, rand_heights = np.array(rand_widths), np.array(rand_heights)

        return rand_heights, rand_widths  # Y-coord, X-coord

    def _uniform_sample_points_torch(mask: torch.Tensor, num_points: int = 1):
        """
        mask (torch.Tensor): ground truth mask to sample points prompt from
        num_points (int): number of points to sample
        """
        # If the mask is not yet normalized
        norm_mask = mask
        if mask.max() > 1:
            norm_mask = mask / 255.0

        # Extract points of the mask
        height_non0, width_non0 = torch.where(norm_mask == 1)

        rand_heights, rand_widths = [], []
        if width_non0.shape[0] > 0:  # if we have points in the mask
            for i in range(num_points):
                index = torch.randint(width_non0.shape[0], (1,))
                rand_height, rand_width = height_non0[index], width_non0[index]
                rand_heights.append(rand_height)
                rand_widths.append(rand_width)
            rand_heights, rand_widths = torch.stack(rand_heights), torch.stack(rand_widths)
        else:
            rand_heights, rand_widths = torch.tensor(rand_heights), torch.tensor(rand_widths)

        return rand_heights, rand_widths  # Y-coord, X-coord

    if isinstance(mask, torch.Tensor):
        rand_heights, rand_widths = _uniform_sample_points_torch(mask, num_points)
    else:
        rand_heights, rand_widths = _uniform_sample_points_numpy(mask, num_points)
    return rand_heights, rand_widths


def sample_center_point(mask: np.ndarray, num_points: int = 1):
    # If the mask is not yet normalized
    norm_mask = mask
    if (max(mask.flatten()) > 1):
        norm_mask = mask / 255

    flat_mask = norm_mask.flatten()
    split = np.unique(np.sort(flat_mask), return_index=True)[1]
    points = []
    for idx in np.split(flat_mask.argsort(), split)[2:][:num_points]:
        points.append(np.array(np.unravel_index(idx, mask.shape)).mean(axis=1))
    points = np.asarray(points, dtype=np.int_) # (Width, Height)
    # Turn to (Height, Width)
    points = np.asarray([point[::-1] for point in points])
    return points


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
