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
                 use_box_prompt: bool = True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.num_points = num_points
        self.use_box_prompt = use_box_prompt

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

    def _filter_box(self, boxes: np.ndarray):
        """
        Filter out box that does not meet the standard
        """
        new_boxes = []
        for box in boxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            if (w * h) > 20: # smaller than 20 pixel square
                new_boxes.append(box)
        return np.asarray(new_boxes)

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

        if self.use_box_prompt:
            # Extract Boxes
            box_prompts = self.sample_box(mask)
            box_prompts = self._filter_box(box_prompts)

            # Extract Points within the box
            num_box = box_prompts.shape[0]
            for i in range(num_box):
                # Get the box region
                box = box_prompts[i]
                # Extract the mask within the box region
                region = mask[box[1] : box[3], box[0] : box[2]]
                # Create the fake original mask with the extracted mask above
                _mask = np.zeros(mask.shape, dtype=np.uint8)
                _mask[box[1] : box[3], box[0] : box[2]] = region
                rand_height, rand_width = self.uniform_sample_points(_mask, num_points=self.num_points)
                point_prompt = np.hstack([rand_height, rand_width])
                point_label = np.ones((self.num_points, ))
                point_prompts.append(point_prompt)
                point_labels.append(point_label)
            point_prompts = np.asarray(point_prompts)
            point_labels = np.asarray(point_labels)
        else:
            # Extract Points
            rand_height, rand_width = self.uniform_sample_points(mask, num_points=self.num_points)
            point_prompt = np.hstack([rand_height, rand_width])
            point_label = np.ones((self.num_points,))
            point_prompts.append(point_prompt)
            point_labels.append(point_label)
            point_prompts = np.asarray(point_prompts)
            point_labels = np.asarray(point_labels)
            box_prompts = np.asarray([0] * self.num_points)

        # To Tensor
        image = ToTensor()(image)
        mask = ToTensor()(mask) # binary mask
        point_prompts = torch.as_tensor(point_prompts, dtype=torch.float)
        point_prompts = point_prompts.view(-1, 2)
        point_labels = torch.as_tensor(point_labels, dtype=torch.int)
        point_labels = point_labels.view(-1)
        box_prompts = torch.as_tensor(box_prompts, dtype=torch.float)

        return image, mask, point_prompts, point_labels, box_prompts


def collate_fn(batch):
    images, masks, point_prompts, point_labels, box_prompts = zip(*batch)

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

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
        pad_prompt = torch.tensor([[0, 0]], dtype=torch.float)
        for i in range(num_to_pad):
            point_prompt = torch.concatenate([point_prompt, pad_prompt], dim=0)
        new_point_prompts.append(point_prompt)
    point_prompts = torch.stack(new_point_prompts, dim=0)

    new_point_labels = []
    # Process Labels: Pad in negative point at (0, 0)
    for point_label in point_labels:
        num_to_pad = max_num_box - point_label.shape[0]
        pad_prompt = torch.tensor([0], dtype=torch.int)
        for i in range(num_to_pad):
            point_label = torch.concatenate([point_label, pad_prompt], dim=0)
        new_point_labels.append(point_label)
    point_labels = torch.stack(new_point_labels)

    return images, masks, point_prompts, point_labels, box_prompts


def create_dataloader(image_paths: list,
                      mask_paths: list,
                      use_box_prompt: bool = True,
                      image_size: int = 1024,
                      num_points: int = 1,
                      batch_size: int = 16,
                      num_workers: int = 4,
                      shuffle: bool = True):
    dataset = PromptPolypDataset(image_paths, mask_paths, image_size, num_points, use_box_prompt)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn)

    return dataset, dataloader


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