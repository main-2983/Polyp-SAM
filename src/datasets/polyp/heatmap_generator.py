from typing import Any
import numpy as np
import torch
import cv2

class HeatmapGenerator(object):

    def __init__(self):
        pass

    def __call__(self, gt_kpts, input_size, maskmap=None):
        width, height = input_size
        stride = 1
        num_keypoints = 1
        sigma = 200
        method = "gaussian"
        heatmap = np.zeros((num_keypoints + 1, height // stride, width // stride), dtype=np.float32)
        start = stride / 2.0 - 0.5

        for i in range(len(gt_kpts)):
            for j in range(num_keypoints):

                x = gt_kpts[i][j][0]
                y = gt_kpts[i][j][1]
                y_range = [i for i in range(int(height // stride))]
                x_range = [i for i in range(int(width // stride))]
                xx, yy = np.meshgrid(x_range, y_range)
                xx = xx * stride + start
                yy = yy * stride + start
                d2 = (xx - x) ** 2 + (yy - y) ** 2
                if method == 'gaussian':
                    exponent = d2 / 2.0 / sigma / sigma
                elif method == 'laplace':
                    exponent = np.sqrt(d2) / 2.0 / sigma

                else:
                    print('Not support heatmap method.')
                    exit(1)

                mask = exponent <= 4.6052
                cofid_map = np.exp(-exponent)
                cofid_map = np.multiply(mask, cofid_map)
                heatmap[j:j+1, :, :] += cofid_map[np.newaxis, :, :]

                heatmap[j:j+1, :, :][heatmap[j:j+1, :, :] > 1.0] = 1.0

            heatmap[num_keypoints, :, :] = 1.0 - np.max(heatmap[:-1, :, :], axis=0)

        heatmap = torch.from_numpy(heatmap)
        if maskmap is not None:
            heatmap = heatmap * maskmap
        return heatmap
    
class HeatmapGeneratorVer2(object):
    def __init__(self):
        pass

    def gaussian(self, x, y, x0, y0, sigma):
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def __call__(self, mask: np.ndarray, with_mask: bool) -> np.ndarray:
        """
        Generate a heatmap from a binary mask.
        """
        # Convert to uint8
        mask = mask.astype(np.uint8)
        # Find the contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Area of each region
        contour_areas = [cv2.contourArea(
            cv2.convexHull(contour)) for contour in contours]
        # Create an empty heatmap
        heatmap = np.zeros_like(mask, dtype=np.float32)
        # Generate the heatmap
        for contour, contour_area in zip(contours, contour_areas):
            # Find the moments:
            M = cv2.moments(contour)
            # Ensure that the moment is not zero to avoid division by zero
            if M["m00"] != 0:
                # Find the centroid
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # heatmap += gaussian(*np.indices(mask.shape),
                #                     cY, cX, contour_area / 100)
                heatmap += self.gaussian(*np.indices(mask.shape),
                                    cY, cX, np.sqrt(contour_area) / 7)
        # Normalize the heatmap
        if with_mask:
            heatmap = heatmap * mask
        heatmap = heatmap / np.max(heatmap)
        return heatmap