import numpy as np
import torch


class HeatmapGenerator(object):

    def __init__(self):
        pass

    def __call__(self, gt_kpts, input_size, maskmap=None):
        width, height = input_size
        stride = 1
        num_keypoints = 1
        sigma = 30
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
