from typing import Union

import numpy as np
import torch


class DistanceBboxCoder:

    def encode(self,
               points: Union[torch.Tensor, np.ndarray],
               bbox: Union[torch.Tensor, np.ndarray]):
        left = points[..., 0] - bbox[..., 0]
        right = bbox[..., 2] - points[..., 0]
        top = points[..., 1] - bbox[..., 1]
        bottom = bbox[..., 3] - points[..., 1]
        if isinstance(points, torch.Tensor):
            return torch.stack([left, right, top, bottom], -1)
        elif isinstance(points, np.ndarray):
            return np.stack([left, right, top, bottom], -1)
        else:
            raise ValueError

    def decode(self,
               points: Union[torch.Tensor, np.ndarray],
               distances: Union[torch.Tensor, np.ndarray]):
        x1 = points[..., 0] - distances[..., 0]
        y1 = points[..., 1] - distances[..., 1]
        x2 = points[..., 0] + distances[..., 2]
        y2 = points[..., 1] + distances[..., 3]

        if isinstance(points, torch.Tensor):
            return torch.stack([x1, y1, x2, y2], -1)
        elif isinstance(points, np.ndarray):
            return np.stack([x1, y1, x2, y2], -1)
        else:
            raise ValueError


if __name__ == '__main__':
    coder = DistanceBboxCoder()
    # np single
    point = np.asarray([0, 0])
    bbox = np.asarray([-1, 1, 1, -3])
    print(coder.encode(point, bbox))
    # np multiple
    points = np.asarray([[0, 0], [0, -1]])
    print(coder.encode(points, bbox))
    bboxes = np.asarray([[-1, 1, 1, -3], [-1, 1, 1, -3]])
    print(coder.encode(points, bboxes))
