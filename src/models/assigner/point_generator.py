from typing import Tuple, Union, Sequence
from torch import Tensor

import torch

DeviceType = Union[str, torch.device]


class PointGenerator:
    def __init__(self,
                 offset: float = 0.5):
        self.offset = offset

    def _meshgrid(self,
                  x: Tensor,
                  y: Tensor,
                  row_major: bool = True) -> Tuple[Tensor, Tensor]:
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self,
                    featmap_size: Tuple[int, int],
                    stride=16,
                    device: DeviceType = 'cuda',
                    with_stride: bool = False) -> Tensor:
        """Generate grid points of a single level.

        Args:
            featmap_size (tuple[int, int]): Size of the feature maps.
            stride (int): The stride of corresponding feature map.
            device (str | torch.device): The device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: grid point in a feature map.
        """
        feat_h, feat_w = featmap_size
        shift_x = (torch.arange(0., feat_w, device=device) + self.offset) * stride
        shift_y = (torch.arange(0., feat_w, device=device) + self.offset) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        stride = shift_x.new_full((shift_xx.shape[0], ), stride)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w = shift_xx.new_full((shift_xx.shape[0], ),
                                         stride)
            stride_h = shift_xx.new_full((shift_yy.shape[0], ),
                                         stride)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts.to(device)
        return all_points
