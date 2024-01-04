import torch
import torch.nn as nn


class Scale(nn.Module):
    """A learnable scale parameter."""
    def __init__(self,
                 channels,
                 init_val=1.0):
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_val * torch.ones(channels, 1, 1),
                         dtype=torch.float),
            requires_grad=True)

    def forward(self, x):
        return x * self.scale