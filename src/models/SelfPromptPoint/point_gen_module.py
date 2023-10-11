import torch
import torch.nn as nn


class PointGenModulev3(nn.Module):
    def __init__(
            self,
            image_size: int = 1024,
            num_points: int = 1
    ) -> None:
        super(PointGenModulev3, self).__init__()
        # Input: (batch_size, 3, 1024, 1024)
        # Output: (batch_size, 10, 2)
        self.image_size = image_size
        self.num_points = num_points
        # Define lightweight convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, stride=2, padding=1)
        # # Global Average Pooling (GAP)
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # # Fully connected layer
        # self.fc = nn.Linear(128, num_points * 2)
        # Activation function
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, 3, 256, 256)
        # Output: (batch_size, 10, 2)
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        x = x.squeeze(1)
        coor = self.soft_argmax2d(x)
        coor = coor.unsqueeze(1)
        # x = self.global_avg_pool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = self.sigmoid(x)
        # x = x.view(-1, self.num_points, 2)
        # print(coor * (self.image_size - 1))
        # print(coor * (self.image_size - 1))
        return coor * (self.image_size - 1)
        # return x * self.image_size
    def test(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        return x
    def soft_argmax2d(self, input: torch.Tensor, beta = 64) -> torch.Tensor:
        """
        Compute the soft argmax 2D of a given input heatmap.
        Arguments
        ---------
        input : torch.Tensor
            the given input heatmap with shape :math:(B, H, W).
        Returns
        -------
        torch.Tensor
            the soft argmax heatmap with shape :math:(B, 2).
        """
        # Compute softmax over the input heatmap
        softmax_heatmap = nn.functional.softmax(input.reshape(input.shape[0], -1) * beta, dim=1)
        softmax_heatmap = softmax_heatmap.reshape(input.shape)
        # Create coordinates indices grid
        x_idx = torch.arange(input.size(2), dtype=input.dtype, device=input.device)
        y_idx = torch.arange(input.size(1), dtype=input.dtype, device=input.device)
        x_idx, y_idx = torch.meshgrid(x_idx, y_idx, indexing='xy')
        # Compute the expected x and y coordinates
        expected_y = torch.sum(y_idx * softmax_heatmap, dim=(1, 2))
        expected_x = torch.sum(x_idx * softmax_heatmap, dim=(1, 2))
        # Normalize the coordinates from [0, 1]
        expected_x /= float(input.size(2) - 1)
        expected_y /= float(input.size(1) - 1)
        return torch.stack([expected_x, expected_y], dim=1)

    def _str_(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super(PointGenModulev3, self)._str_() + f'\nTrainable parameters: {n_params}'


class PointGenModuleWithViT(nn.Module):
    def __init__(
            self,
            image_size: int = 1024,
            num_points: int = 1
    ) -> None:
        super(PointGenModuleWithViT, self).__init__()
        # Input: (batch_size, 3, 1024, 1024)
        # Output: (batch_size, 10, 2)
        self.image_size = image_size
        self.num_points = num_points
        # Define lightweight convolutional layers
        self.conv1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, 3, 256, 256)
        # Output: (batch_size, 10, 2)
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.squeeze(1)
        coor = self.soft_argmax2d(x)
        coor = coor.unsqueeze(1)
        # x = self.global_avg_pool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = self.sigmoid(x)
        # x = x.view(-1, self.num_points, 2)
        # print(coor * (self.image_size - 1))
        # print(coor * (self.image_size - 1))
        return coor * (self.image_size - 1)
        # return x * self.image_size
    def test(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        return x
    def soft_argmax2d(self, input: torch.Tensor, beta = 64) -> torch.Tensor:
        """
        Compute the soft argmax 2D of a given input heatmap.
        Arguments
        ---------
        input : torch.Tensor
            the given input heatmap with shape :math:(B, H, W).
        Returns
        -------
        torch.Tensor
            the soft argmax heatmap with shape :math:(B, 2).
        """
        # Compute softmax over the input heatmap
        softmax_heatmap = nn.functional.softmax(input.reshape(input.shape[0], -1) * beta, dim=1)
        softmax_heatmap = softmax_heatmap.reshape(input.shape)
        # Create coordinates indices grid
        x_idx = torch.arange(input.size(2), dtype=input.dtype, device=input.device)
        y_idx = torch.arange(input.size(1), dtype=input.dtype, device=input.device)
        x_idx, y_idx = torch.meshgrid(x_idx, y_idx, indexing='xy')
        # Compute the expected x and y coordinates
        expected_y = torch.sum(y_idx * softmax_heatmap, dim=(1, 2))
        expected_x = torch.sum(x_idx * softmax_heatmap, dim=(1, 2))
        # Normalize the coordinates from [0, 1]
        expected_x /= float(input.size(2) - 1)
        expected_y /= float(input.size(1) - 1)
        return torch.stack([expected_x, expected_y], dim=1)

    def _str_(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super(PointGenModuleWithViT, self)._str_() + f'\nTrainable parameters: {n_params}'