import torch
import torch.nn as nn


class PointGenModule(nn.Module):
    def __init__(
            self,
            image_size: int = 1024,
            num_points: int = 10
    ) -> None:
        super(PointGenModule, self).__init__()

        # Input: (batch_size, 3, 1024, 1024)
        # Output: (batch_size, 10, 2)

        self.image_size = image_size
        self.num_points = num_points

        # Define lightweight convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.fc = nn.Linear(128, num_points * 2)

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

        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.sigmoid(x)

        x = x.view(-1, self.num_points, 2)

        return x * self.image_size

    def __str__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return super(PointGenModule, self).__str__() + f'\nTrainable parameters: {n_params}'