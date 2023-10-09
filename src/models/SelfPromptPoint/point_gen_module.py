import torch
import torch.nn as nn

from segment_anything.modeling.common import LayerNorm2d

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
        
    def test(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        
        return x[:, 1, :, :]

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
        
        return x

        # return x * self.image_size

    def __str__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return super(PointGenModule, self).__str__() + f'\nTrainable parameters: {n_params}'


# class PointGenModulev2(nn.Module):
#     def __init__(self,
#                  image_size: int = 1024,
#                  num_points: int = 1):
#         super(PointGenModulev2, self).__init__()
#         # Input: (batch_size, 256, 64, 64)
#         # Output: (batch_size, num_points, 2)
#         self.image_size = image_size
#         self.num_points = num_points

#         self.conv = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=1),
#             LayerNorm2d(128),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             LayerNorm2d(128)
#         )

#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(128, self.num_points * 2)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.pool(x)
#         x = x.view(x.shape[0], -1)
#         x = self.fc(x)
#         x = self.sigmoid(x)
#         x = x.view(-1, self.num_points, 2)

#         return x * self.image_size

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
            the given input heatmap with shape :math:`(B, H, W)`.
        Returns
        -------
        torch.Tensor
            the soft argmax heatmap with shape :math:`(B, 2)`.
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

    def __str__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return super(PointGenModulev3, self).__str__() + f'\nTrainable parameters: {n_params}'
    

class PointGenModulev2(nn.Module):
    def __init__(self,
                 image_size: int = 1024,
                 num_points: int = 1):
        super(PointGenModulev2, self).__init__()
        # Input: (batch_size, 256, 64, 64)
        # Output: (batch_size, num_points, 2)
        self.image_size = image_size
        self.num_points = num_points

        self.conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            LayerNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            LayerNorm2d(128)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, self.num_points * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = x.view(-1, self.num_points, 2)

        return x * self.image_size

class PointGenModuleWViT(nn.Module):
    def __init__(
            self,
            image_size: int = 1024,
            num_points: int = 1
    ) -> None:
        super(PointGenModuleWViT, self).__init__()

        # Input: (batch_size, 3, 1024, 1024)
        # Output: (batch_size, 10, 2)

        self.image_size = image_size
        self.num_points = num_points

        # Define lightweight convolutional layers
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        # # Global Average Pooling (GAP)
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # # Fully connected layer
        # self.fc = nn.Linear(128, num_points * 2)

        # Activation function
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(32)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, 3, 256, 256)
        # Output: (batch_size, 10, 2)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
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
            the given input heatmap with shape :math:`(B, H, W)`.
        Returns
        -------
        torch.Tensor
            the soft argmax heatmap with shape :math:`(B, 2)`.
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

    def __str__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return super(PointGenModuleWViT, self).__str__() + f'\nTrainable parameters: {n_params}'    




# class HourglassBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(HourglassBlock, self).__init__()
#         self.down = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
#         self.proc = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.down(x)
#         x = self.up(x)
#         x = self.proc(x)
#         return x

# class HourglassNet(nn.Module):
#     def __init__(self):
#         super(HourglassNet, self).__init__()

#         # Initial downsample
#         self.init_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.init_bn = nn.BatchNorm2d(64)
#         self.init_relu = nn.ReLU(inplace=True)

#         # Hourglass modules
#         self.hg1 = HourglassBlock(64, 128)
#         self.hg2 = HourglassBlock(128, 256)
#         # self.hg3 = HourglassBlock(256, 512)
#         # self.hg4 = HourglassBlock(512, 1024)

#         # Output layer to generate heatmap
#         # self.out_conv = nn.Conv2d(1024, 1, kernel_size=1)
#         # self.out_conv = nn.Conv2d(512, 1, kernel_size=1)
#         self.out_conv = nn.Conv2d(256, 1, kernel_size=1)

#     def forward(self, x):
#         x = self.init_conv(x)
#         x = self.init_bn(x)
#         x = self.init_relu(x)

#         x = self.hg1(x)
#         x = self.hg2(x)
#         # x = self.hg3(x)
#         # x = self.hg4(x)

#         out = self.out_conv(x)
#         out = out.squeeze(1)
#         coor = self.soft_argmax2d(out)
#         coor = coor.unsqueeze(1)
#         return coor * 1023
    
#     def soft_argmax2d(self, input: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the soft argmax 2D of a given input heatmap.
#         Arguments
#         ---------
#         input : torch.Tensor
#             the given input heatmap with shape :math:`(B, H, W)`.
#         Returns
#         -------
#         torch.Tensor
#             the soft argmax heatmap with shape :math:`(B, 2)`.
#         """

#         # Compute softmax over the input heatmap
#         softmax_heatmap = nn.functional.softmax(input.reshape(input.shape[0], -1), dim=1)
#         softmax_heatmap = softmax_heatmap.reshape(input.shape)

#         # Create coordinates indices grid
#         x_idx = torch.arange(input.size(2), dtype=input.dtype, device=input.device)
#         y_idx = torch.arange(input.size(1), dtype=input.dtype, device=input.device)
#         x_idx, y_idx = torch.meshgrid(x_idx, y_idx, indexing='xy')
        
#         # Compute the expected x and y coordinates
#         expected_y = torch.sum(y_idx * softmax_heatmap, dim=(1, 2))
#         expected_x = torch.sum(x_idx * softmax_heatmap, dim=(1, 2))

#         # Normalize the coordinates from [0, 1]
#         expected_x /= float(input.size(2) - 1)
#         expected_y /= float(input.size(1) - 1)

#         return torch.stack([expected_x, expected_y], dim=1)
    
#     def __str__(self):
#         n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

#         return super(HourglassNet, self).__str__() + f'\nTrainable parameters: {n_params}'    
    
# if __name__ == "__main__":
#     model = HourglassNet()
#     image = torch.rand(1, 3, 1024, 1024)
#     output = model(image)
    
#     print(output)