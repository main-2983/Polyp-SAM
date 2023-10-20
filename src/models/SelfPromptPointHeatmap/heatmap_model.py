import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_cfg, act_cfg, kernel_size, padding, heatmap_channel, cat = False):
        super(ConvBlock, self).__init__()
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.kernel_size = kernel_size
        self.padding = padding

        if cat:
            self.Conv_block = nn.Sequential(
                ConvModule(in_channel + heatmap_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                ConvModule(out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                ConvModule(out_channel, heatmap_channel, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            )
        else:
            self.Conv_block = nn.Sequential(
                ConvModule(in_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                ConvModule(out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                ConvModule(out_channel, heatmap_channel, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            )

    def forward(self, x):
        x = self.Conv_block(x)
        return x

# class OutputBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, norm_cfg, act_cfg, kernel_size, padding):
#         super(OutputBlock, self).__init__()
#         self.norm_cfg = norm_cfg
#         self.act_cfg = act_cfg

#         self.Conv_block = nn.Sequential(
#             ConvModule(in_channel, in_channel, kernel_size=3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=in_channel),
#             ConvModule(in_channel, out_channel, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
#         )

#     def forward(self, x):
#         pass

class PointModelHeatmap(nn.Module):
    def __init__(self) -> None:
        super(PointModelHeatmap, self).__init__()

        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.act_cfg = dict(type='ReLU')
        self.convblock1 = ConvBlock(256, 128, self.norm_cfg, self.act_cfg, 3, 1, 1, False)
        self.convblock2 = ConvBlock(256, 128, self.norm_cfg, self.act_cfg, 3, 1, 1, True)
        self.convblock3 = ConvBlock(256, 128, self.norm_cfg, self.act_cfg, 3, 1, 1, True)
        # self.convblock4 = ConvBlock(256, 128, self.norm_cfg, self.act_cfg, 1, 0, 1, True)
        self.convblock4 = ConvBlock(256, 128, self.norm_cfg, self.act_cfg, 3, 1, 1, True)

    def forward(self, input_embed):

        out1 = self.convblock1(input_embed)
        cat_1 = torch.cat([input_embed, out1], 1)
        out2 = self.convblock2(cat_1)
        cat_2 = torch.cat([input_embed, out2], 1)
        out3 = self.convblock3(cat_2)
        cat_3 = torch.cat([input_embed, out3], 1)
        out4 = self.convblock4(cat_3)
        heat_map = [out1, out2, out3, out4]

        return heat_map


