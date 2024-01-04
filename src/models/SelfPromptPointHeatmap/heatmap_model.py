import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import nn
from torchinfo import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_cfg, act_cfg, kernel_size, padding, heatmap_channel, cat = False, final_block = False):
        super(ConvBlock, self).__init__()
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.kernel_size = kernel_size
        self.padding = padding
        if final_block and cat:
            self.Conv_block = nn.Sequential(
                ConvModule(in_channel + out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                ConvModule(out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                ConvModule(out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                ConvModule(out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                ConvModule(out_channel, heatmap_channel, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            )
        else:
            if cat:
                self.Conv_block = nn.Sequential(
                    ConvModule(in_channel + out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                    # ConvModule(out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                    # ConvModule(out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                    ConvModule(out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                    ConvModule(out_channel, heatmap_channel, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
                )
            else:
                self.Conv_block = nn.Sequential(
                    ConvModule(in_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                    # ConvModule(out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                    # ConvModule(out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                    ConvModule(out_channel, out_channel, kernel_size=self.kernel_size, padding=self.padding, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, groups=1),
                    ConvModule(out_channel, heatmap_channel, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
                )

    def forward(self, x):
        x = self.Conv_block(x)
        return x
    
    def forward_block(self, x):
        for i, module in enumerate(self.Conv_block):
            x = module(x)
            if i == 1:
                feature_map = x
            elif i == 2:
                heatmap = x
        return feature_map, heatmap

    def forward_infer(self, x):
        # x = self.Conv_block(x)
        for i, module in enumerate(self.Conv_block):
            x = module(x)
            if i == 3:
                feature_map = x
            if i == 4:
                final_feature_map = x
        feature_map = torch.sum(feature_map, dim=1, keepdim=True)
        return feature_map, final_feature_map

class PointModelHeatmap(nn.Module):
    def __init__(self) -> None:
        super(PointModelHeatmap, self).__init__()

        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.act_cfg = dict(type='ReLU')
        self.convblock1 = ConvBlock(256, 128, self.norm_cfg, self.act_cfg, 3, 1, 1, False, False)
        self.convblock2 = ConvBlock(256, 128, self.norm_cfg, self.act_cfg, 3, 1, 1, True, False)
        self.convblock3 = ConvBlock(256, 128, self.norm_cfg, self.act_cfg, 3, 1, 1, True, False)
        # self.convblock4 = ConvBlock(256, 128, self.norm_cfg, self.act_cfg, 1, 0, 1, True)
        self.convblock4 = ConvBlock(256, 128, self.norm_cfg, self.act_cfg, 3, 1, 1, True, True)

    def forward_infer(self, input_embed):

        feature_map1, out1 = self.convblock1.forward_block(input_embed)
        cat_1 = torch.cat([input_embed, feature_map1], 1)
        feature_map2, out2 = self.convblock2.forward_block(cat_1)
        cat_2 = torch.cat([input_embed, feature_map2], 1)
        feature_map3, out3 = self.convblock3.forward_block(cat_2)
        cat_3 = torch.cat([input_embed, feature_map3], 1)
        feature_map, out4= self.convblock4.forward_infer(cat_3)
        heat_map = [out1, out2, out3, out4]

        return heat_map, feature_map
    
    def forward(self, input_embed):

        feature_map1, out1 = self.convblock1.forward_block(input_embed)
        cat_1 = torch.cat([input_embed, feature_map1], 1)
        feature_map2, out2 = self.convblock2.forward_block(cat_1)
        cat_2 = torch.cat([input_embed, feature_map2], 1)
        feature_map3, out3 = self.convblock3.forward_block(cat_2)
        cat_3 = torch.cat([input_embed, feature_map3], 1)
        out4 = self.convblock4(cat_3)
        heat_map = [out1, out2, out3, out4]

        return heat_map
    
model =  PointModelHeatmap()
summary(model, input_size=(1, 256, 64, 64))