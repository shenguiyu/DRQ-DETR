import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import DeformConv2d

class EfficientOffset(nn.Module):
    """轻量级偏移量生成网络"""
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.offset_conv = nn.Sequential(
            
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, 
                      kernel_size=1, bias=True)
        )
        
    def forward(self, x):
        return self.offset_conv(x)

class LightweightChannelAttention(nn.Module):
    """轻量级自适应通道注意力"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        
        gamma = 2
        b = 1
        t = int(abs(math.log(channel, 2) + b) / gamma)
        kernel_size = t if t % 2 else t + 1
        
        
        self.conv = nn.Conv1d(1, 1, kernel_size=int(kernel_size), 
                              padding=int((kernel_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        
        y = self.avg_pool(x).view(b, c, 1)
        
        
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        y = self.sigmoid(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)

class DeformableConvBranch(nn.Module):
    def __init__(self, in_channels, dilation, groups):
        super().__init__()
        kernel_size = 3
        padding = dilation * (kernel_size - 1) // 2
        
        self.offset_gen = EfficientOffset(in_channels, kernel_size)
        self.deform_conv = DeformConv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        
        offset = self.offset_gen(x)
        
        
        x = self.deform_conv(x, offset)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MultiScaleDeformableConv(nn.Module):
    """多尺度可变形卷积模块"""
    def __init__(self, in_channels, groups):
        super().__init__()
        self.groups = groups
        
        
        self.branches = nn.ModuleList([
            DeformableConvBranch(in_channels, dilation=1, groups=groups),
            DeformableConvBranch(in_channels, dilation=2, groups=groups),
            DeformableConvBranch(in_channels, dilation=3, groups=groups)
        ])
        
        
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        
        fused = torch.cat(branch_outputs, dim=1)
        return self.fusion(fused)

class MSDCA(nn.Module):
    """多尺度可变形通道注意力"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        groups = max(1, in_channels // 8)  
        
        
        self.deform_conv = MultiScaleDeformableConv(in_channels, groups)
        
        
        self.channel_att = LightweightChannelAttention(in_channels, reduction)
        
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        identity = x
        
        
        deform_feat = self.deform_conv(x)
        
        
        att_feat = self.channel_att(deform_feat)
        
        
        out = self.conv_out(att_feat) + identity
        return out