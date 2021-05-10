""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,in_channels,out_channels,ks=3):
        super().__init__()
        pl = int((ks-1)/2)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=ks,padding=pl),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=ks,padding=pl),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self,in_channels,out_channels,kspool=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kspool,stride=2)
        self.dc = DoubleConv(in_channels,out_channels)

    def forward(self,x):
        y = self.maxpool(x)
        return self.dc(y)
    
class Up(nn.Module):
    """Upscaling then doubleconv"""

    def __init__(self,in_channels,cat_channels,out_channels,kspool=2):
        super().__init__()
        self.up_rd = nn.Upsample(scale_factor=2,mode='bilinear',align_corners = True)
        self.dc = DoubleConv(in_channels+cat_channels,out_channels)
    
    def forward(self,x1,x2):
        x1u = self.up_rd(x1)
        # since x2 and x3 come in batches, the cat should happen at dim=1 (N,[C],H,W)
        x = torch.cat([x2,x1u],dim=1)
        
        return self.dc(x)
