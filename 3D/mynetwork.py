# my networks for different purpose
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from Unet_parts import *

class CO2mask(nn.Module):
    def __init__(self,c1=16):
        super().__init__()
        
        # from (C:2,D,H,W) to (C:c1,D,H,W)
        self.Init = DoubleConv(2,c1)
        #downward
        # from (C:c1,D,H,W) to (C:2*c1,D/2,H/2,W/2)
        self.down1 = Down(c1,2*c1)
        # from (C:2*c1,D/2,H/2,W/2) to (C:4*c1,D/4,H/4,W/4)
        self.down2 = Down(2*c1,4*c1)
        # from (C:32*c1,D/4,H/4,W/4) to (C:32*c1,D/8,H/8,W/8)
        self.down3 = Down(4*c1,16*c1)
        
        # upward
        # from (C:32*c1,D/8,H/8,W/8)"+"(C:4*c1,D/4,H/4,W/4) to (C:4*c1,D/4,H/4,W/4)
        self.up1 = Up(16*c1,4*c1,4*c1)
        # from (C:4*c1,D/4,H/4,W/4)"+"(C:2*c1,D/2,H/2,W/2) to (C:2*c1,H/2,W/2)
        self.up2 = Up(4*c1,2*c1,2*c1)
        # from (C:2*c1,H/2,W/2)"+"(C:c1,D,H,W) to (C:c1,H,W)
        self.up3 = Up(2*c1,c1,c1)
        # from (C:c1,D,H,W) to (C:1,D,H,W)
        self.lin = nn.Conv3d(c1,1,kernel_size=1)
        self.out = nn.Sigmoid()
        
    def forward(self, t):
        # the dimension of t is (C:2,D,H,W)
        t0 = self.Init(t)
        # the dimension of t0 is (C:c1,D,H,W)
        # go downward
        td1 = self.down1(t0)
        td2 = self.down2(td1)
        td3 = self.down3(td2)
        # the dimension of td3 is (C:32*c1,D/8,H/8,W/8): td3 provide all base feature maps
        # go upward
        tu1 = self.up1(td3,td2)
        tu2 = self.up2(tu1,td1)
        tu3 = self.up3(tu2,t0)
        # final conv
        y = self.out(self.lin(tu3))

        return y