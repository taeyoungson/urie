import torch

import torch.nn as nn

from .skunet_parts import SKUp, SKDown


class SKUNet(nn.Module):
    def __init__(self, bilinear=True):
        super(SKUNet, self).__init__()
        self.bilinear = bilinear

        self.down1 = nn.Conv2d(kernel_size=9, padding=4,in_channels=3, out_channels=32)
        self.down2 = SKDown(3, 1, False, 16, 32, 64)
        self.down3 = SKDown(3, 1, False, 16, 64, 64)
        self.up1 = SKUp(3, 1, False, 16, 128, 32, bilinear)
        self.up2 = SKUp(3, 1, False, 16, 64, 16, bilinear)
        self.up3 = nn.Conv2d(kernel_size=3, padding=1, in_channels=16, out_channels=3)

    def forward(self, x):
        x_origin = x
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x)

        return torch.add(x, x_origin), x
