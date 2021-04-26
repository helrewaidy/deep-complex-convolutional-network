'''
Created on May 21, 2018

@author: helrewaidy
'''
# sub-parts of the U-Net model

import torch

import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''(conv => ReLU => BN) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, affine=True),  # , affine=False
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, affine=True),  # , affine=False
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_ch):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, [3, 3], stride=(2, 2), padding=1),
            nn.BatchNorm2d(in_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down_conv = DownConv(in_ch)
        self.double_conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        down_x = self.down_conv(x)
        x = self.double_conv(down_x)
        return x, down_x


class BottleNeck(nn.Module):
    def __init__(self, in_ch, out_ch, residual_connection=True):
        super(BottleNeck, self).__init__()
        self.residual_connection = residual_connection
        self.down_conv = DownConv(in_ch)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, 2 * in_ch, 3, padding=1),
            nn.BatchNorm2d(2 * in_ch, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(2 * in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        down_x = self.down_conv(x)
        if self.residual_connection:
            x = self.double_conv(down_x) + down_x
        else:
            x = self.double_conv(down_x)

        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch, affine=True),  # , affine=False
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class MagPhaseCombine(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MagPhaseCombine, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, [1, 1], padding=(0, 0))
        )

    def forward(self, x):
        t = torch.split(x, int(x.size()[2] / 2), dim=2)
        xt = [i for i in t]
        x1 = xt[0]
        x2 = xt[1]
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1d(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, [1, 1])

    def forward(self, x):
        x = self.conv(x)
        return x
