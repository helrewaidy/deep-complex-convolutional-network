'''
Created on May 21, 2018

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from complexnet.cmplxconv import ComplexConv2d
from complexnet.cmplxmodrelu import ModReLU
from complexnet.cmplxupsample import ComplexUpsample
from complexnet.radialbn2 import RadialBatchNorm2d
from complexnet.zrelu import ZReLU
from parameters import Parameters

params = Parameters()


def Activation(*args):
    if params.activation_func == 'CReLU':
        return nn.ReLU(inplace=True)
    elif params.activation_func == 'modReLU':
        return ModReLU(*args)
    elif params.activation_func == 'ZReLU':
        return ZReLU(polar=False)


class DoubleConv(nn.Module):
    '''(conv => ReLU => BN) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ComplexConv2d(in_ch, out_ch, [3, 3], padding=(1, 1)),
            RadialBatchNorm2d(out_ch),
            Activation(out_ch),
            #            ComplexDropout2d(params.dropout_ratio),
            ComplexConv2d(out_ch, out_ch, [3, 3], padding=(1, 1)),
            RadialBatchNorm2d(out_ch),
            Activation(out_ch),
            #            ComplexDropout2d(params.dropout_ratio)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_ch):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            ComplexConv2d(in_ch, in_ch, [3, 3], stride=(2, 2), padding=(1, 1)),
            RadialBatchNorm2d(in_ch),
            Activation(in_ch)
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
            #            ComplexDropout2d(params.dropout_ratio),
            ComplexConv2d(in_ch, 2 * in_ch, [3, 3], padding=(1, 1)),
            RadialBatchNorm2d(2 * in_ch),
            Activation(2 * in_ch),
            #            ComplexDropout2d(params.dropout_ratio),
            ComplexConv2d(2 * in_ch, out_ch, [3, 3], padding=(1, 1)),
            RadialBatchNorm2d(out_ch),
            Activation(out_ch)
        )

    def forward(self, x):
        down_x = self.down_conv(x)
        if self.residual_connection:
            x = self.double_conv(down_x) + down_x
        else:
            x = self.double_conv(down_x)

        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()

        self.up = ComplexUpsample(scale_factor=2, mode='bilinear')

        self.conv = nn.Sequential(
            ComplexConv2d(in_ch * 2, in_ch, [3, 3], padding=(1, 1)),
            RadialBatchNorm2d(in_ch),
            Activation(in_ch),
            ComplexConv2d(in_ch, out_ch, [3, 3], padding=(1, 1)),
            RadialBatchNorm2d(out_ch),
            Activation(out_ch)
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

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = ComplexConv2d(in_ch, out_ch, [1, 1])

    def forward(self, x):
        x = self.conv(x)
        return x
