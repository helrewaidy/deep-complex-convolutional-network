import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_net.cmplx_conv import ComplexConv1d, ComplexConv2d, ComplexConv3d
from complex_net.cmplx_activation import CReLU, ModReLU, ZReLU
from complex_net.cmplx_upsample import ComplexUpsample
from complex_net.radial_bn import RadialBatchNorm1d, RadialBatchNorm2d, RadialBatchNorm3d
from complex_net.cmplx_dropout import ComplexDropout
from configs import config


def complex_conv(in_ch, out_ch, **kwargs):
    conv = {
        1: ComplexConv1d,
        2: ComplexConv2d,
        3: ComplexConv3d
    }[config.spatial_dimentions]
    if 'kernel_size' not in kwargs:
        kwargs['kernel_size'] = config.kernel_size
    return conv(
        in_ch,
        out_ch,
        bias=config.bias,
        **kwargs
    )


def activation(in_channels=None, **kwargs):
    if config.activation == 'modReLU':
        kwargs['in_channels'] = in_channels
    return {
        'CReLU':   CReLU,
        'modReLU': ModReLU,
        'ZReLU':   ZReLU
    }[config.activation](**kwargs)


def batch_norm(in_channels=None, **kwargs):
    bn = {
        1: RadialBatchNorm1d,
        2: RadialBatchNorm2d,
        3: RadialBatchNorm3d
    }[config.spatial_dimentions]
    return bn(in_channels, t=config.bn_t, **kwargs)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            complex_conv(in_ch, out_ch, padding=1),
            batch_norm(out_ch),
            activation(out_ch),
            ComplexDropout(config.dropout_ratio),
            complex_conv(out_ch, out_ch, padding=1),
            batch_norm(out_ch),
            activation(out_ch),
            ComplexDropout(config.dropout_ratio)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_ch):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            complex_conv(in_ch, in_ch, stride=2, padding=1),
            batch_norm(in_ch),
            activation(in_ch)
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
            complex_conv(in_ch, 2 * in_ch, padding=1),
            batch_norm(2 * in_ch),
            activation(2 * in_ch),
            ComplexDropout(config.dropout_ratio),
            complex_conv(2 * in_ch, out_ch, padding=1),
            batch_norm(out_ch),
            activation(out_ch),
            ComplexDropout(config.dropout_ratio)
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
            complex_conv(in_ch * 2, in_ch, padding=1),
            batch_norm(in_ch),
            activation(in_ch),
            ComplexDropout(config.dropout_ratio),
            complex_conv(in_ch, out_ch, padding=1),
            batch_norm(out_ch),
            activation(out_ch),
            ComplexDropout(config.dropout_ratio)
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
        self.conv = complex_conv(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
