import torch.nn as nn
import complex_net.cmplx_blocks as unet_cmplx
from configs import config


class CUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CUNet, self).__init__()
        self.inc = unet_cmplx.InConv(in_channels, 64)
        self.down1 = unet_cmplx.Down(64, 128)
        self.down2 = unet_cmplx.Down(128, 256)
        self.bottleneck = unet_cmplx.BottleNeck(256, 256, False)
        self.up2 = unet_cmplx.Up(256, 128)
        self.up3 = unet_cmplx.Up(128, 64)
        self.up4 = unet_cmplx.Up(64, 64)
        self.ouc = unet_cmplx.OutConv(64, out_channels)

    def forward(self, x):
        x0 = x
        x1 = self.inc(x)
        x2, _ = self.down1(x1)
        x3, _ = self.down2(x2)
        x4 = self.bottleneck(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = x + x0 if config.unet_global_residual_conn else x
        x = self.ouc(x)

        return x
