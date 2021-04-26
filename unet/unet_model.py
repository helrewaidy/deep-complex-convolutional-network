import torch.nn as nn
from utils.save_net import *

params = Parameters()

import unet.unet_complex_parts as unet_cmplx
import unet.unet_real_parts as unet_real


#########################################################
# params.MODEL = 0 # Original U-net implementation
# params.MODEL = 3  # Complex U-net (URUS)
# params.MODEL = 3.1 # Complex stacked convolution layers
# params.MODEL = 3.2 # Complex U-net with different kernel configuration
# params.MODEL = 4 # Complex U-Net with residual connection
# params.MODEL = 7 # Real shallow U-net layer [double size] (magNet)
##########################################################

class UNet(nn.Module):
    def __init__(self, n_channels, n_outchannels, img_size=[64, 64]):
        super(UNet, self).__init__()
        if params.MODEL == 0:  # Deep Unet
            self.inc = unet_real.InConv(n_channels, 64)
            self.down1 = unet_real.Down(64, 128)
            self.down2 = unet_real.Down(128, 256)
            self.down3 = unet_real.Down(256, 512)
            self.down4 = unet_real.Down(512, 512)
            self.up1 = unet_real.Up(1024, 256)
            self.up2 = unet_real.Up(512, 128)
            self.up3 = unet_real.Up(256, 64)
            self.up4 = unet_real.Up(128, 64)
            self.outc = unet_real.OutConv(64, n_outchannels)

        elif params.MODEL == 3:
            self.inc = unet_cmplx.InConv(n_channels, 64)
            self.down1 = unet_cmplx.Down(64, 128)
            self.down2 = unet_cmplx.Down(128, 256)
            self.bottleneck = unet_cmplx.BottleNeck(256, 256, False)
            self.up2 = unet_cmplx.Up(256, 128)
            self.up3 = unet_cmplx.Up(128, 64)
            self.up4 = unet_cmplx.Up(64, 64)
            self.ouc = unet_cmplx.OutConv(64, n_channels)

        elif params.MODEL == 3.1:
            self.inc = unet_cmplx.InConv(n_channels, 64)
            self.conv64 = unet_cmplx.InConv(64, 64)
            self.conv64_32 = unet_cmplx.InConv(64, 32)
            self.conv32_64 = unet_cmplx.InConv(32, 64)
            self.conv32 = unet_cmplx.InConv(32, 32)
            self.outc = unet_cmplx.OutConv(64, n_channels)
        #             self.ouc = unet_cmplx.OutConv(n_channels, n_channels)

        elif params.MODEL == 3.2:
            self.inc = unet_cmplx.InConv(n_channels, 256)
            self.down1 = unet_cmplx.Down(256, 128)
            self.down2 = unet_cmplx.Down(128, 64)
            self.bottleneck = unet_cmplx.BottleNeck(64, 64, False)
            self.up2 = unet_cmplx.Up(64, 128)
            self.up3 = unet_cmplx.Up(128, 256)
            self.up4 = unet_cmplx.Up(256, n_channels)
        #             self.ouc = unet_cmplx.OutConv(n_channels, n_channels)

        elif params.MODEL == 4:
            self.inc = unet_cmplx.InConv(n_channels, 64)
            self.down1 = unet_cmplx.Down(64, 128)
            self.down2 = unet_cmplx.Down(128, 256)
            self.bottleneck = unet_cmplx.BottleNeck(256, 256)
            self.up2 = unet_cmplx.Up(256, 128)
            self.up3 = unet_cmplx.Up(128, 64)
            self.up4 = unet_cmplx.Up(64, n_channels)

        elif params.MODEL == 7:  # Deep Unet
            ps = 2
            self.inc = unet_real.InConv(n_channels, 64 * ps)
            self.down1 = unet_real.Down(64 * ps, 128 * ps)
            self.down2 = unet_real.Down(128 * ps, 256 * ps)
            self.bottleneck = unet_real.BottleNeck(256 * ps, 256 * ps, False)
            self.up2 = unet_real.Up(256 * ps, 128 * ps)
            self.up3 = unet_real.Up(128 * ps, 64 * ps)
            self.up4 = unet_real.Up(64 * ps, 64 * ps)
            self.outc = unet_real.OutConv(64 * ps, 1)

    def forward(self, x):

        if params.MODEL == 0:  # Deep Unet
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
            x = self.mpcomb(x)

        elif params.MODEL == 3:
            x0 = x
            x1 = self.inc(x)
            x2, down_x1 = self.down1(x1)
            x3, down_x2 = self.down2(x2)
            x4 = self.bottleneck(x3)
            x = self.up2(x4, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)  # + x0
            x = self.ouc(x)

        elif params.MODEL == 3.1:
            x = self.inc(x)
            x = self.conv64(x)
            x = self.conv64_32(x)
            x = self.conv32(x)
            x = self.conv32_64(x)
            x = self.conv64(x)
            x = self.outc(x)

        elif params.MODEL == 3.2:
            x0 = x
            x1 = self.inc(x)
            x2, down_x1 = self.down1(x1)
            x3, down_x2 = self.down2(x2)
            x4 = self.bottleneck(x3)
            x = self.up2(x4, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)  # + x0

        elif params.MODEL == 4:

            x0 = x
            x1 = self.inc(x)
            x2, down_x1 = self.down1(x1)
            x3, down_x2 = self.down2(x2)
            x4 = self.bottleneck(x3)  # the residual connection is added inside
            x = self.up2(x4, x3) + down_x2
            x = self.up3(x, x2) + down_x1
            x = self.up4(x, x1) + x0

        elif params.MODEL == 7:
            x0 = x
            x1 = self.inc(x)
            x2, down_x1 = self.down1(x1)
            x3, down_x2 = self.down2(x2)
            x4 = self.bottleneck(x3)
            x = self.up2(x4, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)  # + x0
            x = self.outc(x)
        return x
