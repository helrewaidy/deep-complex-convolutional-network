from utils.saveNet import *
import torch.nn as nn

params = Parameters()

import unet.unet_complex_parts as unet_cmplx
import unet.unet_real_parts as unet_real

#########################################################
# self.MODEL = 0 # Original U-net implementation
# self.MODEL = 3  # Complex U-net (URUS)
# self.MODEL = 3.1 # Complex stacked convolution layers
# self.MODEL = 3.2 # Complex U-net with different kernel configuration
# self.MODEL = 4 # Complex U-Net with residual connection
# self.MODEL = 7 # Real shallow U-net layer [double size] (magNet)
##########################################################

class UNet(nn.Module):
    def __init__(self, n_channels, n_outchannels, img_size=[64, 64]):
        super(UNet, self).__init__()
        if params.MODEL == 0:  # Deep Unet
            self.inc = unet_real.inconv(n_channels, 64)
            self.down1 = unet_real.down(64, 128)
            self.down2 = unet_real.down(128, 256)
            self.down3 = unet_real.down(256, 512)
            self.down4 = unet_real.down(512, 512)
            self.up1 = unet_real.up(1024, 256)
            self.up2 = unet_real.up(512, 128)
            self.up3 = unet_real.up(256, 64)
            self.up4 = unet_real.up(128, 64)
            self.outc = unet_real.outconv(64, n_outchannels)

        elif params.MODEL == 3:
            self.inc = unet_cmplx.inconv(n_channels, 64)
            self.down1 = unet_cmplx.down(64, 128)
            self.down2 = unet_cmplx.down(128, 256)
            self.bottleneck = unet_cmplx.bottleneck(256, 256, False)
            self.up2 = unet_cmplx.up(256, 128)
            self.up3 = unet_cmplx.up(128, 64)
            self.up4 = unet_cmplx.up(64, 64)
            self.ouc = unet_cmplx.outconv(64, n_channels)

        elif params.MODEL == 3.1:
            self.inc = unet_cmplx.inconv(n_channels, 64)
            self.conv64 = unet_cmplx.inconv(64, 64)
            self.conv64_32 = unet_cmplx.inconv(64, 32)
            self.conv32_64 = unet_cmplx.inconv(32, 64)
            self.conv32 = unet_cmplx.inconv(32, 32)
            self.outc = unet_cmplx.outconv(64, n_channels)
        #             self.ouc = unet_cmplx.outconv(n_channels, n_channels)

        elif params.MODEL == 3.2:
            self.inc = unet_cmplx.inconv(n_channels, 256)
            self.down1 = unet_cmplx.down(256, 128)
            self.down2 = unet_cmplx.down(128, 64)
            self.bottleneck = unet_cmplx.bottleneck(64, 64, False)
            self.up2 = unet_cmplx.up(64, 128)
            self.up3 = unet_cmplx.up(128, 256)
            self.up4 = unet_cmplx.up(256, n_channels)
        #             self.ouc = unet_cmplx.outconv(n_channels, n_channels)

        elif params.MODEL == 4:
            self.inc = unet_cmplx.inconv(n_channels, 64)
            self.down1 = unet_cmplx.down(64, 128)
            self.down2 = unet_cmplx.down(128, 256)
            self.bottleneck = unet_cmplx.bottleneck(256, 256)
            self.up2 = unet_cmplx.up(256, 128)
            self.up3 = unet_cmplx.up(128, 64)
            self.up4 = unet_cmplx.up(64, n_channels)

        elif params.MODEL == 5:
            self.inc = unet_cmplx.inconv(n_channels, 64)
            self.down1 = unet_cmplx.down(64, 128)
            self.down2 = unet_cmplx.down(128, 128)
            self.up3 = unet_cmplx.up(256, 128)
            self.up4 = unet_cmplx.up(192, 64)
            self.outc = unet_cmplx.outconv(64, n_channels)
            self.coilcmp = unet_cmplx.outconv(n_channels, 1)

        elif params.MODEL == 7:  # Deep Unet
            ps = 2
            self.inc = unet_real.inconv(n_channels, 64 * ps)
            self.down1 = unet_real.down(64 * ps, 128 * ps)
            self.down2 = unet_real.down(128 * ps, 256 * ps)
            self.bottleneck = unet_real.bottleneck(256 * ps, 256 * ps, False)
            self.up2 = unet_real.up(256 * ps, 128 * ps)
            self.up3 = unet_real.up(128 * ps, 64 * ps)
            self.up4 = unet_real.up(64 * ps, 64 * ps)
            self.outc = unet_real.outconv(64 * ps, 1)

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

        elif params.MODEL == 5:
            x0 = x
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up3(x3, x2)
            x = self.up4(x, x1)
            x = self.outc(x) + x0  #### Residual connection
            x = self.coilcmp(x)

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
