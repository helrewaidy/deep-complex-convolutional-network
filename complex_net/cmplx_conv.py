import torch
import torch.nn as nn


class ComplexConv(nn.Module):
    def __init__(
        self,
        rank,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        conv_transposed=False
    ):
        super(ComplexConv, self).__init__()
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv_transposed = conv_transposed

        self.conv_args = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "groups": self.groups,
            "bias": self.bias
        }

        if self.conv_transposed:
            self.conv_args["output_padding"] = self.output_padding
        else:
            self.conv_args["dilation"] = self.dilation

        self.conv_func = {1: nn.Conv1d if not self.conv_transposed else nn.ConvTranspose1d,
                          2: nn.Conv2d if not self.conv_transposed else nn.ConvTranspose2d,
                          3: nn.Conv3d if not self.conv_transposed else nn.ConvTranspose3d}[self.rank]

        self.real_conv = self.conv_func(**self.conv_args)
        self.imag_conv = self.conv_func(**self.conv_args)

    def forward(self, input):
        '''
            Considering a complex-valued input z = x + iy to be convolved by complex-valued filter h = a + ib
            where Output O = z * h, where * is a complex convolution operator, then O = x*a + i(x*b)+ i(y*a) - y*b
            so we need to calculate each of the 4 convolution operations in the previous equation,
            one simple way to implement this as two conolution layers, one layer for the real weights (a)
            and the other for imaginary weights (b), this can be done by concatenating both real and imaginary
            parts of the input and convolve over both of them as follows:
            c_r = [x; y] * a , and  c_i= [x; y] * b, so that
            O_real = c_r[real] - c_i[real], and O_imag = c_r[imag] - c_i[imag]
        '''
        input_real, input_imag = torch.unbind(input, dim=-1)

        output_real = self.real_conv(input_real) - self.imag_conv(input_imag)
        output_imag = self.real_conv(input_imag) + self.imag_conv(input_real)
        
        output = torch.stack([output_real, output_imag], dim=-1)
        return output


class ComplexConv1d(ComplexConv):
    """Applies a 1D Complex convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, L_{in}, 2)`
        - Output: :math:`(N, C_{out}, L_{out}, 2)`
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size, 2)
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels, 2)
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(ComplexConv1d, self).__init__(
            rank=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )


class ComplexConv2d(ComplexConv):
    """Applies a 2D Complex convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, L_{in}, 2)`
        - Output: :math:`(N, C_{out}, L_{out}, 2)`
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size, 2)
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels, 2)
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1),
                 padding=(0, 0),
                 dilation=(1, 1),
                 groups=1,
                 bias=True):
        super(ComplexConv2d, self).__init__(
            rank=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )


class ComplexConv3d(ComplexConv):
    """Applies a 3D complex convolution over an input signal composed of several input
    planes.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, L_{in}, 2)`
        - Output: :math:`(N, C_{out}, L_{out}, 2)`
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size, 2)
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels, 2)
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1, 1),
                 padding=(0, 0, 0),
                 dilation=(1, 1, 1),
                 groups=1,
                 bias=True):
        super(ComplexConv3d, self).__init__(
            rank=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
