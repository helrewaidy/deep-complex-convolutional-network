import torch.nn as nn
from utils.saveNet import *

class ComplexConv(nn.Module):
    def __init__(self, rank, 
                 in_channels,
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0,
                 output_padding=0, 
                 dilation=1, 
                 groups=1, 
                 bias=True,
                 normalize_weight=False,
                 epsilon=1e-7,
                 conv_transposed=False):             
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
        
        self.conv_args = {"in_channels": self.in_channels,
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
            assume complex input z = x + iy needs to be convolved by complex filter h = a + ib
            where Output O = z * h, where * is convolution operator, then O = x*a + i(x*b)+ i(y*a) - y*b
            so we need to calculate each of the 4 convolution operations in the previous equation,
            one simple way to implement this as two conolution layers, one layer for the real weights (a)
            and the other for imaginary weights (b), this can be done by concatenating both real and imaginary
            parts of the input and convolve over both of them as follows:
            c_r = [x; y] * a , and  c_i= [x; y] * b, so that
            O_real = c_r[real] - c_i[real], and O_imag = c_r[imag] - c_i[imag]
        '''

        ndims = input.ndimension()

        input_real = input.narrow(ndims-1, 0, 1).squeeze(ndims-1)
        input_imag = input.narrow(ndims-1, 1, 1).squeeze(ndims-1)

        output_real = self.real_conv(input_real) - self.imag_conv(input_imag)
        output_imag = self.real_conv(input_imag) + self.imag_conv(input_real)

        output = torch.stack([output_real,output_imag], dim=ndims-1)

        return output
    

class ComplexConv1d(ComplexConv):
    
    """Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, L)` and output :math:`(N, C_{out}, L_{out})` can be
    precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor \frac{\text{out_channels}}{\text{in_channels}} \right\rfloor`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid
         `cross-correlation`_, and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, L_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

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
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels)

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
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
    
    """Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, L)` and output :math:`(N, C_{out}, L_{out})` can be
    precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor \frac{\text{out_channels}}{\text{in_channels}} \right\rfloor`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid
         `cross-correlation`_, and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, L_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

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
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels)

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """    

    
    def __init__(self, in_channels,
                 out_channels, 
                 kernel_size, 
                 stride=(1,1), 
                 padding=(0,0), 
                 dilation=(1,1), 
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
    
    """Applies a 3D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, L)` and output :math:`(N, C_{out}, L_{out})` can be
    precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor \frac{\text{out_channels}}{\text{in_channels}} \right\rfloor`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid
         `cross-correlation`_, and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, L_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

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
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels)

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """    

    
    def __init__(self, in_channels,
                 out_channels, 
                 kernel_size, 
                 stride=(1,1,1), 
                 padding=(0,0,0), 
                 dilation=(1,1,1), 
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

    
