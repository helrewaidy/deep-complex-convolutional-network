import torch.nn as nn
from utils.saveNet import *

class _cmplxDropoutNd(nn.Module):

    def __init__(self, rank, p=0.5, inplace=True):
        super(_cmplxDropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.rank = rank
        self.p = p
        self.inplace = inplace
#         self.dropout = {1: nn.Dropout,
#                         2: nn.Dropout2d,
#                         3: nn.Dropout3d}[self.rank](self.p, self.inplace)
        
    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.p, inplace_str)
    
    def forward(self, input):
        if not self.training or self.p == 0:
            return input
        
        input_shape = input.shape
        
        if self.p == 1:
            return torch.FloatTensor(input_shape).to(input.device).zero_() 

        ndims = input.ndimension()
        msk = torch.FloatTensor(input_shape[:-1]).to(input.device).uniform_().unsqueeze(-1) > self.p  
        msk = torch.cat([msk,msk], ndims-1)
        
        output = input * msk.to(torch.float32)
        
        return output

class ComplexDropout(_cmplxDropoutNd):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. The elements to zero are randomized on every forward call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def __init__(self, p=0.5, inplace=False):             
        super(ComplexDropout, self).__init__(
            rank=1,
            p=p,
            inplace = inplace
            )    




class ComplexDropout2d(_cmplxDropoutNd):
    r"""Randomly zeroes whole channels of the input tensor.
    The channels to zero-out are randomized on every forward call.

    Usually the input comes from :class:`nn.Conv2d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> m = nn.Dropout2d(p=0.2)
        >>> input = torch.randn(20, 16, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280
    """
    def __init__(self, p=0.5, inplace=False):             
        super(ComplexDropout2d, self).__init__(
            rank=2,
            p=p,
            inplace=inplace
            )    
        


class ComplexDropout3d(_cmplxDropoutNd):
    r"""Randomly zeroes whole channels of the input tensor.
    The channels to zero are randomized on every forward call.

    Usually the input comes from :class:`nn.Conv3d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout3d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> m = nn.Dropout3d(p=0.2)
        >>> input = torch.randn(20, 16, 4, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280
    """

    def __init__(self, p=0.5, inplace=False):             
        super(ComplexDropout3d, self).__init__(
            rank=3,
            p=p,
            inplace=inplace
            )    


