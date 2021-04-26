import torch.nn as nn
from utils.polar_transforms import *
from utils.save_net import *


class RadialBatchNormalize(nn.Module):

    def __init__(self, rank, num_features, t=5, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 polar=False):
        super(RadialBatchNormalize, self).__init__()
        self.rank = rank
        self.num_features = num_features
        self.t = t
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.polar = polar

        self.bn_func = {1: nn.BatchNorm1d,
                        2: nn.BatchNorm2d,
                        3: nn.BatchNorm3d}[self.rank](num_features=num_features,
                                                      eps=eps,
                                                      momentum=momentum,
                                                      affine=affine,
                                                      track_running_stats=track_running_stats)

    def forward(self, input):
        ndims = input.ndimension()

        input_real = input.narrow(ndims - 1, 0, 1).squeeze(ndims - 1)
        input_imag = input.narrow(ndims - 1, 1, 1).squeeze(ndims - 1)

        if not self.polar:
            mag, phase = convert_cylindrical_to_polar(input_real, input_imag)
        else:
            mag = input_real
            phase = input_imag

        # normalize the magnitude (see paper: El-Rewaidy et al. "Deep complex convolutional network for fast reconstruction of 3D late gadolinium enhancement cardiac MRI", NMR in Biomedicne, 2020)
        output_mag_norm = self.bn_func(
            mag) + self.t  # Normalize the radius to be around self.t (i.e. 5 std) (1 also works fine)

        if not self.polar:
            output_real, output_imag = convert_polar_to_cylindrical(output_mag_norm, phase)
        else:
            output_real = output_mag_norm
            output_imag = phase

        output = torch.stack((output_real, output_imag), dim=ndims - 1)

        return output


class RadialBatchNorm1d(RadialBatchNormalize):
    r"""Applies Radial Batch Normalization over a 2D and 3D  input (a mini-batch of 1D
    inputs with optional additional channel dimension)
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L, 2)`
        - Output: :math:`(N, C)` or :math:`(N, C, L, 2)` (same shape as input)

    """

    def __init__(self,
                 num_features,
                 t=5,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 polar=False):
        super(RadialBatchNorm1d, self).__init__(
            rank=1,
            num_features=num_features,
            t=t,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            polar=polar
        )


class RadialBatchNorm2d(RadialBatchNormalize):
    r"""Applies Radial Batch Normalization over a 4D  input (a mini-batch of 2D
    inputs with optional additional channel dimension)
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W, 2)`
        - Output: :math:`(N, C, H, W, 2)` (same shape as input)
    """

    def __init__(self,
                 num_features,
                 t=5,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 polar=False):
        super(RadialBatchNorm2d, self).__init__(
            rank=2,
            num_features=num_features,
            t=t,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            polar=polar
        )


class RadialBatchNorm3d(RadialBatchNormalize):
    r"""Applies Radial Batch Normalization over a 4D  input (a mini-batch of 2D
    inputs with optional additional channel dimension)
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W, 2)`
        - Output: :math:`(N, C, D, H, W, 2)` (same shape as input)
    """

    def __init__(self,
                 num_features,
                 t=5,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 polar=False):
        super(RadialBatchNorm3d, self).__init__(
            rank=3,
            num_features=num_features,
            t=t,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            polar=polar
        )
