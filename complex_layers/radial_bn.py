import torch
import torch.nn as nn
from utils.polar_transforms import (
    convert_cylindrical_to_polar,
    convert_polar_to_cylindrical,
)


class RadialNorm(nn.Module):

    def __init__(
        self,
        rank,
        num_features,
        t=5,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        polar=False
    ):
        """Radial Batch Normalization

        Parameters
        ----------
        rank : int
            The spatial dimension of the input tensor.
        num_features : int
            The number of features of the input tensor.
        t : float
            The threshold for the normalization.
        eps : float
            The epsilon for the normalization.
        momentum : float
            The momentum for the normalization.
        affine : bool
            If True, this module has learnable affine parameters.
        track_running_stats : bool
            If True, this module tracks the running mean and variance,
            and during training time uses the running mean and variance to normalize.
            During testing time, this module uses the mean and variance of the
            input statistics to normalize.
        polar : bool
            If True, the input is in the polar form (magnitude and phase).
            If False, the input is in the cylindrical form (real and imag).
        """
        super(RadialNorm, self).__init__()
        self.rank = rank
        self.num_features = num_features
        self.t = t
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.polar = polar

        bns = {
            1: nn.BatchNorm1d,
            2: nn.BatchNorm2d,
            3: nn.BatchNorm3d
        }
        self.bn_func = bns[self.rank](
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )

    def forward(self, input):
        real, imag = torch.unbind(input, -1)
        mag, phase = convert_cylindrical_to_polar(real, imag) if not self.polar \
            else (real, imag)

        # normalize the magnitude (see paper: El-Rewaidy et al. "Deep complex 
        # convolutional network for fast reconstruction of 3D late gadolinium 
        # enhancement cardiac MRI", NMR in Biomedicne, 2020)
        # Normalize the radius to be around self.t (i.e. 5 std) (1 also works fine)
        norm_mag = self.bn_func(mag) + self.t
        
        output_real, output_imag = convert_polar_to_cylindrical(norm_mag, phase) \
            if not self.polar else (norm_mag, phase)
        output = torch.stack((output_real, output_imag), dim=-1)

        return output


class RadialBatchNorm1d(RadialNorm):
    r"""Applies Radial Batch Normalization over a 2D and 3D  input (a mini-batch of 1D
    complex inputs with optional additional channel dimension)

    Parameters
    ----------
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


class RadialBatchNorm2d(RadialNorm):
    r"""Applies Radial Batch Normalization over a 5D complex input (a mini-batch of 2D
    complex inputs with optional additional channel dimension)
    
    Parameters
    ----------
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


class RadialBatchNorm3d(RadialNorm):
    r"""Applies Radial Batch Normalization over a 6D complex input (a mini-batch of 3D
    complex inputs with optional additional channel dimension)

    Parameters
    ----------
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
