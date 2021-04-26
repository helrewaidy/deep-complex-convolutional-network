import numpy as np
import torch.nn as nn

from .polar_transforms import *


def unsqueeze_n(x, dims=[-1]):
    for i in dims:
        x = x.unsqueeze(i)
    return x


def magnitude(input):
    real, imag = torch.unbind(input, -1)
    return (real ** 2 + imag ** 2) ** 0.5


def complex_std(x):
    '''
    Standard deviation of real and imaginary channels
    STD = sqrt( E{(x-mu)(x-mu)*} ), where * is the complex conjugate,

    - Source: https://en.wikipedia.org/wiki/Variance#Generalizations
    '''
    mu = torch.mean(torch.mean(x, 2, True), 3, True)

    xm = torch.sum(((x - mu) ** 2), 2, True)
    return torch.mean(torch.mean(xm, 2, True), 3, True) ** (0.5)


def normalizeComplexBatch(x):
    ''' normalize real and imaginary channels'''
    return (x - torch.mean(torch.mean(x, 2, True), 3, True)) / complex_std(x)


def log_mag(x, polar=False):
    '''calculates the log of the magnitude in a complex tensor x'''
    if not polar:
        x = convert_cylindrical_to_polar(x)

    ndims = x.ndimension()
    mag, phase = torch.unbind(x, -1)
    x = torch.stack([torch.log(1 + mag), phase], dim=ndims - 1)

    if not polar:
        x = convert_polar_to_cylindrical(x)

    return x


def exp_mag(x, polar=False):
    '''calculates the exponential of the magnitude in a complex tensor x'''
    if not polar:
        x = convert_cylindrical_to_polar(x)

    ndims = x.ndimension()
    mag, phase = torch.unbind(x, -1)
    x = torch.stack([torch.exp(mag) - 1, phase], dim=ndims - 1)

    if not polar:
        x = convert_polar_to_cylindrical(x)

    return x


def normalize_complex_batch_by_magnitude_only(x, polar=False, normalize_over_channel=False):
    '''
    normalize the complex batch by making the magnitude of mean 1 and std 1, and keep the phase as it is
    :param:
    x: is the input tensor to be normalized
    polar: if x is in the polar form already (i.e. magnitude and phase)
    normalize_over_channel: if the normalization will be performed over all channels
    '''

    shift_mean = 5
    if not polar:
        x = convert_cylindrical_to_polar(x)

    mag, phase = torch.unbind(x, -1)
    mdims = 1 if normalize_over_channel else 2

    sqz_mdims = [-1] * (x.ndims - mdims)
    dim_prod = np.prod(mag.shape[mdims:])

    mag_s = mag.reshape((mag.shape[0], dim_prod))
    norm_mag = (mag - unsqueeze_n(mag_s.mean(-1), sqz_mdims)) / unsqueeze_n(mag_s.std(-1), sqz_mdims) + shift_mean
    x = torch.stack([norm_mag, phase], dim=x.ndims - 1)

    if not polar:
        x = convert_polar_to_cylindrical(x)
    return x


class ComplexBatchNormalize(nn.Module):
    def __init__(self):
        super(ComplexBatchNormalize, self).__init__()

    def forward(self, input):
        return normalize_complex_batch_by_magnitude_only(input)
