import torch.nn as nn
import torch
import numpy as np
from utils.polarTransforms import *

def magnitude(input):
    if input.ndimension() == 4:
        return (input[:, :, :, 0] ** 2 + input[:, :, :, 1] ** 2) ** (0.5)
    elif input.ndimension() == 5:
        return (input[:, :, :, :, 0] ** 2 + input[:, :, :, :, 1] ** 2) ** (0.5)
    elif input.ndimension() == 6:
        return (input[:, :, :, :, :, 0] ** 2 + input[:, :, :, :, :, 1] ** 2) ** (0.5)


def complexSTD(x):
    '''
    Standard deviation of real and imaginary channels
    STD = sqrt( E{(x-mu)(x-mu)*} ), where * is the complex conjugate,

    - Source: https://en.wikipedia.org/wiki/Variance#Generalizations
    '''
    mu = torch.mean(torch.mean(x, 2, True), 3, True)

    xm = torch.sum(((x - mu) ** 2), 2, True);  # (a+ib)(a-ib)* = a^2 + b^2
    return torch.mean(torch.mean(xm, 2, True), 3, True) ** (0.5)


def normalizeComplexBatch(x):
    ''' normalize real and imaginary channels'''
    return (x - torch.mean(torch.mean(x, 2, True), 3, True)) / complexSTD(x)


def normalizeComplexBatch_byMagnitudeOnly(x, polar=False):
    ''' normalize the complex batch by making the magnitude of mean 1 and std 1, and keep the phase as it is'''

    ndims = x.ndimension()
    shift_mean = 1
    if not polar:
        x = cylindricalToPolarConversion(x)

    if ndims == 4:
        mag = x[:, :, :, 0]
        mdims = mag.ndimension()
        mag_shaped = mag.reshape((mag.shape[0], mag.shape[1], mag.shape[2]))
        normalized_mag = (mag - torch.mean(mag_shaped, mdims - 1, keepdim=True).unsqueeze(mdims)) / torch.std(
            mag_shaped, mdims - 1, keepdim=True).unsqueeze(mdims) + shift_mean
        x = torch.stack([normalized_mag, x[:, :, :, :, 1]], dim=3)

    elif ndims == 5:
        mag = x[:, :, :, :, 0]
        mdims = mag.ndimension()
        mag_shaped = mag.reshape((mag.shape[0], mag.shape[1], mag.shape[2] * mag.shape[3]))
        normalized_mag = (mag - torch.mean(mag_shaped, mdims - 2, keepdim=True).unsqueeze(mdims - 1)) / torch.std(
            mag_shaped, mdims - 2, keepdim=True).unsqueeze(mdims - 1) + shift_mean
        x = torch.stack([normalized_mag, x[:, :, :, :, 1]], dim=4)

    elif ndims == 6:
        mag = x[:, :, :, :, :, 0];
        mdims = mag.ndimension()
        mag_shaped = mag.reshape((mag.shape[0], mag.shape[1], mag.shape[2] * mag.shape[3] * mag.shape[4]))
        normalized_mag = (mag - torch.mean(mag_shaped, mdims - 3, keepdim=True).unsqueeze(mdims - 2)) / torch.std(
            mag_shaped, mdims - 3, keepdim=True).unsqueeze(mdims - 2) + shift_mean
        x = torch.stack([normalized_mag, x[:, :, :, :, :, 1]], dim=5)

    x[x.ne(x)] = 0.0
    if not polar:
        x = polarToCylindricalConversion(x)
    return x


class ComplexBatchNormalize(nn.Module):

    def __init__(self):
        super(ComplexBatchNormalize, self).__init__()

    def forward(self, input):
        return normalizeComplexBatch_byMagnitudeOnly(input)


