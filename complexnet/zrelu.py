
import numpy as np
import torch
import torch.nn as nn
from utils.polar_transforms import *


class ZReLU(nn.Module):
    '''TODO: This module needs to be tested'''
    def __init__(self, polar=False):
        super(ZReLU, self).__init__()
        self.polar = polar

    def forward(self, input):

        ndims = input.ndimension()
        input_real = input.narrow(ndims - 1, 0, 1).squeeze(ndims - 1)
        input_imag = input.narrow(ndims - 1, 1, 1).squeeze(ndims - 1)

        if not self.polar:
            mag, phase = convert_cylindrical_to_polar(input_real, input_imag)
        else:
            phase = input_imag

        phase = phase.unsqueeze(-1)
        phase = torch.cat([phase, phase], ndims - 1)

        output = torch.where(phase >= 0.0, input, torch.tensor(0.0).to(input.device))
        output = torch.where(phase <= np.pi / 2, output, torch.tensor(0.0).to(input.device))

        return output
