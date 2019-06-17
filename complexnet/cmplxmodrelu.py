from torch.nn.parameter import Parameter

import torch.nn as nn
from utils.cmplxBatchNorm import magnitude
from utils.saveNet import *


class ModReLU(nn.Module):
    def __init__(self, in_channels, inplace=True):
        super(ModReLU, self).__init__()
        self.inplace = inplace
        self.in_channels = in_channels
        self.b = Parameter(torch.Tensor(in_channels), requires_grad=True)
        self.reset_parameters()
        self.relu = nn.ReLU(self.inplace)

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.0)

    def forward(self, input):
        eps = 1e-5;
        ndims = input.ndimension()
        mag = magnitude(input).unsqueeze(-1) + eps
        mag = torch.cat([mag, mag], ndims - 1)
        if ndims == 4:
            brdcst_b = self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(mag)
        elif ndims == 5:
            brdcst_b = self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(mag)
        elif ndims == 6:
            brdcst_b = self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand_as(mag)

        output = torch.where((mag + brdcst_b) < 0.3, torch.tensor(0.0).to(input.device), input)
        return output


