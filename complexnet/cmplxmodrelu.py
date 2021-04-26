import torch.nn as nn
from torch.nn.parameter import Parameter
from utils.cmplx_batchnorm import magnitude
from utils.save_net import *
import torch

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

    def _unsqueeze_at_indices(self, x, indices=[]):
        for i in indices:
            x = x.unsqueeze(i)
        return x

    def forward(self, input):
        eps = 1e-5;
        ndims = input.ndimension()
        mag = magnitude(input).unsqueeze(-1) + eps
        mag = torch.cat([mag, mag], ndims - 1)
        sqz_idx = [0]+list(range(2, ndims))
        brdcst_b = self._unsqueeze_at_indices(self.b, sqz_idx).expand_as(mag)

        output = torch.where((mag + brdcst_b) < 0.3, torch.tensor(0.0).to(input.device), input)
        return output
