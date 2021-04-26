import torch

import torch.nn as nn


class ComplexUpsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear'):
        super(ComplexUpsample, self).__init__()

        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode)

    def forward(self, input):
        ndims = input.ndimension()

        input_real = input.narrow(ndims - 1, 0, 1).squeeze(ndims - 1)
        input_imag = input.narrow(ndims - 1, 1, 1).squeeze(ndims - 1)

        output = torch.stack((self.upsample(input_real), self.upsample(input_imag)), dim=ndims - 1)

        return output
