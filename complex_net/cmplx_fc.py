import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.real_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.imag_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, input):
        real, imag = torch.unbind(input, dim=-1)

        real_out = self.real_linear(real) - self.imag_linear(imag)
        imag_out = self.real_linear(imag) + self.imag_linear(real)

        output = torch.stack((real_out, imag_out), dim=-1)

        return output
