import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from utils.polar_transforms import (
    convert_polar_to_cylindrical,
    convert_cylindrical_to_polar
)


class CReLU(nn.ReLU):
    def __init__(self, inplace: bool=False):
        super(CReLU, self).__init__(inplace)


class ModReLU(nn.Module):
    def __init__(self, in_channels, inplace=True):
        """ModReLU

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        inplace : bool
            If True, the input is modified.
        """
        super(ModReLU, self).__init__()
        self.inplace = inplace
        self.in_channels = in_channels
        self.b = Parameter(torch.Tensor(in_channels), requires_grad=True)
        self.reset_parameters()
        self.relu = nn.ReLU(self.inplace)

    def reset_parameters(self):
        self.b.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        real, imag = torch.unbind(input, -1)
        mag, phase = convert_cylindrical_to_polar(real, imag)
        brdcst_b = torch.swapaxes(torch.broadcast_to(self.b, mag.shape), -1, 1)
        mag = self.relu(mag + brdcst_b)
        real, imag = convert_polar_to_cylindrical(mag, phase)
        output = torch.stack((real, imag), dim=-1)
        return output


class ZReLU(nn.Module):
    def __init__(self):
        super(ZReLU, self).__init__()

    def forward(self, input):
        real, imag = torch.unbind(input, dim=-1)
        mag, phase = convert_cylindrical_to_polar(real, imag)

        phase = torch.stack([phase, phase], dim=-1)
        output = torch.where(phase >= 0.0, input, torch.tensor(0.0).to(input.device))
        output = torch.where(phase <= np.pi / 2, output, torch.tensor(0.0).to(input.device))

        return output
