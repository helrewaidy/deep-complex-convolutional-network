import torch


def convert_polar_to_cylindrical(x1, x2=None):
    '''
    converts the polar representation (i.e. magnitude and phase) of the complex tensor x1 ( or tensors x1 and x2)
    to cylindrical representation (i.e. real and imaginary)
    :param:
        x1: is a tensor contains both magnitude and phase channels in the last dims if x2=None;
        or contains only magnitude part if x2 contains phase component.
        x2: is a tensor similar to x2 or None
     '''

    if x2 is None:  # x1 contains both magnitude and phase components
        ndims = x1.ndimension()
        mag_input = x1.narrow(ndims - 1, 0, 1).squeeze(ndims - 1)
        phase_input = x1.narrow(ndims - 1, 1, 1).squeeze(ndims - 1)

        real = mag_input * torch.cos(phase_input)
        imag = mag_input * torch.sin(phase_input)
        return torch.stack((real, imag), dim=x1.ndimension() - 1)

    else:  # x1 contains magnitude component and x2 contains phase component
        real = x1 * torch.cos(x2)
        imag = x1 * torch.sin(x2)
        return real, imag


def convert_cylindrical_to_polar(x1, x2=None):
    '''
    converts the cylindrical representation (i.e. real and imaginary) of the complex tensor x1 ( or tensors x1 and x2)
    to polar representation (i.e. magnitude and phase)
    :param:
        x1: is a tensor contains both real and imaginary channels in the last dims if x2=None;
        or contains only real part if x2 contains imaginary component.
        x2: is a tensor similar to x2 or None
     '''

    if x2 is None:  # x1 contains both real and imaginary components
        ndims = x1.ndimension()
        real = x1.narrow(ndims - 1, 0, 1).squeeze(ndims - 1)
        imag = x1.narrow(ndims - 1, 1, 1).squeeze(ndims - 1)

        mag = (real ** 2 + imag ** 2) ** (0.5)
        phase = torch.atan2(imag, real)
        phase[phase.ne(phase)] = 0.0  # remove NANs

        return torch.stack((mag, phase), dim=x1.ndimension() - 1)

    else:  # x1 contains real component and x2 contains imaginary component
        mag = (x1 ** 2 + x2 ** 2) ** (0.5)
        phase = torch.atan2(x2, x1)

        phase[phase.ne(phase)] = 0.0  # remove NANs
        return mag, phase
