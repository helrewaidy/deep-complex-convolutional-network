import torch
from typing import Tuple


def convert_polar_to_cylindrical(
    magnitude: torch.Tensor,
    phase: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert the polar representation (i.e. magnitude and phase) to 
        cylindrical representation (i.e. real and imaginary)

    Parameters
    ----------
    magnitude : torch.Tensor
        The magnitude of the complex tensor
    phase : torch.Tensor
        The phase of the complex tensor

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The real and imaginary part of the complex tensor
    """
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    return real, imag


def convert_cylindrical_to_polar(
    real: torch.Tensor,
    imag: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert the cylindrical representation (i.e. real and imaginary) to
        polar representation (i.e. magnitude and phase)

    Parameters
    ----------
    real : torch.Tensor
        The real part of the complex tensor
    imag : torch.Tensor
        The imaginary part of the complex tensor
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The magnitude and phase of the complex tensor
    """
    mag = (real ** 2 + imag ** 2) ** (0.5)
    phase = torch.atan2(imag, real)
    phase[phase.ne(phase)] = 0.0  # remove NANs if any
    return mag, phase

