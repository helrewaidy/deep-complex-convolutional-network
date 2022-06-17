import torch
import torch.nn as nn


class ComplexDropout(nn.Module):

    def __init__(self, rank, p=0.5, inplace=True):
        super(ComplexDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.rank = rank
        self.p = p
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.p, inplace_str)

    def forward(self, input):
        if not self.training or self.p == 0:
            return input

        if self.p == 1:
            return torch.FloatTensor(input.shape).to(input.device).zero_()

        msk = torch.FloatTensor(input.shape[:-1]).to(input.device).uniform_() > self.p
        msk = torch.stack([msk, msk], dim=-1)

        output = input * msk.to(torch.float32)

        return output


class ComplexDropout1d(ComplexDropout):
    r"""Randomly zeroes whole channels of the complex input tensor.
        The channels to zero are randomized on every forward call.
        Usually the input comes from :class:`nn.Conv3d` modules.
        Args:
            p (float, optional): probability of an element to be zeroed.
            inplace (bool, optional): If set to ``True``, will do this operation
                in-place
        Shape:
            - Input: :math:`(N, C, D, H, W, 2)`
            - Output: :math:`(N, C, D, H, W, 2)` (same shape as input)
    """
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout1d, self).__init__(
            rank=1,
            p=p,
            inplace=inplace
        )


class ComplexDropout2d(ComplexDropout):
    r"""Randomly zeroes whole channels of the complex input tensor.
    The channels to zero-out are randomized on every forward call.
    Usually the input comes from :class:`nn.Conv2d` modules.
    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place
    Shape:
        - Input: :math:`(N, C, H, W, 2)`
        - Output: :math:`(N, C, H, W, 2)` (same shape as input)

    """
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout2d, self).__init__(
            rank=2,
            p=p,
            inplace=inplace
        )


class ComplexDropout3d(ComplexDropout):
    r"""Randomly zeroes whole channels of the complex input tensor.
    The channels to zero are randomized on every forward call.
    Usually the input comes from :class:`nn.Conv3d` modules.
    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place
    Shape:
        - Input: :math:`(N, C, D, H, W, 2)`
        - Output: :math:`(N, C, D, H, W, 2)` (same shape as input)

    """
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout3d, self).__init__(
            rank=3,
            p=p,
            inplace=inplace
        )


if __name__ == '__main__':
    x = torch.rand((2, 2, 8, 8, 2))
    print(f"non-zero elements: {len(x.nonzero())}")
    dropout = ComplexDropout2d(p=0.5)
    y = dropout(x)
    print(f"non-zero elements: {len(y.nonzero())}")
