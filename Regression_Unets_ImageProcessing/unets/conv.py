import torch.nn as nn
from torch import Tensor
from torch.nn import Conv2d, Conv3d, GroupNorm, ReLU


#Code adapted from https://github.com/fepegar/unet/blob/v0.7.5/unet/conv.py
class ConvUnit(nn.Module):
    """A combination convolution, (group) normalization, and activation layer"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, is_2d: bool = False
    ):
        super().__init__()
        Conv = Conv2d if is_2d else Conv3d
        self.conv = Conv(in_channels, out_channels, kernel_size, padding=(kernel_size + 1) // 2 - 1)
        self.gnorm = GroupNorm(num_groups=1, num_channels=out_channels)
        self.act = ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.gnorm(x)
        x = self.act(x)
        return x
