from typing import Dict, List, Tuple

import torch.nn as nn
from torch import Tensor
from torch.nn import MaxPool2d, MaxPool3d
from unets.conv import ConvUnit

class EncodeBlock(nn.Module):
    """The downward / encoding layers of the U-Net"""

    def __init__(
        self,
        features_out: int,
        depth: int,
        kernel_size: int = 3,
        is_input: bool = False,
        in_channels: int = 1,
        is_2d: bool = False,
    ):
        super().__init__()
        inch, ouch = self._in_out_channels(depth, features_out)
        in0 = in_channels if is_input else inch[0]
        MaxPool = MaxPool2d if is_2d else MaxPool3d
        self.conv0 = ConvUnit(
            in_channels=in0, out_channels=ouch[0], kernel_size=kernel_size, is_2d=is_2d
        )
        self.conv1 = ConvUnit(
            in_channels=inch[1], out_channels=ouch[1], kernel_size=kernel_size, is_2d=is_2d
        )
        self.pool = MaxPool(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv0(x)
        x = self.conv1(x)
        skip = x
        x = self.pool(x)  # type: ignore
        return x, skip

    @staticmethod
    def _in_out_channels(
        depth: int, features_out: int = 32
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Abstract counting logic. Returns dicts in_ch, out_ch."""
        if depth == 0:  # input layer
            in0 = 1
            in1 = out0 = features_out
        else:
            in1 = out0 = in0 = features_out * 2 ** (depth)
        out1 = in1 * 2
        return {0: in0, 1: in1}, {0: out0, 1: out1}


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features_out: int = 8,
        depth: int = 3,
        kernel_size: int = 3,
        is_2d: bool = False,
    ):
        """Build the encoding side (downward convolution portion, left side of the U).

        Parameters
        ----------
        features_out: int
            The number of features to extract in the very first convolution
            layer (e.g. size of the very first filter bank in the first
            convolutional block).

        depth: int
            How deep the U-Net should be, not including the bottom layer (e.g.
            layer without skip connections). The classic Ronnberger 3D U-Net
            thus has depth 3.

        kernel_size: int
            Size of kernel in double convolutional blocks.
        """
        super().__init__()
        self.depth_ = depth
        self.features_out = f = features_out
        self.blocks = nn.ModuleList()

        for d in range(depth):
            self.blocks.append(
                EncodeBlock(
                    features_out=f,
                    depth=d,
                    kernel_size=kernel_size,
                    is_input=(d == 0),
                    in_channels=in_channels,  # ignored if d != 0
                    is_2d=is_2d,
                )
            )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        skips: List[Tensor] = []
        for i, encode in enumerate(self.blocks):
            x, skip = encode(x)
            skips.append(skip)
        skips.reverse()
        return x, skips
