from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ConvTranspose2d as UpConv2d
from torch.nn import ConvTranspose3d as UpConv3d

from unets.conv import ConvUnit
from unets.encode import Encoder


class DecodeBlock(nn.Module):
    """The upsampling layer, followed by two ConvUnits, the first of which takes
    the skip connection.

    Note that almost all the params of the decoder are dependent on those chosen for
    the Encoder. We can use this to our advantage in construction."""

    def __init__(self, encoder: Encoder, depth: int, kernel_size: int = 3, is_2d: bool = False):
        super().__init__()
        inch, ouch, upch = self._in_out_channels(depth, encoder.features_out)
        UpConv = UpConv2d if is_2d else UpConv3d
        self.upconv = UpConv(in_channels=upch, out_channels=upch, kernel_size=2, stride=2)
        self.conv0 = ConvUnit(
            in_channels=inch[0], out_channels=ouch[0], kernel_size=kernel_size, is_2d=is_2d
        )
        self.conv1 = ConvUnit(
            in_channels=inch[1], out_channels=ouch[1], kernel_size=kernel_size, is_2d=is_2d
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upconv(x)
        # remember, batch tensors are shape [B_size, Ch_size, <InputDims>]
        cropped = self.crop(x, skip)
        x = torch.cat([cropped, x], dim=1)  # concat along channel dimension
        x = self.conv0(x)
        x = self.conv1(x)
        return x

    @staticmethod
    def crop(x: Tensor, skip: Tensor) -> Tensor:
        # (left, right, top, bottom, front, back) is F.pad order
        # we are just implementing our own version of
        # https://github.com/fepegar/unet/blob/9f64483d351b4f7d95c0d871aa7aa587b8fdb21b/unet/decoding.py#L142
        # but fixing their bug which won't work for odd numbers
        shape_diffs = torch.tensor(skip.shape)[2:] - torch.tensor(x.shape)[2:]
        halfs = torch.true_divide(shape_diffs, 2)
        halfs_left = -torch.floor(halfs).to(dtype=int)  # type: ignore
        halfs_right = -torch.ceil(halfs).to(dtype=int)  # type: ignore
        pads = torch.stack([halfs_left, halfs_right]).t().flatten().tolist()
        cropped = F.pad(skip, pads)
        return cropped

    @staticmethod
    def _in_out_channels(
        depth: int, features_out: int = 32
    ) -> Tuple[Dict[int, int], Dict[int, int], int]:
        """Abstract counting logic. Returns dicts in_ch, out_ch. Assumes an incoming skip connection.

        Parameters
        ----------
        depth: int
            depth == 0 is the last decoding block (e.g. top right of U-Net)

        features_out: int
            The amount of initial ouput features in the first convolution of the
            encoder.
        """
        f = features_out
        skip = f * 2 ** (depth + 1)  # number of in_channels / features of the skip connection
        in0 = 3 * skip  # first in is always skip + 2*skip channels
        out1 = in1 = out0 = skip
        up = 2 * skip
        return {0: in0, 1: in1}, {0: out0, 1: out1}, up


class Decoder(nn.Module):
    def __init__(self, encoder: Encoder, kernel_size: int = 3, is_2d: bool = True):
        """Abstract counting logic. Returns dicts in_ch, out_ch. Assumes an incoming skip connection.

        Parameters
        ----------
        encoder: Encoder
            The Encoder to be decoded.

        kernel_size: int
            The size of the kernel for the internal convolutional units.

        normaliation: bool
            If True (default) apply GroupNormalization3D in convolution units.

        depth: int
            depth == 0 is the last decoding block (e.g. top right of U-Net)
        """
        super().__init__()
        self.depth_ = encoder.depth_
        self.features_out = encoder.features_out

        self.blocks = nn.ModuleList()
        for depth in reversed(range(self.depth_)):
            self.blocks.append(DecodeBlock(encoder, depth, kernel_size, is_2d))

    def forward(self, x: Tensor, skips: List[Tensor]) -> Tensor:
        for skip, decode in zip(skips, self.blocks):
            x = decode(x, skip)
        return x
