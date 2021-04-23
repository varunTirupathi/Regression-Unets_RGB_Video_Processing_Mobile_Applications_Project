import torch as t
import torch.nn as nn
from torch import Tensor
from torch.nn import Conv2d, Conv3d
from unets.conv import ConvUnit
from unets.decode import Decoder
from unets.encode import Encoder


class JoinBlock(nn.Module):
    """The bottom block of the U-Net, which performs no real down- or up-sampling

    Parameters
    ----------
    encoder: Encoder
        The Encoder module to be joined to a Decoder module.

    kernel_size: int
        The size of the kernels in the internal double convolution blocks /
        units.
    """

    def __init__(self, encoder: Encoder, kernel_size: int = 3, is_2d: bool = False):
        super().__init__()
        d, f = encoder.depth_, encoder.features_out
        in1 = out0 = in0 = f * (2 ** d)
        out1 = 2 * in1
        self.conv0 = ConvUnit(in0, out0, kernel_size=kernel_size, is_2d=is_2d)
        self.conv1 = ConvUnit(in1, out1, kernel_size=kernel_size, is_2d=is_2d)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        return x


class UNet3d(nn.Module):
    """Build the classic Ronnberger 3D U-Net.

    Parameters
    ----------
    initial_features: int
        The number of channels / filters / features to extract in the very first
        convolution. Sets the resulting sizes for the entire net.

    depth: int
        Encoder / decoder depth. E.g. a depth of 3 will results in 3 encoding
        blocks (two convolution layers + downsample), one joining bottom later, and 3
        decoding blocks (upsample + two convolution layers).
    """

    def __init__(
        self,
        in_channels: int = 1,
        initial_features: int = 8,
        n_labels: int = 2,
        depth: int = 3,
        kernel_size: int = 3,
        is_2d: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            features_out=initial_features,
            depth=depth,
            kernel_size=kernel_size,
            is_2d=is_2d,
        )
        self.joiner = JoinBlock(self.encoder, kernel_size=kernel_size, is_2d=is_2d)
        self.decoder = Decoder(self.encoder, kernel_size=kernel_size, is_2d=is_2d)
        Conv = Conv2d if is_2d else Conv3d
        self.segmenter = Conv(
            in_channels=2 * initial_features, out_channels=n_labels, kernel_size=1, stride=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x, skips = self.encoder(x)
        x = self.joiner(x)
        x = self.decoder(x, skips)
        x = self.segmenter(x)
        return x


class UNet3d2d(nn.Module):
    """Build a U-Net with 3D encoder and 2D decoder. UsageL

    ```python
    from unets.unet import UNet3d2D

    unet = UNet3d2d()
    ```

    Parameters
    ----------
    initial_features: int
        The number of channels / filters / features to extract in the very first
        convolution. Sets the resulting sizes for the entire net.

    depth: int
        Encoder / decoder depth. E.g. a depth of 3 will results in 3 encoding
        blocks (two convolution layers + downsample), one joining bottom later, and 3
        decoding blocks (upsample + two convolution layers).
    """

    def __init__(
        self,
        in_channels: int = 1,
        initial_features: int = 8,
        n_labels: int = 2,
        depth: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            features_out=initial_features,
            depth=depth,
            kernel_size=kernel_size,
            is_2d=False,
        )
        self.joiner = JoinBlock(self.encoder, kernel_size=kernel_size, is_2d=False)
        self.decoder = Decoder(self.encoder, kernel_size=kernel_size, is_2d=True)
        self.segmenter = Conv2d(
            in_channels=2 * initial_features, out_channels=n_labels, kernel_size=1, stride=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x, skips = self.encoder(x)
        skips = list(map(lambda skip: t.mean(skip, dim=-1), skips))  # flatten skips for later
        x = self.joiner(x)
        x = t.mean(x, dim=-1)  # assuming time in last dimension
        x = self.decoder(x, skips)
        x = self.segmenter(x)
        return x
