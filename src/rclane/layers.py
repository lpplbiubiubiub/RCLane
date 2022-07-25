import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
import collections.abc
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob

        self.shape = P.Shape()
        self.ones = P.Ones()
        self.dropout = P.Dropout(self.keep_prob)

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)  # B N C
            mask = self.ones((x_shape[0], 1, 1), mindspore.float32)
            x = self.dropout(mask) * x
        return x

class DoubleConv(nn.Cell):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.SequentialCell(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def construct(self, x):
        return self.double_conv(x)

class OutConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              padding=0, pad_mode='pad', has_bias=False)
    def construct(self, x):
        return self.conv(x)