# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""rclane head."""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import context
import mindspore.common.dtype as mstype
from src.rclane.layers import DoubleConv, OutConv

context.set_context(mode=context.PYNATIVE_MODE)

class MLP(nn.Cell):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Dense(input_dim, embed_dim)

    def construct(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(0, 2, 1)
        x = self.proj(x)
        return x


class RCLaneHead(nn.Cell):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, embedding_dim, middle_dim,
                 input_channels=[64, 128, 320, 512], bilinear=True):
        super(RCLaneHead, self).__init__()
        self.feature_strides = feature_strides
        self.in_channels = input_channels
        self.in_index = [0, 1, 2, 3]
        self.middle_dim = middle_dim
        self.resize = P.ResizeBilinear((80, 200), align_corners=False)
        self.out_resize = P.ResizeBilinear((320, 800), align_corners=False)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.SequentialCell(
            nn.Conv2d(embedding_dim*4, embedding_dim, kernel_size=1, stride=1, has_bias=False),
            nn.BatchNorm2d(embedding_dim)
        )
        self.dropout = nn.Dropout(1 - 0.1)
        self.linear_pred = nn.Conv2d(embedding_dim, self.middle_dim, kernel_size=1, stride=1, has_bias=False)

        self.out_seg = nn.SequentialCell(DoubleConv(middle_dim, middle_dim), OutConv(middle_dim, 2))
        self.up_arrow = nn.SequentialCell(DoubleConv(middle_dim, middle_dim), OutConv(middle_dim, 2))
        self.down_arrow = nn.SequentialCell(DoubleConv(middle_dim, middle_dim), OutConv(middle_dim, 2))
        self.up_bound = nn.SequentialCell(DoubleConv(middle_dim, middle_dim), OutConv(middle_dim, 2))
        self.down_bound = nn.SequentialCell(DoubleConv(middle_dim, middle_dim), OutConv(middle_dim, 2))

        self.cat = P.Concat(1)


    def transform_input(self, inputs):
        inputs = [inputs[i] for i in self.in_index]
        return inputs

    def construct(self, inputs):
        x = self.transform_input(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).transpose(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = self.resize(_c4)
        _c3 = self.linear_c3(c3).transpose(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = self.resize(_c3)
        _c2 = self.linear_c2(c2).transpose(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = self.resize(_c2)
        _c1 = self.linear_c1(c1).transpose(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c = self.linear_fuse(self.cat((_c4, _c3, _c2, _c1)))
        _c = self.linear_pred(_c)
        _x = self.dropout(_c)

        seg = self.out_seg(_x)
        up_arrow = self.up_arrow(_x)
        down_arrow = self.down_arrow(_x)
        up_bound = self.up_bound(_x)
        down_bound = self.down_bound(_x)

        seg = self.out_resize(seg)
        up_arrow = self.out_resize(up_arrow)
        down_arrow = self.out_resize(down_arrow)
        up_bound = self.out_resize(up_bound)
        down_bound = self.out_resize(down_bound)

        ret = dict(
            seg_map=seg,
            up_arrow=up_arrow,
            down_arrow=down_arrow,
            up_bound=up_bound,
            down_bound=down_bound,
        )
        return ret