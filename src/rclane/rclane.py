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

import numpy as np
import mindspore.nn as nn
from mindspore.ops import Softmax
from mindspore import Tensor, context
import mindspore.common.dtype as mstype
from src.rclane.layers import DoubleConv, OutConv
from src.rclane.segformer import build_segformer_backbone
from src.rclane.rclane_head import RCLaneHead
from src.rclane.rclane_loss import RCLaneLoss

context.set_context(mode=context.PYNATIVE_MODE)

class RCLane(nn.Cell):
    def __init__(self, vision, embedding_dim, middle_dim):
        super(RCLane, self).__init__()
        self.vision = vision
        self.embedding_dim = embedding_dim
        self.middle_dim = middle_dim
        self.backbone = build_segformer_backbone(self.vision)
        if self.vision == 'b0':
            input_channels = [32, 64, 160, 256]
        else:
            input_channels = [64, 128, 320, 512]
        self.rc_head = RCLaneHead(feature_strides=[4, 8, 16, 32], input_channels=input_channels,
                                  embedding_dim=self.embedding_dim, middle_dim=self.middle_dim)

        self.loss = RCLaneLoss()
        self.seg2keypoint = SegMapToKeyPoints(2)
        self.seg_softmax = Softmax(-1)

    def construct(self, inputs, gt_data):
        feature_maps = self.backbone(inputs, gt_data)
        result = self.rc_head(feature_maps)
        result = {k: v.transpose(0, 2, 3, 1) for k, v in result.items()}

        if self.training:
            rc_loss = self.loss(result, gt_data)
            return rc_loss

