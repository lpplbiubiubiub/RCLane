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
import mindspore
import mindspore.nn as nn
from mindspore.ops import TopK, LogSoftmax, clip_by_value, Cast
from mindspore import Tensor

def find_k_th_small_in_a_tensor(target_tensor, k_th):
    topk = TopK(sorted=True)
    val, idxes = topk(-target_tensor, k_th)
    return -val[-1]

class RCLaneLoss(nn.Cell):
    def __init__(self):
        super(RCLaneLoss, self).__init__()
        self.ALPHA = 1
        self.NEGATIVE_RATIO = 15
        self.smooth_l1 = nn.SmoothL1Loss()
        self.cast = Cast()

    def construct(self, rc_predict, gt_spec):
        gt_seg_map = gt_spec['seg_map']
        gt_up_arrow = gt_spec['up_arrow']
        gt_down_arrow = gt_spec['down_arrow']
        gt_up_bound = gt_spec['up_bound']
        gt_down_bound = gt_spec['down_bound']

        pr_seg_map = rc_predict['seg_map']
        pr_up_arrow = rc_predict['up_arrow']
        pr_down_arrow = rc_predict['down_arrow']
        pr_up_bound = rc_predict['up_bound']
        pr_down_bound = rc_predict['down_bound']

        pmask = gt_seg_map[..., 1]
        nmask = gt_seg_map[..., 0]
        fpmask = pmask.float()
        fnmask = nmask.float()
        bg_fg_predict = LogSoftmax(pr_seg_map, dim=-1)

        total_postive_num = fpmask.sum()
        total_negative_num = fnmask.sum()
        negative_num = clip_by_value(total_postive_num * self.NEGATIVE_RATIO,
                                     clip_value_min=Tensor(1, mindspore.float32),
                                     clip_value_max=total_negative_num).astype('int32')
        positive_num = clip_by_value(total_postive_num, clip_value_min=Tensor(1, mindspore.float32)).astype('int32')

        fg_predict = bg_fg_predict[..., 1]
        bg_predict = bg_fg_predict[..., 0]
        max_hard_pred = find_k_th_small_in_a_tensor(bg_predict[nmask].detach(), negative_num)
        fnmask_ohem = self.cast(bg_predict <= max_hard_pred, mindspore.float32) * fnmask
        total_cross_pos = -(self.ALPHA * fg_predict * fpmask).sum() / positive_num
        total_cross_neg = -(self.ALPHA * bg_predict * fnmask_ohem).sum() / positive_num

        up_arrow_loss = self.smooth_l1(pr_up_arrow[pmask], gt_up_arrow[pmask]).sum() / positive_num
        down_arrow_loss = self.smooth_l1(pr_down_arrow[pmask], gt_down_arrow[pmask]).sum() / positive_num
        up_bound_loss = self.smooth_l1(pr_up_bound[pmask], gt_up_bound[pmask]).sum() / positive_num
        down_bound_loss = self.smooth_l1(pr_down_bound[pmask], gt_down_bound[pmask]).sum() / positive_num

        return dict(total_cross_pos=total_cross_pos,
                    total_cross_neg=total_cross_neg,
                    up_arrow_loss=up_arrow_loss,
                    down_arrow_loss=down_arrow_loss,
                    up_bound_loss=up_bound_loss,
                    down_bound_loss=down_bound_loss)