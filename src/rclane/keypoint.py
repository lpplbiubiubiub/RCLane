import torch
import mindspore
import mindspore.nn as nn
from mindspore.ops import Exp, TopK, Sort, Stack, Pow, Sqrt, ZerosLike


class SegMapToKeyPoints(nn.Cell):
    def __init__(self, r):
        super().__init__()
        self.r = mindspore.Tensor(r, dtype=mindspore.float32)
        self.exp = Exp()
        self.topk = TopK(sorted=True)
        self.sort = Sort(descending=True)
        self.stack = Stack(axis=-1)
        self.pow = Pow()
        self.sqrt = Sqrt()
        self.triu = nn.Triu()
        self.zeroslike = ZerosLike()

    def tmp_func(self, x):
        return self.exp(-x)

    def seg2keypoint(self, seg_map):
        seg_map_flatten = seg_map.flatten()
        # thresh = seg_map_flatten.topk(10000).values.min()
        thresh, _ = self.topk(seg_map_flatten, 10000)
        thresh = thresh.min()
        row_series, col_series = torch.where(seg_map > thresh)
        scores = seg_map[row_series, col_series]
        # scores, idx = scores.sort(descending=True)
        scores, idx = self.sort(scores)
        # pts_pre = torch.stack([row_series, col_series], dim=-1)
        pts_pre = self.stack(row_series, col_series)
        pts = pts_pre[idx]
        # pts_float = pts.float()
        # dis_mat = (pts_float - pts_float[..., None, :]).pow(2).sum(dim=-1).sqrt()
        pts_float = pts.astype(mindspore.float32)
        dis_mat = self.pow(pts_float - pts_float[..., None, :], 2).sum(-1)
        dis_mat = self.sqrt(dis_mat)
        dis_mat = self.tmp_func(dis_mat)
        # dis_mat.triu_(diagonal=1)
        # keep = dis_mat.max(dim=0)[0] < self.tmp_func(self.r)
        dis_mat = self.triu(dis_mat, 1)
        keep = dis_mat.max(0) < self.tmp_func(self.r)
        pts_ret = pts[keep]
        # ret = torch.zeros_like(seg_map)
        ret = self.zeroslike(seg_map)
        ret[pts_ret[..., 0], pts_ret[..., 1]] = seg_map[pts_ret[..., 0], pts_ret[..., 1]]
        return ret

    def construct(self, seg_map):
        ret = []
        for seg_map_sigmoid_spec in seg_map:
            sparse_mask = self.seg2keypoint(seg_map_sigmoid_spec[..., 1]) > 0
            seg_map_spec_ret = self.zeroslike(seg_map_sigmoid_spec)
            seg_map_spec_ret[sparse_mask] = seg_map_sigmoid_spec[sparse_mask]
            ret.append(seg_map_spec_ret)
        return torch.stack(ret)

