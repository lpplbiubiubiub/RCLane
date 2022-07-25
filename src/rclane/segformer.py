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
"""pvt backbone."""

import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, context
import mindspore.common.dtype as mstype
from src.rclane.layers import to_2tuple, DropPath
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import TruncatedNormal, Normal, Zero, One, initializer
import math

ms_cast_type = mstype.float32

context.set_context(mode=context.PYNATIVE_MODE)

def conv2d_weight_init(kernel_size0, kernel_size1, dim):
    fan_out = kernel_size0 * kernel_size1 * dim
    fan_out //= dim
    weights = Normal(math.sqrt(2.0 / fan_out), 0)
    return weights

class DWConv(nn.Cell):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, pad_mode='pad',
                                weight_init=conv2d_weight_init(3, 3, dim), has_bias=True, bias_init=Zero(),
                                group=dim).to_float(ms_cast_type)

    def construct(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(0, 2, 1).reshape((B, C, H, W))
        x = self.dwconv(x)
        x = x.reshape(B, C, H*W).transpose(0, 2, 1)
        return x

class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, act_layer=P.GeLU(), out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features, weight_init=TruncatedNormal(0.02), bias_init=Zero())
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer
        self.fc2 = nn.Dense(hidden_features, out_features, weight_init=TruncatedNormal(0.02), bias_init=Zero())
        self.drop = nn.Dropout(1 - drop)

    def construct(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias, weight_init=TruncatedNormal(0.02))
        self.kv = nn.Dense(dim, dim*2, has_bias=qkv_bias, weight_init=TruncatedNormal(0.02))
        self.attn_drop = nn.Dropout(1 - attn_drop)
        self.proj = nn.Dense(dim, dim, weight_init=TruncatedNormal(0.02))
        self.proj_drop = nn.Dropout(1 - proj_drop)

        self.unstack = P.Unstack(0)
        self.q_matmul_k = P.BatchMatMul(transpose_b=True)
        self.activation = nn.Softmax()
        self.attn_matmul_v = P.BatchMatMul()

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, padding=1, pad_mode='pad',
                                weight_init=conv2d_weight_init(sr_ratio, sr_ratio, dim))
            self.norm = nn.LayerNorm((dim,), gamma_init=One(), beta_init=Zero())

    def construct(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        k, v = self.unstack(kv)

        attn = self.q_matmul_k(q, k) * self.scale
        attn = self.activation(attn)
        attn = self.attn_drop(attn)

        x = self.attn_matmul_v(attn, v)
        x = x.transpose(0, 2, 1, 3).reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=P.GeLU(), sr_ratio=1):
        super().__init__()
        self.drop_prob = drop_path
        self.norm1 = nn.LayerNorm((dim,))
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        self.norm2 = nn.LayerNorm((dim,), gamma_init=One(), beta_init=Zero())
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def construct(self, x, H, W):
        if self.drop_prob > 0:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        else:
            x = x + self.attn(self.norm1(x), H, W)
            x = x + self.mlp(self.norm2(x), H, W)
        return x

class OverlapPatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[0] // 2, patch_size[1] // 2, patch_size[1] // 2), pad_mode='pad',
                              weight_init=conv2d_weight_init(patch_size[0], patch_size[1], embed_dim),
                              has_bias=False)
        self.norm = nn.LayerNorm((embed_dim,), gamma_init=One(), beta_init=Zero())

    def construct(self, x):
        x = self.proj(x)
        b, c, H, W = x.shape
        x = x.reshape(b, c, H*W).transpose(0, 2, 1)
        x = self.norm(x)

        return x, H, W

class MixVisionTransformer(nn.Cell):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.pos_embed1 = Parameter(initializer(TruncatedNormal(0.02), (1, 16000, embed_dims[0])))
        self.pos_embed2 = Parameter(initializer(TruncatedNormal(0.02), (1, 4000, embed_dims[1])))
        self.pos_embed3 = Parameter(initializer(TruncatedNormal(0.02), (1, 1000, embed_dims[2])))
        self.pos_embed4 = Parameter(initializer(TruncatedNormal(0.02), (1, 250, embed_dims[3])))


        dpr = [x.item() for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.SequentialCell([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = nn.LayerNorm((embed_dims[0],), gamma_init=One(), beta_init=Zero())

        cur += depths[0]
        self.block2 = nn.SequentialCell([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[0])
            for i in range(depths[1])])
        self.norm2 = nn.LayerNorm((embed_dims[1],), gamma_init=One(), beta_init=Zero())

        cur += depths[1]
        self.block3 = nn.SequentialCell([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = nn.LayerNorm((embed_dims[2],), gamma_init=One(), beta_init=Zero())

        cur += depths[2]
        self.block4 = nn.SequentialCell([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = nn.LayerNorm((embed_dims[3],), gamma_init=One(), beta_init=Zero())

    def reset_drop_path(self, drop_path_rate):
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        x = x + self.pos_embed1
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).transpose(0, 3, 1, 2)
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        x = x + self.pos_embed2
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).transpose(0, 3, 1, 2)
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        x = x + self.pos_embed3
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).transpose(0, 3, 1, 2)
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        x = x + self.pos_embed4
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).transpose(0, 3, 1, 2)
        outs.append(x)

        return outs

    def construct(self, x):
        x = self.forward_features(x)
        return x


class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

def build_segformer_backbone(vision):
    if vision == 'b0':
        model = mit_b0()
    elif vision == 'b1':
        model = mit_b1()
    elif vision == 'b2':
        model = mit_b2()
    return model

