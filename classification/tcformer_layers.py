import math
import einops
from turtle import forward
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from deform_conv2d import DeformConv2dPack
from transformer_utils import DropPath, to_2tuple, trunc_normal_
from tcformer_utils import (
    merge_tokens, cluster_dpc_knn, token2map,
    map2token, token_downup)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Attention module with spatial reduction layer
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# Sparse Attention
class SparseAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, 1)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.v_relu = nn.ReLU(inplace=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = self.softmax(q)
        qk = torch.mul(k, q)
        qk = torch.sum(qk, dim=-1, keepdim=True)
        v = self.v_relu(v)
        out = torch.mul(v, qk).squeeze(0)
        out = self.out(out)

        return out
# Transformer blocks
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # self.attn = SparseAttention(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# The first conv layer
class OverlapPatchEmbed(nn.Module):
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
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


# depth-wise conv
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


# conv layer for dynamic tokens
class TokenConv(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        groups = kwargs['groups'] if 'groups' in kwargs.keys() else 1
        self.skip = nn.Conv1d(in_channels=kwargs['in_channels'],
                              out_channels=kwargs['out_channels'],
                              kernel_size=1, bias=False,
                              groups=groups)

    def forward(self, token_dict):
        x = token_dict['x']
        x = self.skip(x.permute(0, 2, 1)).permute(0, 2, 1) # (2,64,3136)->(2,128,3136) channel
        x_map = token2map(token_dict)
        x_map = super().forward(x_map) # downsample
        x = x + map2token(x_map, token_dict)
        return x

class TokenConv_map(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        groups = kwargs['groups'] if 'groups' in kwargs.keys() else 1
        self.skip = nn.Conv1d(in_channels=kwargs['in_channels'],
                              out_channels=kwargs['out_channels'],
                              kernel_size=1, bias=False,
                              groups=groups)

    def forward(self, token_dict):
        x = token_dict['x']
        x = self.skip(x.permute(0, 2, 1)).permute(0, 2, 1) # (2,64,3136)->(2,128,3136) channel
        x_map = token2map(token_dict)
        x_map = super().forward(x_map) # downsample
        x = x + map2token(x_map, token_dict)
        return x, x_map.flatten(2).permute(0, 2, 1).contiguous()


class TokenConv_up(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        groups = kwargs['groups'] if 'groups' in kwargs.keys() else 1
        self.skip = nn.Conv1d(in_channels=kwargs['in_channels'],
                              out_channels=kwargs['out_channels'],
                              kernel_size=1, bias=False,
                              groups=groups)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, token_dict):
        x = token_dict['x']
        x = self.skip(x.permute(0, 2, 1)).permute(0, 2, 1) # (2,64,3136)->(2,128,3136) channel
        x_map = token2map(token_dict)
        x_map = super().forward(x_map) # downsample
        x = x + self.upsample(x_map).flatten(2).permute(0, 2, 1).contiguous()
        return x

class TokenConv_change(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        groups = kwargs['groups'] if 'groups' in kwargs.keys() else 1
        self.skip = nn.Conv1d(in_channels=kwargs['in_channels'],
                              out_channels=kwargs['out_channels'],
                              kernel_size=1, bias=False,
                              groups=groups)

    def forward(self, token_dict):
        x = token_dict['x']
        x = self.skip(x.permute(0, 2, 1)).permute(0, 2, 1) # (2,64,3136)->(2,128,3136) channel
        return x

class change_channel(nn.Module):
    def __init__(self, dim=128):
        super(change_channel, self).__init__()
        self.conv = nn.Conv2d(64, dim, 1, 1, 0, bias=False,)
    
    def forward(self, token_dict):
        x = token_dict['x']
        B, N, C = x.shape
        H, W = token_dict['map_size']
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous() # token2map
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2) # map2token

        return x

# class Multiscale(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.skip = nn.Conv1d(in_channels=in_channels,
#                               out_channels=out_channels,
#                               kernel_size=1, bias=False,
#                               groups=1)
#         self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
#                                 kernel_size=3, stride=2 ,padding=1)
#         self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
#                                 kernel_size=5, stride=2, padding=2)
#         self.conv3 = nn.Conv2d(in_channels=1, out_channels=1,
#                                 kernel_size=3, stride=2 ,padding=1)
#         self.conv4 = nn.Conv2d(in_channels=1, out_channels=1,
#                                 kernel_size=5, stride=2, padding=2)

#     def forward(self, x, map_size, token_score):
#         B = x.shape[0]
#         H, W = map_size[0], map_size[1]
#         x = self.skip(x.permute(0, 2, 1)).permute(0, 2, 1) # (2,64,3136)->(2,128,3136) channel
#         token_score = token_score.permute(0, 2, 1)
#         C = x.shape[2]
#         x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
#         token_score = token_score.reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
#         x1 = self.conv1(x)
#         x2 = self.conv2(x)
#         token_score1 = self.conv3(token_score)
#         token_score2 = self.conv4(token_score)
#         x = x1 + x2
#         token_score = token_score1 + token_score2
#         x = x.flatten(2).transpose(1, 2)
#         token_score = token_score.flatten(2).transpose(1, 2)
#         return x, token_score

class Multiscale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1, bias=False,
                              groups=1)
        # self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
        #                         kernel_size=3, stride=2 ,padding=1)
        self.depth_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=5, stride=2, padding=2, groups=out_channels)
        self.point_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=1, stride=1, padding=0, groups=1)


    def forward(self, x, x_3, map_size):
        B = x.shape[0]
        H, W = map_size[0], map_size[1]
        x = self.skip(x.permute(0, 2, 1)).permute(0, 2, 1) # (2,64,3136)->(2,128,3136) channel
        C = x.shape[2]
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.depth_conv(x)
        x = self.point_conv(x)
        # x2 = self.conv2(x)
        x = x.flatten(2).transpose(1, 2)
        return x + x_3



# Mlp for dynamic tokens
class TCMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = TokenConv(in_channels=hidden_features,
                                out_channels=hidden_features,
                                kernel_size=3, padding=1, stride=1,
                                bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, token_dict):
        token_dict['x'] = self.fc1(token_dict['x'])
        x = self.dwconv(token_dict)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Attention for dynamic tokens
class TCAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_sr_layer=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.use_sr_layer = use_sr_layer
        if sr_ratio > 1:
            if self.use_sr_layer:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, q_dict, kv_dict):
        q = q_dict['x']
        kv = kv_dict['x']
        B, Nq, C = q.shape
        Nkv = kv.shape[1]
        conf_kv = kv_dict['token_score'] if 'token_score' in kv_dict.keys() else kv.new_zeros(B, Nkv, 1)

        q = self.q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            tmp = torch.cat([kv, conf_kv], dim=-1)
            tmp_dict = kv_dict.copy()
            tmp_dict['x'] = tmp
            tmp_dict['map_size'] = q_dict['map_size']
            tmp = token2map(tmp_dict)

            kv = tmp[:, :C]
            conf_kv = tmp[:, C:]

            if self.use_sr_layer:
                kv = self.sr(kv)
                _, _, h, w = kv.shape
                kv = kv.reshape(B, C, -1).permute(0, 2, 1).contiguous()
                kv = self.norm(kv)
            else:
                kv = F.avg_pool2d(kv, kernel_size=self.sr_ratio, stride=self.sr_ratio)
                kv = kv.reshape(B, C, -1).permute(0, 2, 1).contiguous()

            conf_kv = F.avg_pool2d(conf_kv, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            conf_kv = conf_kv.reshape(B, 1, -1).permute(0, 2, 1).contiguous()
        if conf_kv.shape[1] == 196:
            conf_kv = conf_kv.reshape(B, 14, 14, 1).permute(0, 3, 1, 2).contiguous()
            conf_kv = F.avg_pool2d(conf_kv, kernel_size=2, stride=2)
            conf_kv = conf_kv.reshape(B, 1, -1).permute(0, 2, 1).contiguous()

        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        conf_kv = conf_kv.squeeze(-1)[:, None, None, :]
        attn = attn + conf_kv
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Transformer block for dynamic tokens
class TCBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, use_sr_layer=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TCAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, use_sr_layer=use_sr_layer)
        # self.attn = SparseAttention_yj(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TCMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            q_dict, kv_dict = inputs
        else:
            q_dict, kv_dict = inputs, None

        x = q_dict['x']
        # norm1
        q_dict['x'] = self.norm1(q_dict['x'])
        if kv_dict is None:
            kv_dict = q_dict
        else:
            kv_dict['x'] = self.norm1(kv_dict['x'])

        # attn
        x = x + self.drop_path(self.attn(q_dict, kv_dict))

        # mlp
        q_dict['x'] = self.norm2(x)
        x = x + self.drop_path(self.mlp(q_dict))
        q_dict['x'] = x

        return q_dict

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0,3,1,2).contiguous()
# CTM block
class CTM(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k=5, groups = 1, off_kernel = 5, offset_range_factor=None):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        # self.conv = TokenConv(in_channels=embed_dim, out_channels=dim_out, kernel_size=3, stride=2, padding=1)
        self.conv = TokenConv_map(in_channels=embed_dim, out_channels=dim_out, kernel_size=3, stride=2, padding=1)
        self.Multiscale = Multiscale(in_channels=embed_dim, out_channels=dim_out)
        self.norm = nn.LayerNorm(self.dim_out)
        self.norm1 = nn.LayerNorm(self.dim_out)
        self.score = nn.Linear(self.dim_out, 1)
        self.k = k
        self.proj_q = nn.Conv2d(
            self.dim_out, self.dim_out,
            kernel_size=1, stride=1, padding=0
        )
        self.groups = groups
        self.offset_range_factor = offset_range_factor


    def forward(self, token_dict):
        token_dict = token_dict.copy()
        x_kv = token_dict['x']
        #----------------------------------------------------
        x, x_3 = self.conv(token_dict)
        x = self.norm(x)
        #----------------------------------------------------
        token_score = self.score(x)
        token_weight = token_score.exp()
        token_dict['token_score'] = token_score
        token_dict['x'] = x
        B, N, C = x.shape
        H, W = token_dict['map_size'][0], token_dict['map_size'][1]
        cluster_num = max(math.ceil(N * self.sample_ratio), 1)
        idx_cluster, cluster_num = cluster_dpc_knn(
            token_dict, cluster_num, self.k, token_mask=None) # 聚类 
        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num, token_weight) # 加权merge
        down_dict['token_num'] = down_dict['x'].shape[1]

        H, W = token_dict['map_size']
        H = math.floor((H - 1) / 2 + 1)
        W = math.floor((W - 1) / 2 + 1)
        down_dict['map_size'] = [H, W]
        #----------------------------------------------------
        x_kv = self.Multiscale(x_kv, x_3, token_dict['map_size'])
        x_kv = self.norm1(x_kv)
        token_dict['x'] = x_kv
        #----------------------------------------------------
        return down_dict, token_dict, (H, W)
        
# Attention for dynamic tokens, gather token
# to reduce k,v number
class TCGatherAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, extra_gather_layer=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio

        self.extra_gather_layer = extra_gather_layer
        if extra_gather_layer:
            self.gather = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, q_dict, kv_dict):
        q = q_dict['x']
        kv = kv_dict['x']
        B, Nq, C = q.shape
        Nkv = kv.shape[1]
        conf_kv = kv_dict['token_score'] if 'token_score' in kv_dict.keys() else kv.new_zeros(B, Nkv, 1)

        q = self.q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        if 'gather_dict' in q_dict:
            gather_dict = q_dict['gather_dict']

            tmp = torch.cat([kv, conf_kv], dim=-1)
            tmp_dict = kv_dict.copy()
            tmp_dict['x'] = tmp
            tmp_dict['map_size'] = q_dict['map_size']
            tmp = token_downup(gather_dict, tmp_dict)
            kv = tmp[..., :C]
            conf_kv = tmp[..., C:]

        if self.extra_gather_layer:
            kv = self.gather(kv)
            kv = self.norm(kv)
            kv = self.act(kv)

        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        conf_kv = conf_kv.squeeze(-1)[:, None, None, :]
        attn = attn + conf_kv
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Transformer block for dynamic tokens
class TCGatherBlock(TCBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, extra_gather_layer=True):
        super(TCBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TCGatherAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
            extra_gather_layer=extra_gather_layer
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TCMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)



class DCN(nn.Module):
    def __init__(self, embed_dim, dim_out):
        super().__init__()
        self.dim_out = dim_out
        # self.conv = TokenConv(in_channels=embed_dim, out_channels=dim_out, kernel_size=3, stride=2, padding=1)
        self.conv = change_channel(dim=128)
        self.norm = nn.LayerNorm(self.dim_out)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.dim_out, self.dim_out, 5, 1, 2, groups=self.dim_out),
            LayerNormProxy(self.dim_out),
            nn.GELU(),
            nn.Conv2d(self.dim_out, 2, 2, 2, 0, bias=False)
        )

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(1, H_key - 1, H_key - 1, dtype=dtype, device=device), 
            torch.linspace(1, W_key - 1, W_key - 1, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1) # 归一化到-1到1
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B, -1, -1, -1)
        
        return ref
    
    def forward(self, token_dict):
        token_dict = token_dict.copy()
        x = self.conv(token_dict)
        x = self.norm(x)

        dtype, device = x.dtype, x.device

        token_dict['x'] = x
        B, N, C = x.shape
        H, W = token_dict['map_size'][0], token_dict['map_size'][1]

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        offset = self.conv_offset(x)

        
        # 这个是用来限制偏移距离的
        offset_range = torch.tensor([1.0 / H, 1.0 / W], device=device).reshape(1, 2, 1, 1) # 这个是限制offset的大小
        offset = offset.tanh().mul(offset_range).mul(2) # .mul(offset_range).mul(2)是为了现在偏移的距离的
        
        offset = offset.permute(0,2,3,1).contiguous()
        reference = self._get_ref_points(H, W, B, dtype, device) # 这个就相当于散点的坐标系的位置

        pos = offset + reference # 这里是偏移量offset + 每个点的位置reference得到实际的点在坐标系中的位置pos,这个reference感觉是按照align_corners=False设置的
        # pos = (offset + reference).tanh()
        # 这玩意儿好像可微
        x = F.grid_sample(
            input=x.reshape(B, self.dim_out, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
        H, W = token_dict['map_size']
        H = math.floor((H - 1) / 2 + 1)
        W = math.floor((W - 1) / 2 + 1)
        token_dict['map_size'] = [H, W]
        token_dict['x'] = x
        return token_dict, H, W


class DeformablePatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, dim_out, img_size=224, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.kernel_size = 2
        self.stride = 2
        self.padding = 0
        self.c_in = dim
        self.c_out = dim_out
        img_size = to_2tuple(img_size)
        self.dconv = DeformConv2dPack(self.c_in, self.c_out, kernel_size=2, stride=2, padding=0)
        self.norm_layer = nn.BatchNorm2d(self.c_out)
        self.H, self.W = img_size[0] // 2, img_size[1] // 2
        self.num_patches = self.H * self.W
        self.act_layer = nn.GELU()

    # def forward(self, x, return_offset=False):
    #     """
    #     x: B, H*W, C
    #     """
    #     x = token_dict['x']
    #     H, W = token_dict['map_size']
    #     B, L, C = x.shape
    #     assert L == H * W, "input feature has wrong size"
    #     assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

    #     x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous() # token->map
    #     x, offset = self.dconv(x, return_offset=return_offset)
    #     # x = x.flatten(2).transpose(1, 2)
    #     # x= self.norm(x)
    #     x = self.act_layer(self.norm_layer(x)).flatten(2).transpose(1, 2)
    #     token_dict['x'] = x
    #     token_dict['map_size'] = [H // 2, W // 2]
    #     token_dict['token_num'] = x.shape[1]
    #     if return_offset:
    #         return x, offset
    #     else:
    #         return token_dict, (H // 2, W // 2)

    def forward(self, x, hw_shape, return_offset=False):
        """
        x: B, H*W, C
        """
        H, W = hw_shape[0], hw_shape[1]
        B, L, C = x.shape[0], x.shape[2] * x.shape[3], x.shape[1]
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous() # token->map
        x, offset = self.dconv(x, return_offset=return_offset)
        # x = x.flatten(2).transpose(1, 2)
        # x= self.norm(x)
        x = self.act_layer(self.norm_layer(x)).flatten(2).transpose(1, 2)
        if return_offset:
            return x, offset
        else:
            return x, (H // 2, W // 2)



    # def extra_repr(self) -> str:
    #     return f"input_resolution={self.input_resolution}, dim={self.dim}"


    # def extra_repr(self) -> str:
    #     return f"input_resolution={self.input_resolution}, dim={self.dim}"

class Conv_downsample(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.kernel_size = 2
        self.stride = 2
        self.padding = 0
        self.c_in = dim
        self.c_out = dim_out
        self.conv_downsample = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(self.c_out, elementwise_affine=True)

    def forward(self, token_dict):
        """
        x: B, H*W, C
        """
        x = token_dict['x']
        H, W = token_dict['map_size']
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous() # token->map
        x = self.conv_downsample(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        token_dict['x'] = x
        token_dict['map_size'] = [H // 2, W // 2]
        token_dict['token_num'] = x.shape[1]
        return token_dict, H // 2, W // 2

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"