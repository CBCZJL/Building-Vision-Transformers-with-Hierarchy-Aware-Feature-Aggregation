# import math
# import torch.nn as nn
# from mmcv.cnn import ConvModule
# from mmcv.runner import BaseModule
# from ..utils import trunc_normal_
# from ..utils import TCBlock
# from ..utils import token2map, token_downup
# import warnings
# import torch.nn.functional as F
# from mmcv.runner import BaseModule

# from ..builder import NECKS


# @NECKS.register_module()
# # MTA block with typical spatial reduction attention block
# class MTA(BaseModule):
#     def __init__(self,
#                  in_channels=[64, 128, 320, 512],
#                  out_channels=128,
#                  num_outs=1,
#                  start_level=2,
#                  end_level=-1,
#                  num_heads=[2, 2, 2, 2],
#                  mlp_ratios=[4, 4, 4, 4],
#                  sr_ratios=[8, 4, 2, 1],
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.,
#                  norm_layer=nn.LayerNorm,
#                  no_norm_on_lateral=False,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  act_cfg=None,
#                  init_cfg=dict(
#                      type='Xavier', layer='Conv2d', distribution='uniform'),
#                  add_extra_convs=False,
#                  extra_convs_on_inputs=True,
#                  relu_before_extra_convs=False,
#                  use_sr_layer=True,
#                  ):
#         super().__init__(init_cfg)
#         assert isinstance(in_channels, list)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_ins = len(in_channels)
#         self.num_outs = num_outs
#         self.no_norm_on_lateral = no_norm_on_lateral
#         self.fp16_enabled = False
#         self.norm_cfg = norm_cfg
#         self.conv_cfg = conv_cfg
#         self.act_cfg = act_cfg
#         self.mlp_ratios = mlp_ratios

#         self.start_level = start_level
#         if end_level == -1:
#             end_level = len(in_channels) - 1
#         self.end_level = end_level

#         self.lateral_convs = nn.ModuleList()
#         self.merge_blocks = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()

#         for i in range(self.start_level, self.end_level + 1):
#             l_conv = ConvModule(
#                 in_channels[i],
#                 out_channels,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
#                 act_cfg=act_cfg,
#                 inplace=False)
#             fpn_conv = ConvModule(
#                 out_channels,
#                 out_channels,
#                 3,
#                 padding=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#                 inplace=False)

#             self.lateral_convs.append(l_conv)
#             self.fpn_convs.append(fpn_conv)

#         # for i in range(self.start_level, self.end_level):
#         #     merge_block = TCBlock(
#         #         dim=out_channels, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
#         #         qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
#         #         attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
#         #         sr_ratio=sr_ratios[i], use_sr_layer=use_sr_layer,
#         #     )
#         #     self.merge_blocks.append(merge_block)

#         # add extra conv layers (e.g., RetinaNet)
#         self.relu_before_extra_convs = relu_before_extra_convs
#         assert isinstance(add_extra_convs, (str, bool))
#         if isinstance(add_extra_convs, str):
#             # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
#             assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
#             self.add_extra_convs = add_extra_convs
#         elif add_extra_convs:  # True
#             if extra_convs_on_inputs:
#                 # TODO: deprecate `extra_convs_on_inputs`
#                 warnings.simplefilter('once')
#                 warnings.warn(
#                     '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
#                     'Please use "add_extra_convs"', DeprecationWarning)
#                 self.add_extra_convs = 'on_input'
#             else:
#                 self.add_extra_convs = 'on_output'
#         else:
#             self.add_extra_convs = add_extra_convs

#         self.extra_convs = nn.ModuleList()
#         extra_levels = num_outs - (self.end_level + 1 - self.start_level)
#         if self.add_extra_convs and extra_levels >= 1:
#             for i in range(extra_levels):
#                 if i == 0 and self.add_extra_convs == 'on_input':
#                     in_channels = self.in_channels[self.end_level]
#                 else:
#                     in_channels = out_channels
#                 extra_fpn_conv = ConvModule(
#                     in_channels,
#                     out_channels,
#                     3,
#                     stride=2,
#                     padding=1,
#                     conv_cfg=conv_cfg,
#                     norm_cfg=norm_cfg,
#                     act_cfg=act_cfg,
#                     inplace=False)
#                 self.extra_convs.append(extra_fpn_conv)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, inputs):
#         """Forward function."""
#         assert len(inputs) == len(self.in_channels)

#         # build lateral tokens
#         input_dicts = []
#         outs = []
#         for i, lateral_conv in enumerate(self.lateral_convs):
#             if i < len(inputs) - 1:
#                 tmp = inputs[i + self.start_level]
#                 tmp = lateral_conv(tmp.unsqueeze(2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(2)
#                 input_dicts.append(tmp)
#             else:
#                 tmp = inputs[i + self.start_level].copy()
#                 # tmp['x'] = lateral_conv(tmp['x'])
#                 tmp['x'] = lateral_conv(tmp['x'].unsqueeze(2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(2)
#                 input_dicts.append(tmp)

#         # merge from high level to low level
#         for i in range(len(input_dicts) - 2, -1, -1):
#             if i == 2:
#                 input_dicts[i] = input_dicts[i] + token_downup(input_dicts[i], input_dicts[i + 1])

#             else:
#                 prev_shape = input_dicts[i].shape[2:]
#                 input_dicts[i] = input_dicts[i] + F.interpolate(
#                     input_dicts[i + 1], size=prev_shape, mode='nearest')

#         for i in range(self.num_outs-1):
#             if i != len(inputs) - 1 :
#                 outs.append(self.fpn_convs[i](input_dicts[i]))
#             else:
#                 outs.append(self.fpn_convs[i](input_dicts[i]['x']))
#             # input_dicts[i] = self.merge_blocks[i](input_dicts[i]) ???别人都是conv你直接block是吧

#         # part 2: add extra levels
#         used_backbone_levels = len(outs)
#         if self.num_outs > len(outs):
#             # use max pool to get more levels on top of outputs
#             # (e.g., Faster R-CNN, Mask R-CNN)
#             if not self.add_extra_convs:
#                 for i in range(self.num_outs - used_backbone_levels):
#                     outs.append(F.max_pool2d(outs[-1], 1, stride=2))

#             # add conv layers on top of original feature maps (RetinaNet)
#             else:
#                 if self.add_extra_convs == 'on_input':
#                     tmp = inputs[self.end_level]
#                     extra_source = token2map(tmp)
#                 elif self.add_extra_convs == 'on_output':
#                     extra_source = outs[-1]
#                 else:
#                     raise NotImplementedError

#                 outs.append(self.extra_convs[0](extra_source))
#                 for i in range(1, self.num_outs - used_backbone_levels):
#                     if self.relu_before_extra_convs:
#                         outs.append(self.extra_convs[i](F.relu(outs[-1])))
#                     else:
#                         outs.append(self.extra_convs[i](outs[-1]))
#         return outs

import math
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from ..utils import trunc_normal_
from ..utils import TCBlock
from ..utils import token2map, token_downup
import warnings
import torch.nn.functional as F
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
# MTA block with typical spatial reduction attention block
class MTA(BaseModule):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 out_channels=128,
                 num_outs=1,
                 start_level=2,
                 end_level=-1,
                 num_heads=[2, 2, 2, 2],
                 mlp_ratios=[4, 4, 4, 4],
                 sr_ratios=[8, 4, 2, 1],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 use_sr_layer=True,
                 ):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.mlp_ratios = mlp_ratios

        self.start_level = start_level
        if end_level == -1:
            end_level = len(in_channels) - 1
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.end_level + 1):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # for i in range(self.start_level, self.end_level):
        #     merge_block = TCBlock(
        #         dim=out_channels, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
        #         qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
        #         attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
        #         sr_ratio=sr_ratios[i], use_sr_layer=use_sr_layer,
        #     )
        #     self.merge_blocks.append(merge_block)

        # add extra conv layers (e.g., RetinaNet)
        self.relu_before_extra_convs = relu_before_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
            self.add_extra_convs = add_extra_convs
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'
        else:
            self.add_extra_convs = add_extra_convs

        self.extra_convs = nn.ModuleList()
        extra_levels = num_outs - (self.end_level + 1 - self.start_level)
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.end_level]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.extra_convs.append(extra_fpn_conv)

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
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build lateral tokens
        input_dicts = []
        outs = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            tmp = inputs[i + self.start_level].copy()
            tmp['x'] = lateral_conv(tmp['x'].unsqueeze(2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(2)
            input_dicts.append(tmp)

        # merge from high level to low level
        for i in range(len(input_dicts) - 2, -1, -1):
            if i == 2:
                H, W = input_dicts[i]['map_size'][0], input_dicts[i]['map_size'][1]
                B, C = input_dicts[i]['x'].shape[0], input_dicts[i]['x'].shape[2]
                input_dicts[i]['x'] = input_dicts[i]['x'] + token_downup(input_dicts[i], input_dicts[i + 1])
                input_dicts[i]['x'] = input_dicts[i]['x'].reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

                # outs.append(self.fpn_convs[i](input_dicts[i]['x']))
            else:
                prev_shape = input_dicts[i]['map_size']
                H, W = input_dicts[i]['map_size'][0], input_dicts[i]['map_size'][1]
                B, C = input_dicts[i]['x'].shape[0], input_dicts[i]['x'].shape[2]
                input_dicts[i]['x'] = input_dicts[i]['x'].reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                input_dicts[i]['x'] = input_dicts[i]['x'] + F.interpolate(
                    input_dicts[i + 1]['x'], size=prev_shape, mode='nearest')
                # outs.append(self.fpn_convs[i](input_dicts[i]['x'].reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()))
        input_dicts[3]['x'] = token2map(input_dicts[3])
        for i in range(self.num_outs-1):
            outs.append(self.fpn_convs[i](input_dicts[i]['x']))
            # input_dicts[i] = self.merge_blocks[i](input_dicts[i]) ???别人都是conv你直接block是吧

        # part 2: add extra levels
        used_backbone_levels = len(outs)
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))

            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    tmp = inputs[self.end_level]
                    extra_source = token2map(tmp)
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError

                outs.append(self.extra_convs[0](extra_source))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.extra_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.extra_convs[i](outs[-1]))
        return outs

