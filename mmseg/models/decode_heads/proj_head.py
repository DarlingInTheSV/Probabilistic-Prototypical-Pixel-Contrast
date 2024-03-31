# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn import Parameter

from ..builder import HEADS
from .decode_head_decorator import BaseDecodeHeadDecorator


class Uncertainty_head(nn.Module):  # feature -> log(sigma^2)
    def __init__(self, in_feat=2048, out_feat=512):
        super(Uncertainty_head, self).__init__()
        self.fc1 = Parameter(torch.Tensor(out_feat, in_feat))
        self.bn1 = nn.BatchNorm2d(out_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = Parameter(torch.Tensor(out_feat, out_feat))
        self.bn2 = nn.BatchNorm2d(out_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))

        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)




    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = F.linear(x, F.normalize(self.fc1, dim=-1))  # [B, W, H, D]
        x = x.permute(0, 3, 1, 2)  # [B, W, H, D] -> [B, D, W, H]
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = F.linear(x, F.normalize(self.fc2, dim=-1))
        x = x.permute(0, 3, 1, 2)
        x = self.bn2(x)
        x = self.gamma * x + self.beta
        x = x.clamp(max=80.0)
        x = torch.log(torch.exp(x) + 1e-6)
        x = torch.sigmoid(x)
        return x


@HEADS.register_module()
class ProjHead(BaseDecodeHeadDecorator):
    """Projection Head for feature dimension reduction in contrastive loss.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """
    def __init__(self,
                 num_convs=2,
                 kernel_size=1,
                 dilation=1,
                 **kwargs):
        assert num_convs in (0, 1, 2) and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.kernel_size = kernel_size
        super(ProjHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        if self.input_transform == 'multiple_select':
            convs = [[] for _ in range(len(self.in_channels))]
            for i in range(len(self.in_channels)):
                if num_convs > 1:
                    convs[i].append(
                        ConvModule(
                            self.in_channels[i],
                            self.in_channels[i],
                            kernel_size=kernel_size,
                            padding=conv_padding,
                            dilation=dilation,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                convs[i].append(
                    ConvModule(
                        self.in_channels[i],
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            if num_convs == 0:
                self.convs = nn.ModuleList([nn.Identity() for _ in range(len(self.in_channels))])
            else:
                self.convs = nn.ModuleList([nn.Sequential(*convs[i]) for i in range(len(self.in_channels))])

        else:
            if self.input_transform == 'resize_concat':
                self.mid_channels = self.in_channels // len(self.in_index)
            else:
                self.mid_channels = self.in_channels
            convs = []
            if num_convs > 1:
                convs.append(
                    ConvModule(
                        self.in_channels,
                        self.mid_channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            convs.append(
                ConvModule(
                    self.mid_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            if num_convs == 0:
                self.convs = nn.Identity()
            else:
                self.convs = nn.Sequential(*convs)
        self.uncertain_head = Uncertainty_head()


    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs) #  2x2048x80x80
        if isinstance(x, list):
            # multiple_select
            output = [F.normalize(self.convs[i](x[i]), p=2, dim=1) for i in range(len(x))]
        else:
            # resize_concat or single_select
            output = []
            output.append(F.normalize(self.convs(x), p=2, dim=1))  # miu
            # output.append(self.convs(x))  # miu
            output.append(self.uncertain_head(x))  # sigma

        return output



