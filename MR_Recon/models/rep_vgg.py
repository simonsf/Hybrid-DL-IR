#-*-coding:utf-8-*-
# @Time    : 21-7-16 下午3:36
# @Author  : 
# @Site    : 
# @File    : repvgg_block.py
# @Software: PyCharm
import math
import torch
import torch.nn as nn
import numpy as np

from typing import List, Optional, Tuple

from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint


#repvgg_block
# TODO RepVGG可能出现版本兼容问题
class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='reflect', deploy=False, use_act=True):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        # self.reflect_padd = nn.ReflectionPad2d(1)
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2

        if use_act:
            self.nonlinearity = nn.ReLU()
        else:
            self.nonlinearity = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.Identity() if in_channels == out_channels and stride == 1 else None
            self.rbr_dense = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.rbr_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                     padding=padding_11, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            # print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        # refl_inputs = self.reflect_padd(inputs)
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
            # id_out = self.rbr_identity(refl_inputs)
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._tensor_w_b(self.rbr_dense)
        kernel1x1, bias1x1 = self._tensor_w_b(self.rbr_1x1)
        kernelid, biasid = self._tensor_w_b(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _tensor_w_b(self, branch):
        if branch is None:
            kernel, bias = torch.tensor(0), torch.tensor(0)
            return kernel.cuda(), bias.cuda()

        if isinstance(branch, nn.Conv2d):
            kernel = branch.weight
            bias = branch.bias
        else:
            assert isinstance(branch, nn.Identity)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            bias = torch.tensor(0)
        return kernel.cuda(), bias.cuda()

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')


class RepVGG(nn.Module):

    # def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False, use_checkpoint=False):
    def __init__(self, num_blocks, in_chans, mid_chans, out_chans, width_multiplier=None, override_groups_map=None, deploy=False, use_checkpoint=False):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.in_channels = in_chans
        self.out_channels = out_chans
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(mid_chans, int(mid_chans * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=self.in_channels, out_channels=self.in_planes, kernel_size=3, stride=1, padding=1, deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(32 * width_multiplier[0]), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(int(64 * width_multiplier[1]), num_blocks[1], stride=1)
        self.stage3 = self._make_stage(int(128 * width_multiplier[2]), num_blocks[2], stride=1)
        self.stage4 = self._make_stage(int(256 * width_multiplier[3]), num_blocks[3], stride=1)
        # self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # TODO: modified this
        # self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)
        self.last_layer = nn.Conv2d(in_channels=self.in_planes, out_channels=self.out_channels, kernel_size=1)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)

        out = self.last_layer(out)    

        return out


class NormRepVGG(nn.Module):
    """
    Normalized RepVGG model.

    This is the same as a regular RepVGG, but with normalization applied to the
    input before the RepVGG. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_blocks: list=[2, 4, 8, 1],
        in_chans: int = 24,
        out_chans: int = 24,
        width_multiplier: list = [0.75, 0.75, 0.75, 2.5],
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_blocks: Number of blocks in every RepVGG stage.
            in_chans: Number of channels in the input to the RepVGG model.
            out_chans: Number of channels in the output to the RepVGG model.
            width_multiplier: RepVGG model widthe setting
        """
        super().__init__()

        self.repvgg = RepVGG(
            num_blocks=num_blocks,
            in_chans=in_chans,
            mid_chans=chans,
            out_chans=out_chans,
            width_multiplier=width_multiplier,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        # 相当于逐线圈并分通道做Norm
        x = x.view(b, 12, c // 12 * h * w)

        # 如果出现某些通道数据为0的情况下，那么不应该计算该std，否则出错
        mean = x.mean(dim=2).view(b, 12, 1, 1, 1)
        std = x.std(dim=2).view(b, 12, 1, 1, 1)

        x = x.view(b, c // 2, 2, h, w)
        x_norm = (x - mean) / (std + 1e-6)

        x_norm = x_norm.view(b, c, h, w)

        return x_norm, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c // 2, 2, h, w)
        x_unnorm = x * std + mean
        x_unnorm = x_unnorm.view(b, c, h, w)
        return x_unnorm

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        # 将Unet输入尺寸padding成2的下采样次方的倍数
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x = self.repvgg(x)
        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        del mean, std
        torch.cuda.empty_cache()

        return x

# def create_RepVGG_A0(deploy=False, use_checkpoint=False):
#     return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
#                   width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)


# def create_RepVGG_A1(deploy=False, use_checkpoint=False):
#     return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
#                   width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)