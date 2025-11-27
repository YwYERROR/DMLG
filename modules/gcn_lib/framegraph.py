import torch
from torch import nn
import torch.nn.functional as F
import math


class MRConv(nn.Module):
    """Max-Relative Graph Convolution Layer"""

    def __init__(self, in_channels, out_channels):
        super(MRConv, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, x_j):
        """
        x: 原始特征
        x_j: 扩展后的特征
        """
        x_cat = torch.cat([x, x_j], dim=1)  # 拼接原始特征和扩展特征
        x_max_feature = self.nn(x_cat)  # 处理后的特征
        return x + x_max_feature  # 残差连接


class LSGC(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate=2):
        super(LSGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_rate = expansion_rate

        # 1x1 卷积处理特征差异
        self.diff_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

        # 原始特征的变换
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.input_bn = nn.BatchNorm2d(out_channels)

        # Max-Relative Graph Convolution
        self.mrconv = MRConv(out_channels, out_channels)

        # 初始化权重
        nn.init.kaiming_normal_(self.diff_conv.weight)
        nn.init.kaiming_normal_(self.input_conv.weight)

    def compute_bit_depth(self, size):
        """
        计算位深度，向上取整log2(size)
        """
        return math.ceil(math.log2(size))

    def compute_expansion_steps(self, bit_depth):
        """
        根据论文公式计算扩展步长：K×2^(n-1)
        """
        return [self.expansion_rate * (2 ** (n - 1)) for n in range(1, bit_depth + 1)]

    def expand_feature(self, x, shift, dim, direction='forward'):
        """
        在特定维度上扩展特征图，使用循环填充确保输出尺寸与输入相同
        """
        B, C, H, W = x.shape

        # 确保步长在循环意义上有效（取余）
        if dim == 2:  # 垂直方向
            effective_shift = shift % H
            if effective_shift == 0:  # 如果步长是高度的整数倍，则直接返回原始特征
                return x
        elif dim == 3:  # 水平方向
            effective_shift = shift % W
            if effective_shift == 0:  # 如果步长是宽度的整数倍，则直接返回原始特征
                return x
        else:
            raise ValueError("Invalid dimension for expansion")

        # 使用roll函数进行循环移位，确保输出尺寸与输入一致
        if dim == 2:  # 垂直方向扩展（高度）
            if direction == 'forward':
                return torch.roll(x, shifts=effective_shift, dims=2)
            elif direction == 'backward':
                return torch.roll(x, shifts=-effective_shift, dims=2)
        elif dim == 3:  # 水平方向扩展（宽度）
            if direction == 'forward':
                return torch.roll(x, shifts=effective_shift, dims=3)
            elif direction == 'backward':
                return torch.roll(x, shifts=-effective_shift, dims=3)

        return x  # 防止未知情况

    def forward(self, x):
        B, C, H, W = x.shape

        # 计算位深度
        bit_depth_h = self.compute_bit_depth(H)
        bit_depth_w = self.compute_bit_depth(W)
        max_bit_depth = max(bit_depth_h, bit_depth_w)

        # 计算扩展步长
        steps = self.compute_expansion_steps(max_bit_depth)

        # 初始化特征差异存储
        feature_diffs = []

        # 四个方向扩展并计算特征差异
        for step in steps:
            # 向右扩展
            x_right = self.expand_feature(x, step, dim=3, direction='forward')
            diff_right = x - x_right
            feature_diffs.append(diff_right)

            # 向左扩展
            x_left = self.expand_feature(x, step, dim=3, direction='backward')
            diff_left = x - x_left
            feature_diffs.append(diff_left)

            # 向下扩展
            x_down = self.expand_feature(x, step, dim=2, direction='forward')
            diff_down = x - x_down
            feature_diffs.append(diff_down)

            # 向上扩展
            x_up = self.expand_feature(x, step, dim=2, direction='backward')
            diff_up = x - x_up
            feature_diffs.append(diff_up)

        # 特征差异的逐元素最大池化
        feature_max = torch.stack(feature_diffs, dim=0).max(dim=0)[0]

        # 处理特征差异
        x_diff_processed = self.diff_conv(feature_max)
        x_diff_processed = self.bn(x_diff_processed)

        # 处理原始特征
        x_processed = self.input_conv(x)
        x_processed = self.input_bn(x_processed)

        # 使用Max-Relative图卷积更新节点特征
        x_new = self.mrconv(x_processed, x_diff_processed)

        return x_new


class FrameFeatureGraph(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate=2):
        super(FrameFeatureGraph, self).__init__()
        self.graph_construction = LSGC(
            in_channels,
            out_channels,
            expansion_rate
        )

    def forward(self, x):
        x = F.normalize(x, p=2.0, dim=1)
        return self.graph_construction(x)