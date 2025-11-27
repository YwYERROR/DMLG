import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select
from .framegraph import FrameFeatureGraph
import torch.nn.functional as F
from timm.models.layers import DropPath

class EdgeConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = 'gelu', norm: str = None, bias: bool = True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def _select_features(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(y, edge_index[0]) if y is not None else batched_index_select(x, edge_index[0])
        return x_i, x_j

    def forward(self, x, edge_index, y=None):
        x_i, x_j = self._select_features(x, edge_index, y)
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value

class GraphConv2d(nn.Module):
    """
    静态图卷积层（基于 EdgeConv）
    """
    def __init__(self, in_channels, out_channels, act='gelu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        self.conv = 'edge'

    def forward(self, x, edge_index=None, y=None):
        if edge_index is not None:
            return self.gconv(x, edge_index, y)
        return x

class DyGraphConv2d(GraphConv2d):
    """
    动态图卷积层，使用 LSGC 构建局部图
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9,
                 dilation: int = 1, act: str = 'gelu', norm: str = None,
                 bias: bool = True, r: int = 1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, act, norm, bias)
        self.r = r
        # LSGC graph builder
        self.graph_builder = FrameFeatureGraph(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_rate=2
        )

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape

        # 下采样（可选）
        if self.r > 1:
            x_down = F.avg_pool2d(x, self.r, self.r)
            H_down = H // self.r
            W_down = W // self.r
        else:
            x_down = x
            H_down, W_down = H, W

        # 使用 LSGC 构建图
        x_processed = self.graph_builder(x_down)

        # 如果下采样过，则上采样回原始尺寸
        if self.r > 1:
            x_processed = F.interpolate(x_processed, size=(H, W), mode='bilinear', align_corners=False)

        return x_processed

class Grapher(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 9, dilation: int = 1,
                 act: str = 'gelu', norm: str = None, bias: bool = True, r: int = 1,
                 n: int = 196, drop_path: float = 0.0, relative_pos: bool = False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n  # 典型地 n = H * W（或其他）
        self.r = r

        # 前置 1x1 卷积
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        # 动态图卷积
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation,
                                        act, norm, bias, r)
        # 后置 1x1 卷积
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        # DropPath，用于正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 1) 这里用可训练的相对位置编码替换原先的 2D sin/cos 编码；
        # 2) 这里假设输出维度与 n, n//(r*r) 一致，可根据实际需求调整 shape。
        self.relative_pos = None
        if relative_pos:
            # 声明一个可训练权重，初始化时用随机值，可在训练中更新
            # 例如可用形状 [1, 1, n, n//(r*r)]，以匹配后续插值或特征尺寸需求
            self.relative_pos = nn.Parameter(
                torch.randn(1, 1, self.n, self.n // (self.r * self.r)),
                requires_grad=True
            )

    def _get_relative_pos(self, relative_pos, H, W):
        """
        在原实现中将 sincos 生成的 (n, n//(r*r)) 插值到 (H*W, (H*W)//(r*r))。
        如果需要类似功能，可在此处进行调整。
        """
        if relative_pos is None:
            return None

        # N 为当前特征图的像素数
        N = H * W
        # 下采样后对应的像素数
        N_reduced = N // (self.r * self.r)

        # 如果不需插值，或者已经是目标尺寸，可直接返回
        # 这里仅示例如何插值到相同形状
        # 需要根据真实网络结构来判断怎样插值最合适
        rel_pos_resized = F.interpolate(
            relative_pos, size=(N, N_reduced), mode="bicubic"
        )
        return rel_pos_resized

    def forward(self, x):
        # 先经过 fc1
        x = self.fc1(x)
        B, C, H, W = x.shape

        # 获取可训练的位置编码
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)

        # 将可训练的位置编码送入 graph_conv
        x = self.graph_conv(x, relative_pos)

        # 再经过 fc2
        x = self.fc2(x)

        # DropPath
        return self.drop_path(x)