from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F
import numpy as np


class TemporalFeatureGraph(nn.Module):
    def __init__(self, in_channels, k=4, initial_threshold=0.9, threshold_decay=0.1, layer_idx=1,
                 use_dynamic_threshold=True):
        super(TemporalFeatureGraph, self).__init__()
        self.k = k  # 保留k参数以兼容现有代码
        self.reduction_channel = in_channels
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, self.reduction_channel, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0)),
            nn.BatchNorm3d(self.reduction_channel)
        )
        self.up_conv = nn.Sequential(
            nn.Conv3d(in_channels, self.reduction_channel, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0)),
            nn.BatchNorm3d(self.reduction_channel)
        )
        self.gconv = GCNConv(self.reduction_channel, self.reduction_channel)

        # 动态阈值参数
        self.use_dynamic_threshold = use_dynamic_threshold  # 是否使用动态阈值
        self.initial_threshold = initial_threshold  # 初始阈值 t_1
        self.threshold_decay = threshold_decay  # 递减步长 Δt
        self.layer_idx = layer_idx  # 当前层索引 l

    def forward(self, x, batch):
        tlen, c, h, w = x.shape
        x = rearrange(x.view(batch, tlen // batch, c, h, w), "b v c h w-> b c v h w")

        # 使用3D卷积提取特征
        x = self.down_conv(x)
        x = rearrange(x, "b c v h w-> b c v (h w)")

        # 分离相邻帧的特征
        x1, x2 = x[:, :, :-1, :], x[:, :, 1:, :]  # b c t-1 hw

        # 计算负欧氏距离作为相似性分数
        sim = -ForEucDis(x1, x2)
        b, t_1, hw, hw = sim.shape

        if self.use_dynamic_threshold:
            # 实现论文中的方法：计算全局均值和标准差
            sim_flat = sim.reshape(b, t_1, -1)
            mu_s = sim_flat.mean(dim=-1, keepdim=True)  # 公式(10)
            sigma_s = sim_flat.std(dim=-1, keepdim=True)  # 公式(11)

            # Z-score归一化相似性分数
            sim_norm = (sim_flat - mu_s) / (sigma_s + 1e-8)  # 公式(12)
            sim_norm = sim_norm.view(b, t_1, hw, hw)

            # 计算动态阈值
            t_l = self.initial_threshold - (self.layer_idx - 1) * self.threshold_decay  # 公式(13)

            # 计算标准正态分布逆累积分布函数的阈值
            # 使用近似计算，避免依赖scipy
            norm_threshold = norm_ppf(t_l)
            norm_threshold = torch.tensor(norm_threshold).to(x.device)

            # 根据阈值创建边
            finaledge = torch.zeros((b, t_1 * hw * hw, 2), dtype=torch.long)
            edge_count = torch.zeros((b, t_1), dtype=torch.long)

            for b_idx in range(b):
                for t_idx in range(t_1):
                    # 获取满足条件的边
                    curr_edges = (sim_norm[b_idx, t_idx] > norm_threshold).nonzero()

                    if curr_edges.size(0) > 0:
                        # 限制边的数量，避免内存问题
                        max_edges = min(curr_edges.size(0), hw * 10)  # 每个节点最多10个边
                        if curr_edges.size(0) > max_edges:
                            perm = torch.randperm(curr_edges.size(0))[:max_edges]
                            curr_edges = curr_edges[perm]

                        edges_count = curr_edges.size(0)
                        edge_count[b_idx, t_idx] = edges_count

                        # 设置边的索引
                        finaledge[b_idx, t_idx * hw * hw:t_idx * hw * hw + edges_count, 0] = curr_edges[:,
                                                                                             0] + t_idx * hw
                        finaledge[b_idx, t_idx * hw * hw:t_idx * hw * hw + edges_count, 1] = curr_edges[:, 1] + (
                                    t_idx + 1) * hw

            # 移除多余的零
            max_edges_per_batch = torch.sum(edge_count, dim=1).max().item()
            finaledge = finaledge[:, :max_edges_per_batch, :]

            # 创建反向边
            finaledge_re = torch.stack((finaledge[:, :, 1], finaledge[:, :, 0]), dim=-1)
            finaledge = torch.cat((finaledge, finaledge_re), dim=1).permute(0, 2, 1)

        else:
            sim = F.normalize(sim.view(b, t_1, -1), dim=-1)
            sim = torch.where(sim < 0.05, -100, sim)  # 替换为大负值而非100，以便能正确选择topk
            _, topk_indices = torch.topk(sim, k=self.k)
            row_indices = torch.div(topk_indices, hw, rounding_mode='trunc')
            col_indices = topk_indices % hw

            finaledge = torch.zeros((b, t_1, self.k, 2), dtype=torch.long)
            for i in range(t_1):
                finaledge[:, i, :, 0] = row_indices[:, i, :] + i * hw
                finaledge[:, i, :, 1] = col_indices[:, i, :] + (i + 1) * hw

            finaledge = finaledge.view(b, t_1 * self.k, 2)
            finaledge_re = torch.stack((finaledge[:, :, 1], finaledge[:, :, 0]), dim=-1)
            finaledge = torch.cat((finaledge, finaledge_re), dim=1).permute(0, 2, 1)

        # 应用图卷积
        x = rearrange(x, "b c v n-> b (v n) c")
        out = torch.zeros_like(x).to(x.device)

        for i in range(batch):
            out[i] = self.gconv(x[i], finaledge[i].to(x.device))

        # 恢复原始形状并应用上卷积
        x = out.permute(0, 2, 1).view(b, self.reduction_channel, tlen // b, h, w)
        x = self.up_conv(x).permute(0, 2, 1, 3, 4).contiguous().view(tlen, c, h, w)

        return x


def ForEucDis(x, y):
    with torch.no_grad():
        b, c, t, n = x.shape
        x = x.permute(0, 2, 3, 1)  # b t n c
        y = y.permute(0, 2, 3, 1)
        x = x.reshape(b, t, n, c)
        y = y.reshape(b, t, n, c)
        return torch.cdist(x, y)


# 标准正态分布的逆累积分布函数的近似计算
def norm_ppf(p):
    # 根据https://en.wikipedia.org/wiki/Normal_distribution#Quantile_function的近似公式
    if p < 0 or p > 1:
        raise ValueError("Probability p must be between 0 and 1")

    if p == 0:
        return float('-inf')
    elif p == 1:
        return float('inf')

    # 对称性
    if p > 0.5:
        return -norm_ppf(1 - p)

    # 近似计算
    t = np.sqrt(-2 * np.log(p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    return t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)