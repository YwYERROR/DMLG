import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F


class TemporalMaxRelativeConv(nn.Module):
    """Max-Relative graph convolution for flattened temporal graph nodes."""

    def __init__(self, channels):
        super(TemporalMaxRelativeConv, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU()
        )

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index[0], edge_index[1]
        rel = x[src] - x[dst]
        max_rel = torch.full_like(x, -torch.inf)
        scatter_index = dst.unsqueeze(-1).expand_as(rel)

        if hasattr(max_rel, "scatter_reduce_"):
            max_rel.scatter_reduce_(0, scatter_index, rel, reduce='amax', include_self=True)
        else:
            for node in torch.unique(dst):
                max_rel[node] = rel[dst == node].max(dim=0)[0]

        max_rel = torch.where(torch.isfinite(max_rel), max_rel, torch.zeros_like(max_rel))
        return x + self.proj(torch.cat([x, max_rel], dim=-1))


class TemporalFeatureGraph(nn.Module):
    def __init__(self, in_channels, k=4, initial_threshold=0.8, threshold_decay=0.05, layer_idx=1,
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
        self.gconv = TemporalMaxRelativeConv(self.reduction_channel)

        # 动态阈值参数
        self.use_dynamic_threshold = use_dynamic_threshold  # 是否使用动态阈值
        self.initial_threshold = initial_threshold  # 初始阈值 t0
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
            if hw == 1:
                edge_indices = []
                for _ in range(b):
                    if t_1 > 0:
                        src = torch.arange(t_1, device=x.device, dtype=torch.long)
                        dst = src + 1
                        forward_edges = torch.stack((src, dst), dim=0)
                        reverse_edges = torch.stack((dst, src), dim=0)
                        edge_indices.append(torch.cat((forward_edges, reverse_edges), dim=1))
                    else:
                        edge_indices.append(torch.empty((2, 0), dtype=torch.long, device=x.device))
            else:
                # 实现论文中的方法：计算全局均值和标准差
                sim_flat = sim.reshape(b, t_1, -1)
                mu_s = sim_flat.mean(dim=-1, keepdim=True)  # 公式(10)
                sigma_s = sim_flat.std(dim=-1, keepdim=True, unbiased=False)  # 公式(11)

                # Z-score归一化相似性分数
                sim_norm = (sim_flat - mu_s) / (sigma_s + 1e-8)  # 公式(12)
                sim_norm = sim_norm.view(b, t_1, hw, hw)

                # 计算动态阈值
                t_l = self.initial_threshold - (self.layer_idx - 1) * self.threshold_decay  # 公式(13)

                # 计算标准正态分布逆累积分布函数的阈值，避免依赖 scipy。
                norm_threshold = x.new_tensor(norm_ppf(t_l))

                # 根据阈值创建边
                edge_indices = []

                for b_idx in range(b):
                    batch_edges = []
                    for t_idx in range(t_1):
                        # 获取满足条件的边
                        curr_edges = (sim_norm[b_idx, t_idx] > norm_threshold).nonzero(as_tuple=False)

                        if curr_edges.size(0) > 0:
                            src = curr_edges[:, 0] + t_idx * hw
                            dst = curr_edges[:, 1] + (t_idx + 1) * hw
                            batch_edges.append(torch.stack((src, dst), dim=0))

                    if len(batch_edges) > 0:
                        forward_edges = torch.cat(batch_edges, dim=1).long()
                        reverse_edges = torch.stack((forward_edges[1], forward_edges[0]), dim=0)
                        edge_indices.append(torch.cat((forward_edges, reverse_edges), dim=1))
                    else:
                        edge_indices.append(torch.empty((2, 0), dtype=torch.long, device=x.device))

        else:
            sim = F.normalize(sim.view(b, t_1, -1), dim=-1)
            sim = torch.where(sim < 0.05, -100, sim)  # 替换为大负值而非100，以便能正确选择topk
            _, topk_indices = torch.topk(sim, k=self.k)
            row_indices = torch.div(topk_indices, hw, rounding_mode='trunc')
            col_indices = topk_indices % hw

            edge_indices = []
            for b_idx in range(b):
                batch_edges = []
                for i in range(t_1):
                    src = row_indices[b_idx, i, :] + i * hw
                    dst = col_indices[b_idx, i, :] + (i + 1) * hw
                    batch_edges.append(torch.stack((src, dst), dim=0))
                forward_edges = torch.cat(batch_edges, dim=1).long()
                reverse_edges = torch.stack((forward_edges[1], forward_edges[0]), dim=0)
                edge_indices.append(torch.cat((forward_edges, reverse_edges), dim=1))

        # 应用图卷积
        x = rearrange(x, "b c v n-> b (v n) c")
        out = torch.zeros_like(x).to(x.device)

        for i in range(batch):
            out[i] = self.gconv(x[i], edge_indices[i].to(x.device))

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
    if p < 0 or p > 1:
        raise ValueError("Probability p must be between 0 and 1")

    if p == 0:
        return float('-inf')
    elif p == 1:
        return float('inf')

    return torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(float(p))).item()
