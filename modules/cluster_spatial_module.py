import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterBasedSpatialAwarenessModule(nn.Module):
    """
    基于聚类的空间感知模块
    通过多级窗口逐级更新中心节点表示，增强手语动作轨迹的时空一致性建模能力
    """

    def __init__(self, feature_dim, window_levels=[3, 5, 7], dropout=0.1):
        """
        参数:
            feature_dim (int): 特征维度 D
            window_levels (list): 各级窗口半径列表，如[3, 5, 7]表示三级窗口
            dropout (float): dropout率
        """
        super(ClusterBasedSpatialAwarenessModule, self).__init__()
        self.feature_dim = feature_dim
        self.window_levels = window_levels
        self.num_levels = len(window_levels)

        # 可学习参数α和β，用于调整相似性权重
        self.alphas = nn.Parameter(torch.ones(self.num_levels))
        self.betas = nn.Parameter(torch.zeros(self.num_levels))

        # 每级窗口的特征变换矩阵W^(m)
        self.transforms = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(self.num_levels)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入: x 形状为 [seq_len, batch_size, feature_dim] 或 [batch_size, seq_len, feature_dim]
        输出: 形状相同的更新特征
        """
        # 确定输入格式并统一处理为 [batch_size, seq_len, feature_dim]
        batch_first = (x.dim() == 3 and x.size(0) > x.size(1))
        if not batch_first:
            x = x.transpose(0, 1)  # 转换为 batch_first

        batch_size, seq_len, feature_dim = x.shape
        output = x.clone()

        for m in range(self.num_levels):
            window_radius = self.window_levels[m]
            transform = self.transforms[m]
            alpha = self.alphas[m]
            beta = self.betas[m]

            # 对每个时间步更新特征
            for t in range(seq_len):
                # 构建时序窗口内的节点集合N(F_t)
                start_idx = max(0, t - window_radius)
                end_idx = min(seq_len, t + window_radius + 1)

                # 提取邻域节点
                neighbors = x[:, start_idx:end_idx, :]  # [batch, window_size, feature_dim]

                # 计算聚类中心C_t (公式16)
                cluster_center = torch.mean(neighbors, dim=1, keepdim=True)  # [batch, 1, feature_dim]

                # 计算余弦相似度S_tn (公式17)
                center_norm = F.normalize(cluster_center, p=2, dim=2)
                neighbors_norm = F.normalize(neighbors, p=2, dim=2)
                similarity = torch.bmm(neighbors_norm, center_norm.transpose(1, 2)).squeeze(-1)  # [batch, window_size]

                # 调整相似性权重w_tn (公式18)
                weights = torch.sigmoid(alpha * similarity + beta)  # [batch, window_size]
                weights = weights.unsqueeze(-1)  # [batch, window_size, 1]

                # 特征变换与加权聚合 (公式19的一部分)
                transformed_neighbors = transform(neighbors)  # [batch, window_size, feature_dim]
                weighted_features = weights * transformed_neighbors  # [batch, window_size, feature_dim]
                aggregated = torch.sum(weighted_features, dim=1)  # [batch, feature_dim]

                # 更新中心节点的表示
                output[:, t, :] = output[:, t, :] + self.dropout(aggregated)

        # 恢复原始格式
        if not batch_first:
            output = output.transpose(0, 1)

        return output