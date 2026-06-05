import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super(SpatialSelfAttention, self).__init__()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.scale = channels ** -0.5

    def forward(self, x):
        b, t, h, w, c = x.shape
        x_flat = x.reshape(b * t, h * w, c)

        q = self.query(x_flat)
        k = self.key(x_flat)
        v = self.value(x_flat)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        out = out + x_flat
        return out.reshape(b, t, h, w, c)


class DynamicMultiScaleAggregationModule(nn.Module):
    """
    Dynamic Multi-Scale Aggregation Module from the DMLG paper.

    Inputs are four hierarchical stage features in [B, C, T, H, W] format. Each
    scale is resized, channel-aligned, refined by scale-specific spatial attention,
    concatenated, and fused by cross-scale attention before temporal modeling.
    """

    def __init__(self, in_channels, aligned_channels=512, target_size=(7, 7), dropout=0.1):
        super(DynamicMultiScaleAggregationModule, self).__init__()
        self.target_size = target_size
        self.aligned_channels = aligned_channels

        self.align_layers = nn.ModuleList([
            nn.Conv2d(channels, aligned_channels, kernel_size=1)
            for channels in in_channels
        ])
        self.scale_attentions = nn.ModuleList([
            SpatialSelfAttention(aligned_channels, dropout=dropout)
            for _ in in_channels
        ])

        fused_channels = aligned_channels * len(in_channels)
        self.cross_query = nn.Linear(fused_channels, aligned_channels)
        self.cross_key = nn.Linear(fused_channels, aligned_channels)
        self.cross_value = nn.Linear(fused_channels, aligned_channels)
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_scale = aligned_channels ** -0.5

    def _align_feature(self, feature, align_layer):
        b, c, t, h, w = feature.shape
        x = feature.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        x = align_layer(x)
        h_t, w_t = self.target_size
        return x.reshape(b, t, self.aligned_channels, h_t, w_t).permute(0, 1, 3, 4, 2)

    def forward(self, features):
        if len(features) != len(self.align_layers):
            raise ValueError("DMAM expects {} feature scales, got {}".format(len(self.align_layers), len(features)))

        aligned_features = []
        for feature, align_layer, attention in zip(features, self.align_layers, self.scale_attentions):
            aligned = self._align_feature(feature, align_layer)
            aligned_features.append(attention(aligned))

        multi_scale = torch.cat(aligned_features, dim=-1)
        b, t, h, w, c = multi_scale.shape
        tokens = multi_scale.reshape(b * t, h * w, c)

        q = self.cross_query(tokens)
        k = self.cross_key(tokens)
        v = self.cross_value(tokens)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.cross_scale
        attn = self.cross_dropout(torch.softmax(attn, dim=-1))
        fused = torch.matmul(attn, v)

        fused = fused.reshape(b, t, h, w, self.aligned_channels)
        fused = fused.permute(0, 4, 1, 2, 3).contiguous()
        return fused.mean(dim=(-1, -2))
