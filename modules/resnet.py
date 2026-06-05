import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from einops import rearrange
from modules.gcn_lib.torch_vertex import Grapher
from modules.gcn_lib.temporalgraph import TemporalFeatureGraph

# Model URLs
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            residual = self.downsample(x)

        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # DMLG graph stages operate on the final convolutional feature map before pooling.
        stage_nodes = [7 * 7, 4 * 4, 2 * 2, 1 * 1]
        self.frame_graphs = nn.ModuleList([
            Grapher(in_channels=512, kernel_size=3, n=n, relative_pos=True, expansion_rate=1)
            for n in stage_nodes
        ])
        self.cross_frame_graphs = nn.ModuleList([
            TemporalFeatureGraph(k=max(1, n // 4),
                                 in_channels=512,
                                 initial_threshold=0.8,
                                 threshold_decay=0.05,
                                 layer_idx=idx + 1,
                                 use_dynamic_threshold=True)
            for idx, n in enumerate(stage_nodes)
        ])
        self.alpha = nn.Parameter(torch.ones(4, 2), requires_grad=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=(1, stride, stride),
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _stage_size(height, width, stage_idx):
        stride = 2 ** stage_idx
        return max(1, (height + stride - 1) // stride), max(1, (width + stride - 1) // stride)

    @staticmethod
    def _spatial_pool(x, output_size):
        if x.shape[-2:] == output_size:
            return x
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = F.adaptive_avg_pool2d(x, output_size=output_size)
        return rearrange(x, '(b t) c h w -> b c t h w', b=b, t=t)

    def _apply_graph_stage(self, x, stage_idx):
        b, c, t, h, w = x.size()
        x_flat = rearrange(x, 'b c t h w -> (b t) c h w')
        x_flat = x_flat + self.frame_graphs[stage_idx](x_flat) * self.alpha[stage_idx, 0]
        x_flat = x_flat + self.cross_frame_graphs[stage_idx](x_flat, b) * self.alpha[stage_idx, 1]
        return rearrange(x_flat, '(b t) c h w -> b c t h w', b=b)

    def _build_dmlg_stages(self, x):
        _, _, _, base_h, base_w = x.shape
        stages = []
        cur = x
        for idx in range(4):
            target_size = self._stage_size(base_h, base_w, idx)
            cur = self._spatial_pool(cur, target_size)
            cur = self._apply_graph_stage(cur, idx)
            stages.append(cur)
        return stages

    def forward(self, x: torch.Tensor) -> dict:
        N, C, T, H, W = x.size()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        stage_features = self._build_dmlg_stages(x)
        x = rearrange(stage_features[-1], 'N C T H W -> (N T) C H W')
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x).view(N, T, -1).permute(0, 2, 1).contiguous()

        return {
            "sequence_features": x,
            "stage_features": stage_features,
        }


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    for k, v in checkpoint.items():
        if 'conv' in k or 'downsample.0.weight' in k:
            checkpoint[k] = v.unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet34'])
    for k, v in checkpoint.items():
        if 'conv' in k or 'downsample.0.weight' in k:
            checkpoint[k] = v.unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model
