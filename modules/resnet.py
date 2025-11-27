import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
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

        # Graph modules
        self.localG = Grapher(in_channels=256, kernel_size=3, n=14 * 14, relative_pos=True)
        self.localG2 = Grapher(in_channels=512, kernel_size=4, n=7 * 7, relative_pos=True)
        self.temporalG = TemporalFeatureGraph(k=14 * 14 // 4,
                                              in_channels=256,
                                              initial_threshold=0.9,
                                              threshold_decay=0.1,
                                              layer_idx=1,
                                              use_dynamic_threshold=True)
        self.temporalG2 = TemporalFeatureGraph(k=7 * 7, in_channels=512)
        self.alpha = nn.Parameter(torch.ones(4), requires_grad=True)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, H, W = x.size()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        N, C, T, H, W = x.size()
        x = rearrange(x, 'N C T H W -> (N T) C H W')
        x = x + self.localG(x) * self.alpha[0]
        x = x + self.temporalG(x, N) * self.alpha[1]
        x = rearrange(x, '(N T) C H W -> N C T H W', N=N)

        x = self.layer4(x)

        N, C, T, H, W = x.size()
        x = rearrange(x, 'N C T H W -> (N T) C H W')
        x = x + self.localG2(x) * self.alpha[2]
        x = x + self.temporalG2(x, N) * self.alpha[3]
        x = rearrange(x, '(N T) C H W -> N C T H W', N=N)

        x = rearrange(x, 'N C T H W -> (N T) C H W')
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)

        return x


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
