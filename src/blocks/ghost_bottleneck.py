import torch
import torch.nn as nn
from src.layers.ghost_module import GhostModule

class SEBlock(nn.Module):
    def __init__(self, inp, reduction=4):
        super().__init__()
        squeeze_channels = inp // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze_channels),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_channels, inp),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class GhostBottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1, use_se=False):
        super().__init__()

        # 1. Expansion phase (GhostModule)
        self.expand = GhostModule(in_ch, mid_ch, relu=True)

        # 2. Depthwise convolution (only if stride == 2 or for spatial mixing)
        self.dw = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride,
                      padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch)
        ) if stride == 2 else nn.Identity()

        # 3. Optional SE block
        self.se = SEBlock(mid_ch) if use_se else nn.Identity()

        # 4. Projection phase (GhostModule)
        self.project = GhostModule(mid_ch, out_ch, relu=False)

        # Shortcut branch
        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                          padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.expand(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.project(out)
        return out + self.shortcut(x)
