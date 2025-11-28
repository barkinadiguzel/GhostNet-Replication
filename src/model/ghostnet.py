import torch
import torch.nn as nn
from src.layers.ghost_module import GhostModule


class GhostBottleneck(nn.Module):
    def __init__(self, in_ch, exp_ch, out_ch, stride=1, use_se=False):
        super().__init__()

        # 1. Expansion GhostModule
        self.expand = GhostModule(in_ch, exp_ch, relu=True)

        # 2. Depthwise conv ONLY if stride=2 
        if stride == 2:
            self.dw = nn.Sequential(
                nn.Conv2d(exp_ch, exp_ch, kernel_size=3, stride=2,
                          padding=1, groups=exp_ch, bias=False),
                nn.BatchNorm2d(exp_ch),
            )
        else:
            self.dw = nn.Identity()

        # 3. Optional SE 
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(exp_ch, exp_ch // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(exp_ch // 4, exp_ch, 1),
                nn.Hardsigmoid()
            )
        else:
            self.se = nn.Identity()

        # 4. Projection GhostModule
        self.project = GhostModule(exp_ch, out_ch, relu=False)

        # 5. Shortcut 
        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                          padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.expand(x)
        out = self.dw(out)
        out = out * self.se(out)
        out = self.project(out)
        return out + self.shortcut(x)
