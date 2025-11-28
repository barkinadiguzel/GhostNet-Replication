import torch
import torch.nn as nn

class FlattenLayer(nn.Module):
    """Flatten feature map to vector"""
    def forward(self, x):
        return x.view(x.size(0), -1)
