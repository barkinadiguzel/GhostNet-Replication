import torch.nn as nn

class MaxPoolLayer(nn.Module):
    """Max pooling layer"""
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)
    
    def forward(self, x):
        return self.pool(x)
