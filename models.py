import math
import random
import torch
from torch import nn
import torch.nn.functional as F

class FQN(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, device,
                 dim: int=16, depth: int=3, input_window: int=15, input_scale: int=2, hidden_window: int=5):
        super().__init__()
        self.input = Quanv1d(in_channels=num_channels, out_channels=dim, kernel_size=input_window, padding=(input_window-1)//2,
                             stride=input_scale, dilation=1, device=device)

        def quanv_bn_relu(in_channels, out_channels, hidden_window, dilation):
            return nn.Sequential(
                Quanv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=hidden_window, padding=0,
                        stride=1, dilation=dilation, device=device),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )

        self.hidden_layers = nn.Sequential(
            *[quanv_bn_relu(dim, dim, hidden_window, i+1) for i in range(depth)]
        )

        self.output = Quanv1d(in_channels=dim, out_channels=num_classes, kernel_size=1, padding=0, stride=1, dilation=1,
                              device=device)

    def forward(self, x):
        x = self.input(x)
        x = self.hidden_layers(x)
        x = self.output(x)
        x = F.adaptive_avg_pool1d(x, 1)
        return x.view(x.size(0), -1)
