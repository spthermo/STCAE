import torch
import torch.nn as nn


def softmax():
    return nn.Sequential(
        nn.Softmax2d()
    )

def maxpool(kernel, stride):
    return nn.Sequential(
        nn.MaxPool2d(kernel, stride)
    )

def conv(in_channels, out_channels, kernel, pad, dil):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=pad),
        nn.ELU(True)
    )

def conv_downsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.ELU(True)
    )

def conv_seg(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid()
    )

def conv_1x1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
        nn.ELU(True)
    )


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        """
        Args:
            size: expected size after interpolation
            mode: interpolation type (e.g. bilinear, nearest)
        """
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        out = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        
        return out