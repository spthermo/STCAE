import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv_layer(x)

class ConvReluGN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=0):
        super().__init__()

        self.groups = groups
        self.out_channels = out_channels
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU(inplace=True)
        if groups > 0:
            self.norm = nn.GroupNorm(groups, out_channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.activation(out)
        out = self.norm(out)

        return out

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