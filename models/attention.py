import torch
import torch.nn as nn
from models.model_utils import Conv2d

class SoftAttention(nn.Module):
    def __init__(self, in_channels):
        super(SoftAttention, self).__init__()
        
        self.conv_1x1 = Conv2d(in_channels, 1, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x, y):
        feature = torch.cat((x,y), 1)
        out = self.conv_1x1(feature)
        return self.activation(out)