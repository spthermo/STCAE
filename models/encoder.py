import torch.nn as nn

from models.model_utils import ConvReluGN

class Encoder(nn.Module):
    def __init__(self, ndf, ngroups, nchannels):
        super(Encoder, self).__init__()
        
        self.ndf = ndf
        self.groups = ngroups
        self.input_channels = nchannels

        self.conv1 = ConvReluGN(self.input_channels, self.ndf, kernel_size=3, padding=1, groups=0)
        self.conv2 = ConvReluGN(self.ndf, self.ndf, kernel_size=3, padding=1, groups=self.groups)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = ConvReluGN(self.ndf, self.ndf * 2, kernel_size=3, padding=1, groups=self.groups)
        self.conv4 = ConvReluGN(self.ndf * 2, self.ndf * 2, kernel_size=3, padding=1, groups=self.groups)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv5 = ConvReluGN(self.ndf * 2, self.ndf * 4, kernel_size=3, padding=1, groups=self.groups)
        self.conv6 = ConvReluGN(self.ndf * 4, self.ndf * 4, kernel_size=3, padding=1, groups=self.groups)
        self.conv7 = ConvReluGN(self.ndf * 4, self.ndf * 4, kernel_size=3, padding=1, groups=self.groups)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv8 = ConvReluGN(self.ndf * 4, self.ndf * 8, kernel_size=3, padding=1, groups=self.groups)
        self.conv9 = ConvReluGN(self.ndf * 8, self.ndf * 8, kernel_size=3, padding=1, groups=self.groups)
        self.conv10 = ConvReluGN(self.ndf * 8, self.ndf * 8, kernel_size=3, padding=1, groups=self.groups)

    def forward(self, x):
        out = self.conv1(x)
        out_pre_ds_1 = self.conv2(out)
        out = self.pool1(out_pre_ds_1)
        out = self.conv3(out)
        out_pre_ds_2 = self.conv4(out)
        out = self.pool2(out_pre_ds_2)
        out = self.conv5(out)
        out = self.conv6(out)
        out_pre_ds_3 = self.conv7(out)
        out = self.pool3(out_pre_ds_3)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        return out_pre_ds_1, out_pre_ds_2, out_pre_ds_3, out