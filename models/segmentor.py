import torch
import torch.nn as nn

from torch.nn import functional as F
from models.encoder import Encoder
from models.model_utils import ConvReluGN, Conv2d, Interpolate

class Segmentor(nn.Module):
    def __init__(self, ndf, dim, nclasses, ngroups):
        super(Segmentor, self).__init__()

        self.dim = dim
        self.ndf = ndf
        self.nclasses = nclasses
        self.groups = ngroups

        self.downsample_mask3 = Interpolate(size=(dim // 4, dim // 4), mode='bilinear')
        self.upsample_mask2 = Interpolate(size=(dim // 2, dim // 2), mode='bilinear')
        self.upsample_mask1 = Interpolate(size=(dim, dim), mode='bilinear')
        self.tconv7 = ConvReluGN(self.ndf * 8, self.ndf * 4, kernel_size=3, padding=1, groups=self.groups)
        self.upsample3 = Interpolate(size=(dim // 4, dim // 4), mode='bilinear')
        self.tconv_up3 = ConvReluGN(self.ndf * 4, self.ndf * 4, kernel_size=3, padding=1, groups=self.groups)
        self.conv1x1_3 = Conv2d(self.ndf * 8, self.ndf * 4, kernel_size=1)
        self.tconv5 = ConvReluGN(self.ndf * 4, self.ndf * 2, kernel_size=3, padding=1, groups=self.groups)
        self.upsample2 = Interpolate(size=(dim // 2, dim // 2), mode='bilinear')
        self.tconv_up2 = ConvReluGN(self.ndf * 2, self.ndf * 2, kernel_size=3, padding=1, groups=self.groups)
        self.conv1x1_2 = Conv2d(self.ndf * 4, self.ndf * 2, kernel_size=1)
        self.tconv3 = ConvReluGN(self.ndf * 2, self.ndf, kernel_size=3, padding=1, groups=self.groups)
        self.upsample1 = Interpolate(size=(dim, dim), mode='bilinear')
        self.tconv_up1 = ConvReluGN(self.ndf, self.ndf, kernel_size=3, padding=1, groups=self.groups)
        self.conv1x1_1 = Conv2d(self.ndf * 2, self.ndf, kernel_size=1)
        self.tconv2 = ConvReluGN(self.ndf, self.ndf, kernel_size=3, padding=1, groups=self.groups)
        self.tconv1 = Conv2d(self.ndf, self.nclasses, kernel_size=3, padding=1)

    def forward(self, x, mask):
        out = self.tconv7(x[3])
        mask = self.downsample_mask3(mask.unsqueeze(1))
        out = self.upsample3(out)
        out = out * mask
        out = self.tconv_up3(out)
        out_cat3 = torch.cat((out, x[2]), 1)
        out = self.conv1x1_3(out_cat3)
        out = self.tconv5(out)
        mask = self.upsample_mask2(mask)
        out = self.upsample2(out)
        out = out * mask
        out = self.tconv_up2(out)
        out_cat2 = torch.cat((out, x[1]), 1)
        out = self.conv1x1_2(out_cat2)
        out = self.tconv3(out)
        mask = self.upsample_mask1(mask)
        out = self.upsample1(out)
        out = out*mask
        out = self.tconv_up1(out)
        out_cat1 = torch.cat((out, x[0]), 1)
        out = self.conv1x1_1(out_cat1)
        out = self.tconv2(out)
        out = self.tconv1(out)
        return F.log_softmax(out, dim=1)