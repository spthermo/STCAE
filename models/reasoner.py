import torch
import torch.nn as nn

from models.model_utils import ConvReluGN, Conv2d, Interpolate

class Reasoner(nn.Module):
    def __init__(self, ndf, dim, ngroups):
        super(Reasoner, self).__init__()
        self.dim = dim
        self.ndf = ndf
        self.groups = ngroups

        self.attention_conv = Conv2d(self.ndf * 8, 1, kernel_size=1)
        self.upsample_mask3 = Interpolate(size=(dim // 4, dim // 4), mode='bilinear')
        self.upsample_mask2 = Interpolate(size=(dim // 2, dim // 2), mode='bilinear')
        self.upsample_mask1 = Interpolate(size=(dim, dim), mode='bilinear')
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)

        self.tconv7 = ConvReluGN(self.ndf * 8, self.ndf * 4, kernel_size=3, padding=1, groups=self.groups)
        self.upsample3 = Interpolate(size=(dim // 4, dim // 4), mode='bilinear')
        self.conv1x1_3 = Conv2d(self.ndf * 8, self.ndf * 4, kernel_size=1)
        self.tconv5 = ConvReluGN(self.ndf * 4, self.ndf * 2, kernel_size=3, padding=1, groups=self.groups)
        self.upsample2 = Interpolate(size=(dim // 2, dim // 2), mode='bilinear')
        self.conv1x1_2 = Conv2d(self.ndf * 4, self.ndf * 2, kernel_size=1)
        self.tconv3 = ConvReluGN(self.ndf * 2, self.ndf, kernel_size=3, padding=1, groups=self.groups)
        self.upsample1 = Interpolate(size=(dim, dim), mode='bilinear')
        self.conv1x1_1 = Conv2d(self.ndf * 2, self.ndf, kernel_size=1)
        self.conv1x1_0 = Conv2d(self.ndf, 1, kernel_size=3, padding=1)

    def forward(self, x, mask):
        out = self.tconv7(x[3])
        mask = self.upsample_mask3(mask)
        out = self.upsample3(out)
        out = out*mask
        out_cat3 = torch.cat((out, x[2]), 1)
        out = self.conv1x1_3(out_cat3)
        out = self.tconv5(out)
        mask = self.upsample_mask2(mask)
        out = self.upsample2(out)
        out = out * mask
        out_cat2 = torch.cat((out, x[1]), 1)
        out = self.conv1x1_2(out_cat2)
        out = self.tconv3(out)
        mask = self.upsample_mask1(mask)
        out = self.upsample1(out)
        out = out*mask
        out_cat1 = torch.cat((out, x[0]), 1)
        out = self.conv1x1_1(out_cat1)
        out = self.conv1x1_0(out)
        out = self.softmax(out.view(-1, self.dim * self.dim))
        return out.view(-1, self.dim, self.dim).squeeze(1)