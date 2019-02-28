import torch
import torch.nn as nn
from model_utils import *

class Pretrained(nn.Module):
    def __init__(self, ndf, dil, nclasses):
        super(Encoder, self).__init__()

        self.ndf = ndf
        self.dil = dil
        self.nclasses = nclasses

        self.conv1 = conv(3, self.ndf, kernel=7, pad=3, dil=1)
        self.conv2 = conv(self.ndf, self.ndf, kernel=5, pad=2, dil=1)
        self.pool1 = maxpool(kernel=2, stride=2)
        self.conv3 = conv(self.ndf, self.ndf * 2, kernel=3, pad=1, dil=self.dil)
        self.conv4 = conv(self.ndf * 2, self.ndf * 2, kernel=3, pad=1, dil=self.dil)
        self.conv5 = conv(self.ndf * 2, self.ndf * 2, kernel=3, pad=1, dil=self.dil)
        self.pool2 = maxpool(kernel=2, stride=2)
        self.conv6 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=self.dil)
        self.conv7 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=self.dil)
        self.conv8 = conv(self.ndf * 4, self.ndf * 4, kernel=3, pad=1, dil=self.dil)
        self.pool3 = maxpool(kernel=2, stride=2)
        self.conv9 = conv(self.ndf * 4, self.ndf * 8, kernel=3, pad=1, dil=self.dil)
        self.conv10 = conv(self.ndf * 8, self.ndf * 8, kernel=3, pad=1, dil=self.dil)
        self.conv11 = conv(self.ndf * 8, self.ndf * 8, kernel=3, pad=1, dil=self.dil)
        self.pool4 = avgpool(kernel=32, stride=1)
        self.conv12 = conv(self.ndf * 8, self.ndf * 8, kernel=1, pad=0, dil=1)
        self.conv13 = conv(self.ndf * 8, self.ndf * 8, kernel=1, pad=0, dil=1)
        self.conv14 = conv(self.ndf * 8, self.nclasses, kernel=1, pad=0, dil=1)
        self.softmax = softmax()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        out_pre_ds_1 = self.pool1(out)
        out = self.conv3(out_pre_ds_1)
        out = self.conv4(out)
        out = self.conv5(out)
        out_pre_ds_2 = self.pool2(out)
        out = self.conv6(out_pre_ds_2)
        out = self.conv7(out)
        out = self.conv8(out)
        out_pre_ds_3 = self.pool3(out)
        out = self.conv9(out_pre_ds_3)
        out = self.conv10(out)
        out = self.conv11(out)
        out_pre_ds_4 = self.pool4(out)
        out = self.conv12(out_pre_ds_4)
        out = self.conv13(out)
        out = self.conv14(out)
        out = self.softmax(out)


        return out_pre_ds_1, out_pre_ds_2, out_pre_ds_3, out
