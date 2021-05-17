import torch.nn as nn

from torch.nn import functional as F
from models.model_utils import ConvReluGN, Conv2d, Interpolate

class Classifier(nn.Module):
    def __init__(self, ndf, nclasses, ngroups):
        super(Classifier, self).__init__()

        self.ndf = ndf
        self.nclasses = nclasses
        self.groups = ngroups

        self.avg = nn.AvgPool2d(37, 1)
        self.fconv1 = nn.Conv2d(self.ndf, self.ndf, 1, 1)
        self.activ1 = nn.ReLU(True)
        self.fconv2 = nn.Conv2d(self.ndf, self.ndf, 1, 1)
        self.activ2 = nn.ReLU(True)
        self.fconv3 = nn.Conv2d(self.ndf, self.nclasses, 1, 1)
        
    def forward(self, x):
        out = self.avg(x)
        out = self.fconv1(out)
        out = self.activ1(out)
        out = self.fconv2(out)
        out = self.activ2(out)
        out = self.fconv3(out)
        pred = F.log_softmax(out, dim=1)

        return pred.squeeze()