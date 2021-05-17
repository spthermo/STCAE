import torch
import torch.nn as nn

from torch.nn import functional as F
from models.encoder import Encoder
from models.segmentor import Segmentor
from models.clstm import ConvLSTM
from models.classifier import Classifier
from models.reasoner import Reasoner
from models.attention import SoftAttention

class EncLSTM(nn.Module):
    def __init__(self, dim, ndf, nclasses, ngroups, nchannels):
        super(EncLSTM, self).__init__()

        self.dim = dim
        self.ndf = ndf
        self.nclasses = nclasses
        self.ngroups = ngroups
        self.nchannels = nchannels

        self.encoder = Encoder(self.ndf, self.ngroups, self.nchannels)
        self.convlstm =  ConvLSTM(input_channels=self.ndf * 8, hidden_channels=[self.ndf * 8], kernel_size=3, step=5, effective_step=[4])
        self.classifier = Classifier(self.ndf * 8, self.nclasses, self.ngroups)
        self.soft_attention = SoftAttention(self.ndf * 16)

    def forward(self, x):
        out_list = []
        out_pre_ds_1, out_pre_ds_2, out_pre_ds_3, out = self.encoder(x)
        out_seq, _ = self.convlstm(out)
        attention_mask = self.soft_attention(out, out_seq[0])
        pred = self.classifier(out_seq[0])
        out_list.append(out_pre_ds_1)
        out_list.append(out_pre_ds_2)
        out_list.append(out_pre_ds_3)
        out_list.append(out_seq[0])
        return attention_mask, out_list, pred


class Decoder(nn.Module):
    def __init__(self, dim, ndf, nclasses, ngroups):
        super(Decoder, self).__init__()

        self.dim = dim
        self.ndf = ndf
        self.nclasses = nclasses
        self.ngroups = ngroups
        
        self.reasoner = Reasoner(self.ndf, self.dim, self.ngroups)
        self.segmentor = Segmentor(self.ndf, self.dim, self.nclasses, self.ngroups)

    def forward(self, x, mask):
        heatmap = self.reasoner(x, mask)
        seg_pred = self.segmentor(x, heatmap)
        return seg_pred, heatmap