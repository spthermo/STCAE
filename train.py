
from __future__ import print_function
import argparse
import os
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_, zeros_
#from logger import Logger
import model

# hard-wire the gpu id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--cuda', default=1, action='store_true', help='enables cuda')
parser.add_argument('--ndf', type=int, default=64, help='num of channels in conv layers')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda:0" if opt.cuda else "cpu")

ndf = opt.ndf

if __name__ == '__main__':

    seq2seg = model.Seq2Seg(512, 8, 1).to(device)
    #encoder.init_weights()

    input = Variable(torch.randn(8, 3, 512, 512)).cuda()
    out = seq2seg.forward(input)
    print(out.size())

