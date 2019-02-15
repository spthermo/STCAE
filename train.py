
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
import model, utils

# hard-wire the gpu id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--cuda', default=1, action='store_true', help='enables cuda')
parser.add_argument('--ndf', type=int, default=8, help='num of channels in conv layers')
parser.add_argument('--dilation', type=int, default=1, help='dilation value for bottleneck convolutions')
parser.add_argument('--batch_size', type=int, default=8, help='size of batch')
#data
parser.add_argument('--root', type=str, default='../data/samples', help='path to dataset')
parser.add_argument('--json', type=str, default='../data_handler/dataset.json', help='path to dataset json')
#image
parser.add_argument('--crop_size', type=int, default=512, help='Input dimension (e.g. 512x512)')

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

    sor3d = utils.SOR3D(opt.root, opt.json)

    for i in range(len(sor3d)):
        sample, class_id = sor3d[i]

    #TODO: prepare batches from data loader
        
    seq2seg = model.Seq2Seg(opt.crop_size, opt.ndf, opt.dilation).to(device)
    #TODO: prepare train routine

    input = Variable(torch.randn(opt.batch_size, 3, opt.crop_size, opt.crop_size)).cuda()
    out = seq2seg.forward(input)
    print(out.size())

