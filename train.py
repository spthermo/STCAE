
import argparse
import os
import sys
import datetime
import numpy

import torch
import torchvision

#from logger import Logger
from init import *
import utils, opt

# hard-wire the gpu id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--cuda', default=1, action='store_true', help='enables cuda')
parser.add_argument('--ndf', type=int, default=4, help='num of channels in conv layers')
parser.add_argument('--dilation', type=int, default=1, help='dilation value for bottleneck convolutions')
parser.add_argument('--batch_size', type=int, default=2, help='size of batch')
parser.add_argument('--seed', type=int, default=1337, help="Fixed manual seed, zero means no seeding.")
#model
parser.add_argument('--model', default='default', type=str, help='Model selection argument.')
#training
parser.add_argument('--epochs', type=int, default=10, help="number of epochs")
parser.add_argument('--lr', type=float, default=0.0002, help='Optimization Learning Rate.')
#optimizer
parser.add_argument('--optimizer', type=str, default="adam", help='The optimizer that will be used during training.')
parser.add_argument('--momentum', type=float, default=0.9, help='Optimization Momentum.')
parser.add_argument('--momentum2', type=float, default=0.999, help='Optimization Second Momentum (optional, only used by some optimizers).')
parser.add_argument('--epsilon', type=float, default=1e-8, help='Optimization Epsilon (optional, only used by some optimizers).')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization Weight Decay.')
parser.add_argument("--opt_state", type = str, help = "Path to stored optimizer state file (for continuing training)")
#data
parser.add_argument('--train_path', type=str, help='path to training data')
parser.add_argument('--json', type=str, default='../data_handler/dataset.json', help='path to dataset json')
#image
parser.add_argument('--crop_size', type=int, default=256, help='Input dimension (e.g. 512x512)')

args = parser.parse_args()
#print(args)

device = torch.device("cuda:0" if args.cuda else "cpu")

ndf = args.ndf

if __name__ == '__main__':

    print("{} | Torch Version: {}".format(datetime.datetime.now(), torch.__version__))
    if args.seed > 0:
        print("Set to reproducibility mode with seed: {}".format(args.seed))    
        torch.manual_seed(args.seed)
        numpy.random.seed(args.seed)        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    """
    train_data_params = utils.data_loader_params(root_path = args.train_path) 
    train_data_iterator = utils.data_loader(train_data_params)
    
    train_set = torch.utils.data.DataLoader(train_data_iterator,\
        batch_size = args.batch_size, shuffle=True,\
        num_workers = args.batch_size, pin_memory=False)
    """
    # create & init model
    model_params = {
        'dim': args.crop_size,
        'ndf': args.ndf,
        'dilation': args.dilation
    }

    model = get_model(args.model, model_params).to(device)

    # create and init optimizer
    opt_params = opt.OptimizerParameters(learning_rate=args.lr, momentum=args.momentum,\
        momentum2=args.momentum2, epsilon=args.epsilon)
    optimizer = opt.get_optimizer(args.optimizer, model.parameters(), opt_params)

    if args.opt_state is not None:
        opt_state = torch.load(args.opt_state)
        print("Loading previously saved optimizer state from {}".format(args.opt_state))
        optimizer.load_state_dict(opt_state["optimizer_state_dict"]) 
        
    
    #TODO: prepare train routine

    model.train()
    for epoch in range(args.epochs):
        batch = torch.randn(4, 3, 256, 256).to(device)
        out = model(batch)
        print(out.size())
        """
        print("Training | Epoch: {}".format(epoch))
        opt.adjust_learning_rate(optimizer, epoch)
        for batch_id, batch in enumerate(train_set):
            optimizer.zero_grad()
            for frame in batch:
                out = model.forward(batch[frame]["color"].to(device))
        """

