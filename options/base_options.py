import argparse

def parse_arguments(args):
    usage_text = (
        "Encoder-Decoder PyTorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--cuda', type=str, default='0', help='enables cuda')
    parser.add_argument('--ndf', type=int, default=16, help='num of channels in conv layers')
    parser.add_argument('--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--action_clasees', type=int, default=10, help='Number of action classes')
    parser.add_argument('--batch_size', type=int, default=2, help='size of batch')
    parser.add_argument('--seed', type=int, default=1337, help="Fixed manual seed, zero means no seeding.")
    #model
    parser.add_argument('--model', default='default', type=str, help='Model selection argument.')
    parser.add_argument('--pretrain', default=None, type=str, help='Load weights from pretrained VGG16 for the encoder')
    parser.add_argument('--ngroups', type=int, default=4, help='Number of groups for Group Normalization.')
    #training
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.0002, help='Optimization Learning Rate.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 75], help='Decrease LR at predefined epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--Lambda', type=float, default=100, help='Hyper-parameter for L2 loss weight')
    #optimizer
    parser.add_argument('--optimizer', type=str, default="adam", help='The optimizer that will be used during training.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Optimization Momentum.')
    parser.add_argument('--momentum2', type=float, default=0.999, help='Optimization Second Momentum (optional, only used by some optimizers).')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Optimization Epsilon (optional, only used by some optimizers).')
    parser.add_argument('--weight_decay', type=float, default=0, help='Optimization Weight Decay.')
    parser.add_argument("--opt_state", type = str, help = "Path to stored optimizer state file (for continuing training)")
    #data
    parser.add_argument('--save_scheduler', type=int, nargs='+', default=[4, 14, 24, 34, 49, 74, 99, 149, 199], help='Save model and optimizer state at predefined epochs.')
    parser.add_argument('--train_path', type=str, help='path to training data')
    parser.add_argument('--json', type=str, default='../data_handler/dataset.json', help='path to dataset json')
    #image
    parser.add_argument('--crop_size', type=int, default=300, help='Input dimension (e.g. 512x512)')
    #visualization
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-d','--disp_iters', type=int, default=10, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument("--visdom", type=str, nargs='?', default=None, const="127.0.0.1", help = "Visdom server IP (port defaults to 8097)")
    parser.add_argument("--visdom_iters", type=int, default=10, help = "Iteration interval that results will be reported at the visdom server for visualization.")
        
    return parser.parse_known_args(args)