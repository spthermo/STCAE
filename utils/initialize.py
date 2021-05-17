import torch
import torch.nn
import sys

def initialize_vgg_weights(model, init):
    init_func = None
    if init == "xavier":
        init_func = torch.nn.init.xavier_normal_
    elif init == "kaiming":
        init_func = torch.nn.init.kaiming_normal_
    elif init == "gaussian" or init == "normal":
        init_func = torch.nn.init.normal_
    
    pretrained = torch.load('I:\\vcl_backup\\phd\\sensorimotor2\\affordance_detection\\coding\\vgg16.pth')
    layer_cnt = 0
    for module in model.encoder.modules():
            if isinstance(module, torch.nn.Conv2d):
                layer_name_w = 'features.' + str(layer_cnt) + '.weight'
                layer_name_b = 'features.' + str(layer_cnt) + '.bias'
                module.weight = torch.nn.Parameter(pretrained[layer_name_w])
                if module.bias is not None:
                    module.bias = torch.nn.Parameter(pretrained[layer_name_b])
                if layer_cnt < 16:
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False

                layer_cnt += 2
                if layer_cnt == 4 or layer_cnt == 9 or layer_cnt ==16 or layer_cnt == 23:
                    layer_cnt += 1

    if init_func is not None:
        #TODO: logging /w print or lib
        for module in model.classifier.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) \
                or isinstance(module, torch.nn.ConvTranspose2d):
                    init_func(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()       

        for module in model.convlstm.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) \
                or isinstance(module, torch.nn.ConvTranspose2d):
                    init_func(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()    
    else:
        print("Error when initializing model's weights, {} either doesn't exist or is not a valid initialization function.".format(init), \
            file=sys.stderr)


def initialize_weights(model, init = "xavier"):    
    init_func = None
    if init == "xavier":
        init_func = torch.nn.init.xavier_normal_
    elif init == "kaiming":
        init_func = torch.nn.init.kaiming_normal_
    elif init == "gaussian" or init == "normal":
        init_func = torch.nn.init.normal_
      
    if init_func is not None:
        #TODO: logging /w print or lib
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) \
                or isinstance(module, torch.nn.ConvTranspose2d):
                    init_func(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()       
    else:
        print("Error when initializing model's weights, {} either doesn't exist or is not a valid initialization function.".format(init), \
            file=sys.stderr)