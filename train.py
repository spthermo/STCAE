import torch
import torch.nn as nn
import datetime
import sys
import opt
import os
import metrics.segmentation_metrics
import metrics.reasoning_metrics

from models import get_model
from options import parse_arguments
from data import SOR3DLoader, SOR3DLoaderParams
from metrics.segmentation_metrics import F1_Score
from utils import NullVisualizer, VisdomVisualizer, initialize_vgg_weights, initialize_weights, generate_gt_heatmap


if __name__ == '__main__':
    print("{} | Torch Version: {}".format(datetime.datetime.now(), torch.__version__))
    args, uknown = parse_arguments(sys.argv)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    torch.manual_seed(667)
    if device.type == 'cuda':
        torch.cuda.manual_seed(667)

    # visdom init
    visualizer = NullVisualizer() if args.visdom is None\
        else VisdomVisualizer(args.name, args.visdom, count=4)
    if args.visdom is None:
        args.visdom_iters = 0

    # data
    train_data_params = SOR3DLoaderParams(root_path = os.path.join(args.train_path, 'train')) 
    train_data_iterator = SOR3DLoader(train_data_params)
    
    train_set = torch.utils.data.DataLoader(train_data_iterator,\
        batch_size = args.batch_size, shuffle=True,\
        num_workers = 0, pin_memory=False)

    val_data_params = SOR3DLoaderParams(root_path = os.path.join(args.train_path, 'val')) 
    val_data_iterator = SOR3DLoader(val_data_params)
    
    val_set = torch.utils.data.DataLoader(val_data_iterator,\
        batch_size = 1, shuffle=False,\
        num_workers = 0, pin_memory=False)
    
    # create & init model
    model_params = {
        'dim': args.crop_size,
        'ndf': args.ndf,
        'affordance_classes': args.action_clasees,
        'ngroups': args.ngroups,
        'nchannels': 3
    }
    encoder_clstm, decoder = get_model(args.model, model_params)
    initialize_vgg_weights(encoder_clstm, args.weight_init)
    encoder_clstm_params = sum(p.numel() for p in encoder_clstm.parameters() if p.requires_grad)
    encoder_clstm.to(device)
    initialize_weights(decoder, args.weight_init)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    decoder.to(device)
    print ('Model params: ', encoder_clstm_params + decoder_params)

    # create and init optimizer
    opt_params = opt.OptimizerParameters(learning_rate=args.lr, momentum=args.momentum,\
        momentum2=args.momentum2, epsilon=args.epsilon)
    optimizer = opt.get_optimizer(args.optimizer, encoder_clstm.parameters(), opt_params)

    opt_params2 = opt.OptimizerParameters(learning_rate=args.lr, momentum=args.momentum,\
        momentum2=args.momentum2, epsilon=args.epsilon)
    optimizer2 = opt.get_optimizer(args.optimizer, decoder.parameters(), opt_params2)
        
    affordance_class_loss = nn.CrossEntropyLoss().to(device)
    
    affordance_seg_loss = nn.NLLLoss(reduction='mean').to(device) #ignore_index=-1
    kld_loss = nn.KLDivLoss(reduction='batchmean')

    f1_score = F1_Score()
    kld_score = nn.KLDivLoss(reduction='batchmean')

    # training loop
    target = torch.zeros(args.batch_size)
    iterations = 0
    for epoch in range(args.epochs):
        print("Training | Epoch: {}".format(epoch))
        encoder_clstm.train()
        decoder.train()
        total_data_num = len(train_set.dataset.fdata)
        for batch_id, batch in enumerate(train_set):
            if batch_id > ((total_data_num // args.batch_size) - 1):
                break
            optimizer.zero_grad()
            for in_batch_cnt in range(args.batch_size):
                target[in_batch_cnt] = batch["frame_001"]["action"][in_batch_cnt].item()
            sequence_action_loss = 0
            for frame in batch:
                frame_action_loss = 0
                attention_mask, out_list, action_pred = encoder_clstm.forward(batch[frame]["color"].to(device)) #, object_pred
                frame_action_loss = affordance_class_loss(action_pred.to(device), target.long().to(device))
                sequence_action_loss += frame_action_loss
            # predictions
            pred_mask, pred_heatmap = decoder.forward(out_list, attention_mask)
            # segmentation loss
            target_mask = batch[frame]["target"]
            seg_loss = affordance_seg_loss(pred_mask, target_mask.to(device))
            affordance_loss = sequence_action_loss / len(batch)
            # reasoning loss
            heat = batch[frame]["heatmap"]
            heat_processed = generate_gt_heatmap(heat, 5)
            reasoning_loss = kld_loss(pred_heatmap, heat_processed.to(device))
            # total loss
            total_loss =  0.1 * reasoning_loss + 0.3 * seg_loss + 0.6 * affordance_loss
            total_loss.backward()
            optimizer.step()
            optimizer2.step()
            iterations += args.batch_size
            print("Epoch: {}, iteration: {}, learning rate: {}, Total Loss: {}\n"\
                    .format(epoch, iterations, optimizer.param_groups[0]['lr'], total_loss.item()))

            #visualization
            if (iterations) % args.visdom_iters == 0:
                visualizer.show_seg_map(torch.exp(pred_heatmap[0]), 'heatmap prediction 1')
                visualizer.show_seg_map(heat_processed[0], 'heatmap gt 1')
                visualizer.show_seg_map(torch.exp(pred_heatmap[1]), 'heatmap prediction 2')
                visualizer.show_seg_map(heat_processed[1], 'heatmap gt 2')
                visualizer.show_seg_map(pred_mask[0].argmax(0), 'segmentation prediction 1') #[target[1].long()]
                visualizer.show_seg_map(target_mask[0], 'segmentation gt 1')
                visualizer.show_seg_map(pred_mask[1].argmax(0), 'segmentation prediction 2') #[target[1].long()]
                visualizer.show_seg_map(target_mask[1], 'segmentation gt 2')
            
            if (iterations + 1) % args.disp_iters == 0:
                visualizer.append_loss(epoch + 1, iterations, total_loss.item(), "total")
                visualizer.append_loss(epoch + 1, iterations, affordance_loss.item(), "affordance")
                visualizer.append_loss(epoch + 1, iterations, seg_loss.item(), "segmentation")
                visualizer.append_loss(epoch + 1, iterations, reasoning_loss.item(), "reasoning")
        
        print("Validation | Epoch: {}".format(epoch))
        encoder_clstm.eval()
        decoder.eval()
        total_jaccard =  0
        total_f1 = 0
        total_kld = 0
        total_data_num = len(val_set.dataset.fdata)
        for batch_id, batch in enumerate(val_set):
            if batch_id > ((total_data_num // args.batch_size) - 1):
                break
            target[0] = batch["frame_001"]["action"][0].item()
            for frame in batch:
                attention_mask, out_list, action_pred = encoder_clstm.forward(batch[frame]["color"].to(device)) #, object_pred
            # predictions
            target_mask = batch[frame]["target"]
            heat = batch[frame]["heatmap"]
            heat_processed = generate_gt_heatmap(heat, 5)
            pred_mask, pred_heatmap = decoder.forward(out_list, attention_mask)
            _, collapsed_mask = torch.max(torch.exp(pred_mask), 1)
            jaccard = metrics.segmentation_metrics.compute_jaccard(collapsed_mask.cpu().detach().numpy().reshape(-1), target_mask.cpu().detach().numpy().reshape(-1))
            
            binary_pred_mask = torch.exp(pred_mask.squeeze(0))
            binary_pred_mask[binary_pred_mask > 0.75] = 1
            binary_pred_mask[binary_pred_mask <= 0.75] = 0
            binary_target_mask = torch.zeros(model_params['affordance_classes'], target_mask.shape[1], target_mask.shape[2])
            for i in range(target_mask.shape[1]):
                for j in range(target_mask.shape[2]):
                    t_class = target_mask[0][i][j]
                    binary_target_mask[t_class][i][j] = 1
            f1 = f1_score(binary_pred_mask.detach().cpu(), binary_target_mask)
            kld = kld_score(pred_heatmap.cpu(), heat_processed)

            total_jaccard += jaccard
            total_f1 += f1
            total_kld += kld

        print("Epoch: {}, IoU: {}, F1: {}, KLD: {}\n"\
                    .format(epoch, total_jaccard / total_data_num, total_f1 / total_data_num, total_kld / total_data_num))

        #save model params
        if epoch in args.save_scheduler:
            opt.save_checkpoint({
                'epoch': epoch,
                'batch_size': args.batch_size,
                'task': args.model,
                'state_dict_en': encoder_clstm.state_dict(),
                'state_dict_de': decoder.state_dict(),
                'optimizer_en': optimizer.state_dict(),
                'optimizer_de': optimizer.state_dict(),
            }, epoch)




