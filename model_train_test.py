import sys
import os
import numpy as np
import cv2
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from utils import *
from data_loaders import Ego2Hands
from models.CSM import CSM

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--input_edge', action='store_true', default=False)
    parser.add_argument('--energy', action='store_true', default=False)
    parser.add_argument('--adapt', action='store_true', default=False)
    parser.add_argument('--seq_i', default=-1, type=int,
                        dest='seq_i', help='target sequence index.')
    parser.add_argument('--custom', action='store_true', default=False)
    parser.add_argument('--train_all', action='store_true', default=False)
    parser.add_argument('--test_all', action='store_true', default=False)
    parser.add_argument('--num_models', default=1, type=int,
                        dest='num_models', help='Number of pretrained models to train')
    parser.add_argument('--num_stages', default=-1, type=int,
                        dest='num_stages', help='Number of stages for model if implemented')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--speed_test', action='store_true', default=False)
    parser.add_argument('--save_outputs', action='store_true', default=False)
    parser.add_argument('--model_path', default='models_saved/', type=str,
                        help='model path to save parameters')

    return parser.parse_args()

def construct_model_seg(args, config, energy_status, seq_i = -1, model_i = 1):
    if config.model_name == "CSM":
        if not args.eval:
            model = CSM.CSM_baseline(n_classes = config.num_classes, with_energy = args.energy, input_edge = args.input_edge)
        else:
            model = CSM.CSM_baseline(n_classes = config.num_classes, with_energy = args.energy, input_edge = args.input_edge, n_stages = args.num_stages)
    #elif config.model_name == "?"
    #    Add your models here
    else:
        print("Model {} not implemented.".format(config.model_name))

    print("model_seg {} #params: {}".format(config.model_name, sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if not args.custom:
        if seq_i > 0:
            if args.adapt:
                pretrained_path = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i), '{}_{}_seg_adapt_seq{}_pretrained.pth.tar'.format(config.dataset, config.model_name, seq_i))
            else:
                pretrained_path = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i), '{}_{}_seg_pretrained.pth.tar'.format(config.dataset, config.model_name))
        else:
            if args.adapt:
                if not args.eval:
                    pretrained_path = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i), '{}_{}_seg_pretrained.pth.tar'.format(config.dataset, config.model_name))
                else:
                    pretrained_path = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i), '{}_{}_seg_adapt_seq{}_pretrained.pth.tar'.format(config.dataset, config.model_name, seq_i))
            else:
                pretrained_path = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i), '{}_{}_seg_pretrained.pth.tar'.format(config.dataset, config.model_name))
    else:
        if not args.eval:
            pretrained_path = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, "1", '{}_{}_seg_pretrained.pth.tar'.format(config.dataset, config.model_name))
        else:
            pretrained_path = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, "1", '{}_{}_seg_custom_pretrained.pth.tar'.format(config.dataset, config.model_name))
            
                
    # load pretrained model
    if args.adapt or args.eval or args.custom:
        if not os.path.exists(pretrained_path):
            print("Model not found : {}".format(pretrained_path))
            assert args.custom != True, "Error: Adapting to custom scene requires pretrained model."
            return None
        print("Loading {}".format(pretrained_path))
        state_dict = torch.load(pretrained_path)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            if name not in model.state_dict():
                continue
            else:
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    return model

def train_ego2hands_seg(model_seg, args, config, energy_status, model_i = 1, seq_i = -1):
    print("Training for seg on Ego2Hands dataset.")
    if not args.custom:
        print("Model[{}], seq[{}]".format(model_i, seq_i))
    else:
        print("Model[{}], seq[{}]".format(model_i, "custom"))
    cudnn.benchmark = True
    # train
    train_loader = None
    if config.dataset == 'ego2hands':
        hand_dataset_train = Ego2Hands.Ego2HandsData(args, config, mode = "train_seg", seq_i = seq_i)
        train_loader = torch.utils.data.DataLoader(hand_dataset_train,
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)
    else:
        raise Exception("Error, unknown dataset: {}".format(config.dataset))

    print("Dataset loaded. #instances = {}".format(hand_dataset_train.__len__()))
        
    # For output directory
    if args.save_outputs:
        adapt_status = "_custom" if args.custom else "_adapt_seq{}".format(seq_i) if args.adapt else ""
        out_seg_path = "outputs/{}_{}_edge{}_energy{}_seg_train{}/".format(config.dataset, config.model_name, int(args.input_edge), int(args.energy), adapt_status)
        if not os.path.exists(out_seg_path):
            os.makedirs(out_seg_path)
    
    # For model save directory
    # Create save directory for model
    model_dir_path = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i))
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
        
    if not args.custom:
        if not args.adapt:
            model_save_path = os.path.join(model_dir_path, '{}_{}_seg'.format(config.dataset, config.model_name))
        else:
            model_save_path = os.path.join(model_dir_path, '{}_{}_seg_adapt_seq{}'.format(config.dataset, config.model_name, seq_i))
    else:
        model_save_path = os.path.join(model_dir_path, '{}_{}_seg_custom'.format(config.dataset, config.model_name))
        
    # Criterions
    criterion_seg = nn.CrossEntropyLoss().cuda()
    criterion_mse = nn.MSELoss().cuda()

    # Measures
    iou_val_best = 0.0
    loss_meters = (AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter())
    
    # Training params
    if args.adapt or args.custom:
        lr_rate = config.base_lr_seg_adapt
        step_size = config.policy_parameter_seg_adapt.step_size 
        gamma = config.policy_parameter_seg_adapt.gamma
        iters = 0
        max_iter = config.max_iter_seg_adapt
    else:
        lr_rate = config.base_lr_seg
        step_size = config.policy_parameter_seg.step_size 
        gamma = config.policy_parameter_seg.gamma
        iters = 0
        max_iter = config.max_iter_seg
        
    optimizer_seg = torch.optim.Adam(model_seg.parameters(), lr_rate, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_seg, step_size = step_size, gamma = gamma)
    
    model_seg.train()
    
    while iters < max_iter:
        for i, (img_id_batch, img_orig_tensor, img_tensor, seg_tensor, seg_1_2_tensor, seg_1_4_tensor, energy_gt_tensor, energy_gt_1_2_tensor, energy_gt_1_4_tensor) in enumerate(train_loader):
            iters += 1
            if iters > max_iter:
                break
            img_id = np.reshape(img_id_batch.cpu().data.numpy(), (-1))[0]
            img_batch_size = img_tensor.size(0)
            img_h, img_w = img_tensor.size(2), img_tensor.size(3)

            # Prepare variables
            img_var = torch.autograd.Variable(img_tensor.cuda())
            seg_gt_var = torch.autograd.Variable(seg_tensor.cuda())
            seg_1_2_gt_var = torch.autograd.Variable(seg_1_2_tensor.cuda())
            seg_1_4_gt_var = torch.autograd.Variable(seg_1_4_tensor.cuda())
            energy_gt_var = torch.autograd.Variable(energy_gt_tensor.cuda())
            energy_gt_1_2_var = torch.autograd.Variable(energy_gt_1_2_tensor.cuda())
            energy_gt_1_4_var = torch.autograd.Variable(energy_gt_1_4_tensor.cuda())
               
            # Forward pass
            if "CSM" in config.model_name:
                if args.energy:
                    seg_output1, energy_output1, seg_output_final, energy_output_final = model_seg(img_var)
                    
                    loss_seg1 = criterion_seg(seg_output1, seg_1_4_gt_var)
                    loss_e1 = criterion_mse(energy_output1, energy_gt_1_4_var)
                    
                    loss_seg2 = criterion_seg(seg_output_final, seg_1_2_gt_var)
                    loss_e2 = criterion_mse(energy_output_final, energy_gt_1_2_var)
                    
                    loss_seg_total = loss_seg1 + loss_e1 + loss_seg2 + loss_e2
                    
                    loss_meters[0].update(float(loss_seg1), img_batch_size)
                    loss_meters[1].update(float(loss_e1), img_batch_size)
                    loss_meters[2].update(float(loss_seg2), img_batch_size)
                    loss_meters[3].update(float(loss_e2), img_batch_size)
                else:
                    seg_output1, seg_output_final = model_seg(img_var)
                    
                    loss_seg1 = criterion_seg(seg_output1, seg_1_4_gt_var)
                    loss_seg2 = criterion_seg(seg_output_final, seg_1_2_gt_var)
                    
                    loss_seg_total = loss_seg1 + loss_seg2
                    
                    loss_meters[0].update(float(loss_seg1), img_batch_size)
                    loss_meters[1].update(float(0.0), img_batch_size)
                    loss_meters[2].update(float(loss_seg2), img_batch_size)
                    loss_meters[3].update(float(0.0), img_batch_size)
                    
                    energy_output_final = torch.zeros(img_batch_size, 3, 1, 1)
            #elif config.model_name == "?":
            #    Add your models here

            optimizer_seg.zero_grad()
            loss_seg_total.backward()
            optimizer_seg.step()

            lr_scheduler.step()
            del loss_seg_total

            # Display seg info
            if iters % config.display_interval == 0:
                print('Train Iteration: {} iters'.format(iters))
                print("Learning rate: {}".format(lr_scheduler.get_last_lr()))
                print('Loss_seg_stage1 = {loss.avg: .4f}'.format(loss=loss_meters[0]))
                print('Loss_energy_stage1 = {loss.avg: .4f}'.format(loss=loss_meters[1]))
                print('Loss_seg_stage2 = {loss.avg: .4f}'.format(loss=loss_meters[2]))
                print('Loss_energy_stage2 = {loss.avg: .4f}'.format(loss=loss_meters[3]))

                iou_np = compute_iou(seg_output_final, seg_gt_var)
                print("IoU sample = {}".format(iou_np))

                # Visualize Outputs
                if args.save_outputs:
                    img_orig_np = img_orig_tensor.cpu().data.numpy()
                    img_np = img_tensor.cpu().data.numpy().transpose(0,2,3,1)
                    seg_output_np = seg_output_final.cpu().data.numpy().transpose(0,2,3,1)
                    seg_gt_np = seg_gt_var.cpu().data.numpy()
                    energy_output_np = energy_output_final.cpu().data.numpy().transpose(0,2,3,1)
                    energy_gt_np = energy_gt_var.cpu().data.numpy().transpose(0,2,3,1)

                    for batch_i, (img_orig_i, img_i, seg_output_i, seg_gt_i, energy_output_i, energy_gt_i) in enumerate(zip(img_orig_np, img_np, seg_output_np, seg_gt_np, energy_output_np, energy_gt_np)):
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_img_orig.png".format(iters, batch_i)), (img_orig_i).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_img_grayscale.png".format(iters, batch_i)), (img_i[:,:,0]*256.0 + 128.0).astype(np.uint8))
                        if args.input_edge:
                            cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_img_edge.png".format(iters, batch_i)), (img_i[:,:,1]*256.0 + 128.0).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_seg_output.png".format(iters, batch_i)), np.argmax(seg_output_i, axis=-1).astype(np.uint8)*50)
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_seg_gt.png".format(iters, batch_i)), seg_gt_i*50)
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_l_gt.png".format(iters, batch_i)), (energy_gt_i[:,:,1]*255).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_r_gt.png".format(iters, batch_i)), (energy_gt_i[:,:,2]*255).astype(np.uint8))
                        #cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_bg_gt.png".format(iters, batch_i)), (energy_gt_i[:,:,0]*255).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_l_output.png".format(iters, batch_i)), (energy_output_i[:,:,1]*255).astype(np.uint8))
                        cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_r_output.png".format(iters, batch_i)), (energy_output_i[:,:,2]*255).astype(np.uint8))
                        #cv2.imwrite(os.path.join(out_seg_path, "iter{}_batch{}_energy_bg_output.png".format(iters, batch_i)), (energy_output_i[:,:,0]*255).astype(np.uint8))
                        
                # Clear meters
                for loss_meter in loss_meters:
                    loss_meter.reset()
            
            # Save models
            if iters % config.save_interval == 0:
                if not args.custom:
                    model_is_best = False                  
                    if not args.adapt:
                        iou_meter_val = AverageMeter()
                        for seq_j in range(0, config.num_seqs):
                            seq_j = seq_j + 1
                            iou_seq_j, ap_seq_j, inf_time_j = test_ego2hands_seg(model_seg, seq_j, args, config, energy_status = energy_status)
                            print("Evaluating for esquence {}, IoU = {:.4f}, AP = {:.4f}, inf_time = {:.4f}s".format(seq_j, iou_seq_j, ap_seq_j, inf_time_j))
                            model_seg.train()
                            if iou_seq_j >= 0:
                                iou_meter_val.update(iou_seq_j, 1)
                        print("Mean eval iou = {}".format(iou_meter_val.avg))
                        if iou_meter_val.avg >= iou_val_best:
                            iou_val_best = iou_meter_val.avg
                            model_is_best = True
                            print("New best IoU set")  
                    else:
                        iou_seq_j, ap_seq_j, inf_time_j = test_ego2hands_seg(model_seg, seq_i, args, config, energy_status = energy_status)
                        if iou_seq_j >= iou_val_best:
                            iou_val_best = iou_seq_j
                            model_is_best = True
                        print("Evaluating for esquence {}, IoU = {:.4f}, AP = {:.4f}, inf_time = {:.4f}s".format(seq_i, iou_seq_j, ap_seq_j, inf_time_j))
                        model_seg.train()
                    
                    if model_is_best:
                        print("Saving best model at {}".format(model_save_path))
                        save_model({
                             'iter': iters,
                             'state_dict': model_seg.state_dict(),
                        }, is_best = True, is_last = False, filename = model_save_path)
                    
                print("Saving latest model at {}".format(model_save_path))
                save_model({
                     'iter': iters,
                     'state_dict': model_seg.state_dict(),
                }, is_best = False, is_last = False, filename = model_save_path)
                
    # Save the last model as pretrained model
    save_model({
         'iter': iters,
         'state_dict': model_seg.state_dict(),
    }, is_best = False, is_last = True, filename = model_save_path)

def test_ego2hands_seg(model_seg, seq_i, args, config, energy_status):
    cudnn.benchmark = True
    # train
    test_loader = None
    if config.dataset == 'ego2hands':
        hand_dataset = Ego2Hands.Ego2HandsData(args, config, mode = "test_seg", seq_i = seq_i)
        test_loader = torch.utils.data.DataLoader(hand_dataset,
                batch_size=1, shuffle=False,
                num_workers=1, pin_memory=False)
    else:
        raise Exception("Error, unknown dataset: {}".format(config.dataset))

    if args.save_outputs:
        custom_status = "_custom" if args.custom else ""
        out_seg_path = "outputs/{}_{}_edge{}_energy{}_seg_test{}/".format(config.dataset, config.model_name, int(args.input_edge), int(args.energy), custom_status)
        if not os.path.exists(out_seg_path):
            print("Created outputs directory at {}".format(out_seg_path))
            os.makedirs(out_seg_path)
        
    iou_meter = AverageMeter()
    ap_meter = AverageMeter()
    inf_time_meter = AverageMeter()

    model_seg.eval()    

    for i, (img_orig_tensor, img_test_tensor, seg_gt_tensor, box_l_gt_tensor, box_r_gt_tensor) in enumerate(test_loader):
        img_batch_size = img_test_tensor.size(0)
        img_h, img_w = img_test_tensor.size(2), img_test_tensor.size(3)
        img_test_tensor = img_test_tensor.cuda()
        seg_gt_tensor = seg_gt_tensor.cuda()
        
        if args.speed_test:
            start_time = time.time()
        
        # Forward pass
        if "CSM" in config.model_name:
            if args.energy:
                if model_seg.module.n_stages == 1:
                    seg_output_final, energy_output_final = model_seg(img_test_tensor)
                elif model_seg.module.n_stages == 2:
                    seg_output_final, energy_output_final = model_seg(img_test_tensor)
                else:
                     _, _, seg_output_final, energy_output_final = model_seg(img_test_tensor)
                energy_l_final = energy_output_final[:,1,:,:]
                energy_r_final = energy_output_final[:,2,:,:]
            else:
                if model_seg.module.n_stages == 1:
                    seg_output_final = model_seg(img_test_tensor)
                elif model_seg.module.n_stages == 2:
                    seg_output_final = model_seg(img_test_tensor)
                else:
                    _, seg_output_final = model_seg(img_test_tensor)
                energy_l_final = torch.zeros(img_batch_size, 1, 1, 1)
                energy_r_final = torch.zeros(img_batch_size, 1, 1, 1)
        #elif config.model_name == "?":
            #    Add your models here
 
        if args.speed_test:
            end_time = time.time()
            inf_time_meter.update(end_time - start_time, 1)
        else:
            # Evaluation
            close_kernel_size = 7
            
            seg_output_final = nn.functional.interpolate(seg_output_final, size = (img_h, img_w), mode = 'bilinear', align_corners = True)
            if not args.custom:
                iou_np = compute_iou(seg_output_final, seg_gt_tensor)
                iou_meter.update(np.mean(iou_np), 1)
            
            if args.energy:
                energy_l_final = nn.functional.interpolate(energy_l_final.unsqueeze_(0), size = (img_h, img_w), mode = 'bilinear', align_corners = True)
                energy_r_final = nn.functional.interpolate(energy_r_final.unsqueeze_(0), size = (img_h, img_w), mode = 'bilinear', align_corners = True)
            
                if not args.custom:
                    ap_np = compute_ap(energy_l_final.cpu().data.numpy(), energy_r_final.cpu().data.numpy(), box_l_gt_tensor.cpu().data.numpy(), box_r_gt_tensor.cpu().data.numpy(), close_kernel_size = close_kernel_size)
                    ap_meter.update(np.mean(ap_np), 1)
            
            # Visualize Outputs
            if args.save_outputs:
                img_orig_np = img_orig_tensor.cpu().data.numpy()
                img_test_np = img_test_tensor.cpu().data.numpy().transpose(0,2,3,1)
                seg_output_np = seg_output_final.cpu().data.numpy().transpose(0,2,3,1)
                energy_l_np = energy_l_final.cpu().data.numpy().transpose(0,2,3,1)
                energy_r_np = energy_r_final.cpu().data.numpy().transpose(0,2,3,1)
                box_l_gt_np = box_l_gt_tensor.cpu().data.numpy()
                box_r_gt_np = box_r_gt_tensor.cpu().data.numpy()
                
                custom_status = "_custom" if args.custom else seq_i

                for batch_i, (img_orig_i, img_test_i, seg_output_i, energy_l_i, energy_r_i, box_l_gt_i, box_r_gt_i) in enumerate(zip(img_orig_np, img_test_np, seg_output_np, energy_l_np, energy_r_np, box_l_gt_np, box_r_gt_np)):
                    cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_img_orig.png".format(custom_status, i)), img_orig_i.astype(np.uint8))
                    cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_img_grayscale.png".format(custom_status, i)), ((img_test_i[:,:,0]*256+128.0)).astype(np.uint8))
                    if args.input_edge:
                        cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_img_edge.png".format(custom_status, i)), ((img_test_i[:,:,1]*256+128.0)).astype(np.uint8))
                    seg_output_idx_i = np.argmax(seg_output_i, axis=-1).astype(np.uint8)
                    cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_seg_output.png".format(custom_status, i)), seg_output_idx_i*50)
                    img_vis = visualize_seg_detection(img_orig_i, seg_output_idx_i, energy_l_i, energy_r_i, box_gt_l = None, box_gt_r = None, close_kernel_size = close_kernel_size)
                    cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_img_vis.png".format(custom_status, i)), img_vis.astype(np.uint8))
                    if args.energy:
                        energy_vis_l = (energy_l_i*255).astype(np.uint8)
                        _, energy_vis_l = cv2.threshold(energy_vis_l, 127, 255, cv2.THRESH_BINARY)
                        energy_vis_l = visualize_seg_detection(cv2.cvtColor(energy_vis_l, cv2.COLOR_GRAY2RGB), None, energy_l_i, None, box_gt_l = box_l_gt_i, close_kernel_size = close_kernel_size)
                        cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_energy_l.png".format(custom_status, i)), energy_vis_l)
                        energy_vis_r = (energy_r_i*255).astype(np.uint8)
                        _, energy_vis_r = cv2.threshold(energy_vis_r, 127, 255, cv2.THRESH_BINARY)
                        energy_vis_r = visualize_seg_detection(cv2.cvtColor(energy_vis_r, cv2.COLOR_GRAY2RGB), None, None, energy_r_i, box_gt_r = box_r_gt_i, close_kernel_size = close_kernel_size)
                        cv2.imwrite(os.path.join(out_seg_path, "seq{}_{}_energy_r.png".format(custom_status, i)), energy_vis_r)
            
    return iou_meter.avg, ap_meter.avg, inf_time_meter.avg

def train_model(args):
    config = Config(args.config)
    print("Training...Model {}. Energy {}, Edge {}, Adapt {}".format(config.model_name, args.energy, args.input_edge, args.adapt))
    energy_status = "with_energy" if args.energy else "no_energy"
    if not args.input_edge:
        energy_status = "no_edge"

    if not args.train_all:
        model_seg = construct_model_seg(args, config, energy_status = energy_status)
        if not args.adapt:
            train_ego2hands_seg(model_seg, args, config, energy_status = energy_status)
        else:
            # Check if pretrained model exists
            model_i = 1
            model_save_dir = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i))
            model_pretrained_name = "{}_{}_seg_pretrained.pth.tar".format(config.dataset, config.model_name)
            model_pretrained_path = os.path.join(model_save_dir, model_pretrained_name)
            if not os.path.exists(model_pretrained_path):
                print("No pretrained model found at: {}".format(model_pretrained_path))
                print("Skipping adapting for model {}".format(model_i))
                sys.exit()

            for seq_i in range(0, config.num_seqs):
                seq_i = seq_i + 1
                if args.seq_i != -1 and seq_i != args.seq_i:
                    continue
                model_save_dir = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i))
                model_name = "{}_{}_seg_adapt_seq{}_pretrained.pth.tar".format(config.dataset, config.model_name, str(seq_i))
                if os.path.exists(os.path.join(model_save_dir, model_name)):
                    print("Model exists. Skip training for model {}, sequence {}".format(model_i, seq_i))
                    continue
                train_ego2hands_seg(model_seg, args, config, energy_status = energy_status, seq_i = seq_i)
    else:
        if not args.adapt:
            for model_i in range(args.num_models):
                model_i = model_i + 1
                print("Training model {}".format(model_i))
                model_save_dir = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i))
                model_name = "{}_{}_seg_pretrained.pth.tar".format(config.dataset, config.model_name)
                if os.path.exists(os.path.join(model_save_dir, model_name)):
                    print("Model exists. Skip training for model {}".format(model_i))
                    continue
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                model_seg = construct_model_seg(args, config, energy_status = energy_status, model_i = model_i)
                train_ego2hands_seg(model_seg, args, config, energy_status = energy_status, model_i = model_i)
        else:
            for model_i in range(args.num_models):
                model_i = model_i + 1
                model_save_dir = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i))
                model_pretrained_name = "{}_{}_seg_pretrained.pth.tar".format(config.dataset, config.model_name)
                model_pretrained_path = os.path.join(model_save_dir, model_pretrained_name)
                if not os.path.exists(model_pretrained_path):
                    print("No pretrained model found at: {}".format(model_pretrained_path))
                    print("Skipping adapting for model {}".format(model_i))
                    continue
                    
                for seq_i in range(0, config.num_seqs):
                    seq_i = seq_i + 1
                    model_save_dir = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i))
                    model_name = "{}_{}_seg_adapt_seq{}_pretrained.pth.tar".format(config.dataset, config.model_name, str(seq_i))
                    if os.path.exists(os.path.join(model_save_dir, model_name)):
                        print("Model exists. Skip training for model {}, sequence {}".format(model_i, seq_i))
                        continue
                    model_seg = construct_model_seg(args, config, energy_status = energy_status, model_i = model_i)
                    train_ego2hands_seg(model_seg, args, config, energy_status = energy_status, model_i = model_i, seq_i = seq_i)

def test_model(args):
    config = Config(args.config)
    print("Testing...Model {}. Energy {}, Edge {}, Adapt {}".format(config.model_name, args.energy, args.input_edge, args.adapt))
    energy_status = "with_energy" if args.energy else "no_energy"
    if not args.input_edge:
        energy_status = "no_edge"

    if not args.test_all:
        if not args.custom:
            iou_meter = AverageMeter()
            ap_meter = AverageMeter()
            inf_time_meter = AverageMeter()
            for seq_i in range(config.num_seqs):
                seq_i = seq_i + 1
                if args.seq_i != -1 and seq_i != args.seq_i:
                    continue
                print("Evaluating for esquence {}".format(seq_i))
                if not args.adapt:
                    model_seg = construct_model_seg(args, config, energy_status = energy_status)
                else:
                    model_seg = construct_model_seg(args, config, energy_status = energy_status, model_i = 1, seq_i = seq_i)
                iou_seq_i, ap_seq_i, inf_time_i = test_ego2hands_seg(model_seg, seq_i, args, config, energy_status = energy_status)
                iou_meter.update(iou_seq_i, 1)
                ap_meter.update(ap_seq_i, 1)
                inf_time_meter.update(inf_time_i, 1)
                print("Seq {}, IoU = {:.4f}, AP = {:.4f}, inf_time = {:.4f}s".format(seq_i, iou_seq_i, ap_seq_i, inf_time_i))
            print("Final IoU = {:.4f}, AP = {:.4f}, inf_time = {:.4f}s".format(iou_meter.avg, ap_meter.avg, inf_time_meter.avg))
        else:
            model_seg = construct_model_seg(args, config, energy_status = energy_status)
            _, _, inf_time_custom = test_ego2hands_seg(model_seg, -1, args, config, energy_status = energy_status)
            print("Final IoU = Not available, AP = Not available, inf_time = {:.4f}s".format(inf_time_custom))
            print("Evaluation on custom sequence finished. Outputs saved.")
    else:
        iou_meter = AverageMeter()
        ap_meter = AverageMeter()
        inf_time_meter = AverageMeter()
        for model_i in range(args.num_models):
            iou_meter_i = AverageMeter()
            ap_meter_i = AverageMeter()
            inf_time_meter_i = AverageMeter()
            model_i = model_i + 1
            print("Evaluating for model {}".format(model_i))
            # Check if model is available for model_i
            model_save_dir = os.path.join(args.model_path, config.dataset, config.model_name, energy_status, str(model_i))
            model_pretrained_name = "{}_{}_seg_pretrained.pth.tar".format(config.dataset, config.model_name)
            model_pretrained_path = os.path.join(model_save_dir, model_pretrained_name)
            if not os.path.exists(model_pretrained_path):
                print("No pretrained model found at: {}".format(model_pretrained_path))
                print("Skipping testing for model {}".format(model_i))
                continue
        
            for seq_i in range(config.num_seqs):
                seq_i = seq_i + 1
                if args.seq_i != -1 and seq_i != args.seq_i:
                    continue
                print("Evaluating for esquence {}".format(seq_i))
                if not args.adapt:
                    model_seg = construct_model_seg(args, config, energy_status = energy_status, model_i = model_i)
                else:
                    model_seg = construct_model_seg(args, config, energy_status = energy_status, model_i = model_i, seq_i = seq_i)
                if model_seg is None:
                    print("Model not found for seq {}. Adapting = {}".format(seq_i, args.adapt))
                    continue
                iou_seq_i, ap_seq_i, inf_time_i = test_ego2hands_seg(model_seg, seq_i, args, config, energy_status = energy_status)
                iou_meter_i.update(iou_seq_i, 1)
                ap_meter_i.update(ap_seq_i, 1)
                inf_time_meter_i.update(inf_time_i, 1)
                print("*Seq {}, IoU = {:.4f}, AP = {:.4f}, inf_time = {:.4f}s".format(seq_i, iou_seq_i, ap_seq_i, inf_time_i))
            print("**Model {}: IoU = {:.4f}, AP = {:.4f}, inf_time = {:.4f}s".format(model_i, iou_meter_i.avg, ap_meter_i.avg, inf_time_meter_i.avg))
            iou_meter.update(iou_meter_i.avg, 1)
            ap_meter.update(ap_meter_i.avg, 1)
            inf_time_meter.update(inf_time_meter_i.avg, 1)
            
        print("***Final: IoU = {:.4f}, AP = {:.4f}, inf_time = {:.4f}".format(iou_meter.avg, ap_meter.avg, inf_time_meter.avg))

if __name__ == '__main__':
    args = parse()
    if args.custom:
        args.train_all = False
        args.test_all = False
        args.adapt = False
        args.test_all = False
        args.save_outputs = True
    if args.test_all:
        args.eval = True
    if args.energy:
        args.input_edge = True
        
    if not args.eval:
        train_model(args)
    else:
        test_model(args)