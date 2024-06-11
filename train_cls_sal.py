

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from cv2 import LMEDS, Tonemap, log, polarToCart
import numpy as np
from datetime import datetime
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from imageio import imsave

import evaluator_cls_image
import dataset_root

from core.networks import *
from core.loss import SP_CAM_Loss2
from core.datasets import *
import core.models as fcnmodel

from tools.general.Q_util import *
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.augment_utils import *
from tools.ai.randaugment import *

import nni
from nni.utils import merge_parameter
from nni.experiment import Experiment

def get_params():
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument('--num_workers', default=8, type=int) 
    parser.add_argument('--dataset', default='voc12', type=str, choices=['voc12', 'coco'])
    parser.add_argument('--image_size', default=480, type=int)
    parser.add_argument('--min_image_size', default=320, type=int)
    parser.add_argument('--max_image_size', default=640, type=int)
    parser.add_argument('--seed', default=0, type=int)
    
    # Network
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=12, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--wd', default=4e-5, type=float)
    parser.add_argument('--nesterov', default=True, type=str2bool)
    
    # Hyperparameter
    parser.add_argument('--alpha', default=0.3, type=float)     # Loss Balance Coefficient lambda
    parser.add_argument('--clamp_rate', default=0.001, type=float)
    parser.add_argument('--ig_th', default=0.01, type=float)
    parser.add_argument('--patch_number', default=1, type=int)  # The Number of Patch P in IPC Loss
    parser.add_argument('--beta', default=8, type=float)        # the Number of iterations in AP Module
    parser.add_argument('--th', default=0.6, type=float)        # Affinity Threshold gama in AP Module
    
    # others
    parser.add_argument('--SP_CAM', default=True, type=str2bool)  
    parser.add_argument('--Qmodelpath', default='./models_ckpt/Q_model_final.pth', type=str)
    parser.add_argument('--print_ratio', default=0.1, type=float)
    parser.add_argument('--tag', default='train_MSA_VOC_sal', type=str)
    parser.add_argument('--curtime', default='00', type=str)

    args, _ = parser.parse_known_args()
    return args


def main(args):
    set_seed(args.seed)

    # Arguments
    time_string = time.strftime("%Y-%m-%d %H_%M_%S")
    args.curtime = time_string
    print(args.tag)
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/{args.curtime}/')   
    log_tag = create_directory(f'./experiments/logs/{args.tag}/')
    data_tag = create_directory(f'./experiments/data/{args.tag}/')
    model_tag = create_directory(f'./experiments/models/{args.tag}/')
    log_path = log_tag + f'/{args.curtime}.txt'
    data_path = data_tag + f'/{args.curtime}.json'
    model_path = model_tag + f'/{args.curtime}.pth'
    
    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format(args.tag))
    log_func(str(args))
    
    # Transform, Dataset, DataLoader
    if args.dataset == 'voc2012':
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
    else:
        imagenet_mean = [0.471, 0.448, 0.408]
        imagenet_std = [0.234, 0.239, 0.242]
    
    train_transforms = [
        RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip_For_Segmentation(),
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args.image_size),
    ]
    
    train_transform = transforms.Compose(train_transforms + [Transpose_For_Segmentation()])

    domain='train_aug'
    if(args.dataset=='coco'):
         domain='train'

    data_dir = dataset_root.VOC_ROOT if args.dataset == 'voc12' else dataset_root.COCO_ROOT
    saliency_dir = dataset_root.VOC_SAL_ROOT if args.dataset == 'voc12' else dataset_root.COCO_SAL_ROOT
    
    train_dataset = Dataset_with_SAL(data_dir, saliency_dir, domain, train_transform, _dataset=args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    
    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] train_transform is {}'.format(train_transform))
    

    val_iteration = int(len(train_loader))
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration
    
    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    # Network
    if(args.SP_CAM):
        model = SP_CAM_Model3(args.backbone, num_classes=21 if args.dataset == 'voc12' else 81)
    else:
        model = CAM_Model(args.backbone, num_classes=21 if args.dataset == 'voc12' else 81)
    
    model = model.cuda()
    model.train()
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()
    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'
    log_func(use_gpu)
    the_number_of_gpu = len(use_gpu.split(','))

    
    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    val_domain = 'train_600' if args.dataset == 'voc12' else 'train_1000'
    if(args.SP_CAM):
        evaluatorA = evaluator_cls_image.evaluator(args.dataset, domain=val_domain, SP_CAM=True, refine_list=[0], th_list=[0.15, 0.2, 0.3])
    else:
        evaluatorA = evaluator_cls_image.evaluator(args.dataset, domain=val_domain, SP_CAM=False, refine_list=[0])

    # Loss, Optimizer
    param_groups = model.get_parameter_groups()
    params = [
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
        ]
    optimizer = PolyOptimizer(params, lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)

    if the_number_of_gpu > 1:
        log_func ('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    if(args.SP_CAM):
        network_data = torch.load("./models_ckpt/SpixelNet_bsd_ckpt.tar")
        Q_model = fcnmodel.SpixelNet1l_bn(data=network_data).cuda()
        Q_model = nn.DataParallel(Q_model)
        Q_model.load_state_dict(torch.load(args.Qmodelpath))
        Q_model.eval()
    else:
        Q_model=None

    lossfn = SP_CAM_Loss2(args=args)
    lossfn = torch.nn.DataParallel(lossfn).cuda()

    # Train
    data_dic = {
        'train' : [],
        'validation' : [],
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss', 'cls_loss', 'sal_loss'])

    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    torch.autograd.set_detect_anomaly(True)
    best_valid_mIoU =-1
    for iteration in range(max_iteration):
        images, imgids, labels, sailencys = train_iterator.get()
        images = images.cuda()
        labels = labels.cuda()
        sailencys = sailencys.cuda().view(sailencys.shape[0], 1, sailencys.shape[1], sailencys.shape[2])/255.0
        prob = None

        if(args.SP_CAM):
            prob = Q_model(images)

        if(args.SP_CAM):
            logits, logitsmin = model(images, prob)
        else:
            logits, logitsmin = model(images)

        b, c, h, w = logits.shape
        imgmin_mask = F.interpolate(images.float(), size=(h, w))
        imgmin_mask = imgmin_mask.float().sum(dim=1, keepdim=True) != 0
        
        if(args.SP_CAM):
            x4 = poolfeat(sailencys.float(), prob, 16, 16).cuda()   
        else:
            x4 = F.interpolate(sailencys.float(), size=(h, w))  #x4[0,0,11,10]

        x4 = torch.cat([x4, 1-x4], dim=1)
        b, c, h, w = logits.size()
        tagpred = logitsmin  #
        cls_loss = F.multilabel_soft_margin_loss(tagpred[:, 1:].view(tagpred.size(0), -1), labels[:, 1:])
        mask = labels[:, :].unsqueeze(2).unsqueeze(3).cuda()
        fg_cam = make_cam(logits[:, 1:])*mask[:, 1:]
        target_feat = x4.detach() * imgmin_mask     
        target_feat_tile = tile_features(target_feat, args.patch_number)
        fg_cam_tile = tile_features(fg_cam, args.patch_number)
        sal_loss = lossfn(fg_cam_tile, target_feat_tile).mean() * args.alpha   
        loss = cls_loss + sal_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        train_meter.add({
            'loss': loss.item(), 
            'sal_loss': sal_loss.item(), 
            'cls_loss': cls_loss.item()
        })
        
        # For Log
        if (iteration + 1) % log_iteration == 0:
            loss, cls_loss, sal_loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration': iteration + 1,
                'learning_rate': learning_rate,
                'loss': loss,
                'cls_loss': cls_loss, 
                'sal_loss': sal_loss, 
                'time': train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                cls_loss={cls_loss:.4f}, \
                sal_loss={sal_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        
        # Evaluation
        if (iteration + 1) % val_iteration == 0:
            mIoU, para = evaluatorA.evaluate(model, Q_model, beta=args.beta, ite=args.th)[0]
            nni.report_intermediate_result(mIoU)
            
            if(mIoU < 35):
                log_func('miou is too low'+str(mIoU))

            refine_num, threshold = para
            if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                best_valid_mIoU = mIoU
                if(mIoU > 22):
                    save_model_fn()
                    log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'threshold' : threshold,
                'refine_num' : refine_num,
                'mIoU' : mIoU,
                'best_valid_mIoU' : best_valid_mIoU,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                mIoU={mIoU:.2f}%, \
                best_valid_mIoU={best_valid_mIoU:.2f}%, \
                threshold={threshold:.2f}%,\
                refine_num={refine_num:.0f},\
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Evaluation/threshold', threshold, iteration)
            writer.add_scalar('Evaluation/mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_valid_mIoU', best_valid_mIoU, iteration)
    nni.report_final_result(best_valid_mIoU)
    
    write_json(data_path, data_dic)
    writer.close()

if __name__ == '__main__':
        import nni 
    
        args =get_params()
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        args = vars(merge_parameter(args, tuner_params))
        args = DotDict(args)
        main(args)




        