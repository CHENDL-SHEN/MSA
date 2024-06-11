

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.nn.modules.loss import _Loss
import core.models as fcnmodel
import dataset_root
 

import sys
import copy
import shutil
import random
import argparse
import numpy as np
from core.loss import QLoss
import evaluator_cls_image
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.Q_util import *

import time
from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from datetime import datetime


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
start_time = datetime.now().strftime('%Y-%m-%d%H:%M:%S')

parser = argparse.ArgumentParser()


def get_params():
    # Dataset
    parser.add_argument('--dataset', default='voc12', type=str, choices=['voc12', 'coco'])
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--min_image_size', default=320, type=int)
    parser.add_argument('--max_image_size', default=640, type=int)
    parser.add_argument('--cam_npy_path', default='experiments/res/numpy101/', type=str)

    # Network
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--pretrain', default='./models_ckpt/SpixelNet_bsd_ckpt.tar', type=str)
    parser.add_argument('--lr', default=0.00005, type=float)  
    parser.add_argument('--wd', default=4e-5, type=float)
    parser.add_argument('--nesterov', default=True, type=str2bool)

    # Hyperparameter
    parser.add_argument('--th_bg', default=0.05, type=float)
    parser.add_argument('--th_step', default=0.5, type=float)
    parser.add_argument('--th_fg', default=0.05, type=float)
    parser.add_argument('--relu_t', default=0.75, type=float)

    # others
    parser.add_argument('--print_ratio', default=0.1, type=float)
    parser.add_argument('--tag', default='train_Q_oriimg_lab', type=str)
    parser.add_argument('--curtime', default='00', type=str)
    parser.add_argument('--downsize', default=16, type=int)
    
    args = parser.parse_args()
    return args


def main(args):
    
    set_seed(0)
    time_string =  time.strftime("%Y-%m-%d %H_%M_%S")
    args.curtime=time_string
    
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

    log_func()
    Qlossfn = QLoss(args=args)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = [
        RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip_For_Segmentation(),
        Normalize_For_Segmentation_Q(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args.image_size,3),
    ]

    train_transform = transforms.Compose(train_transforms + [Transpose_For_Segmentation()])

    test_transform = transforms.Compose([
        Normalize_For_Segmentation_Q(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])

    meta_dic = read_json('./data/VOC_2012.json')

    data_dir = dataset_root.VOC_ROOT if args.dataset == 'voc12' else dataset_root.COCO_ROOT
    train_domain = 'train_aug' if args.dataset == 'voc12' else 'train'
    test_domain = 'train' if args.dataset == 'voc12' else 'train_1000'

    train_dataset = Dataset_with_LABIMG(data_dir, data_dir + 'JPEGImages/', train_domain, train_transform, _dataset=args.dataset)  
    valid_dataset = Dataset_For_Evaluation(data_dir, test_domain, test_transform, _dataset=args.dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,num_workers=1, shuffle=False, drop_last=True)

    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    # Network
    if(args.pretrain!=''):
        model = fcnmodel.SpixelNet1l_bn().cuda()

    #=========== creat optimizer, we use adam by default ==================
    param_groups = [{'params': model.bias_parameters(), 'weight_decay': 0},
                    {'params': model.weight_parameters(), 'weight_decay': 0}]
    optimizer = torch.optim.Adam(param_groups, args.lr, betas=(0.9, 0.999))
    model.train()
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        the_number_of_gpu = len(use_gpu.split(','))
    except KeyError:
        use_gpu = 0
    log_func(use_gpu)

    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))

    def load_model_fn(): return load_model(
        model, model_path, parallel=the_number_of_gpu > 1)

    def save_model_fn(): return save_model(
        model, model_path, parallel=the_number_of_gpu > 1)

    # Train
    data_dic = {
        'train': [],
        'validation': [],
    }

    train_timer = Timer()  # torch.cuda.device_count()

    eval_timer = Timer()

    train_meter = Average_Meter(['loss', 'sem_loss', 'pos_loss', 'diff_loss'])

    best_valid_mIoU = -1
    spixelID, XY_feat_stack = init_spixel_grid(args)

    def evaluate(loader, _dataset='voc12'):
        model.eval()
        eval_timer.tik()

        class_num = 21 if args.dataset == 'voc12' else 81

        meter = IOUMetric(class_num)

        with torch.no_grad():
            length = len(loader)
            for step, (images, image_ids, tags, gt_masks) in enumerate(loader):
                images = images.cuda()
                _, _, w, h = images.shape
                labels = gt_masks.cuda()
                inuptfeats = labels.clone()
                inuptfeats[inuptfeats == 255] = 0
                inuptfeats = label2one_hot_torch(inuptfeats.unsqueeze(1), C=class_num)
                inuptfeats = F.interpolate(inuptfeats.float(), size=(12, 12), mode='bilinear', align_corners=False)
                inuptfeats = F.interpolate(inuptfeats.float(), size=(w, h), mode='bilinear', align_corners=False)
                prob = model(images)
                inuptfeats, affmat = refine_with_q(inuptfeats, prob, 20, with_aff=True)

                predictions = torch.argmax(inuptfeats, dim=1)
                predictions[predictions == 21] = 255

                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index])

                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    meter.add_batch(pred_mask, gt_mask)

                sys.stdout.write(
                    '\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()

        print(' ')
        model.train()

        _, _, _, _, _, _, _, mIoU, _ = meter.evaluate()
        return mIoU*100, _

    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    torch.autograd.set_detect_anomaly(True)

    for iteration in range(max_iteration):
        images, imgids, tags, sailencys, = train_iterator.get()#sailencys.max()
        tags = tags.cuda()
        images = images.cuda()

        # Inference
        prob = model(images)

        # The part is to calculate losses.
        sailencys = sailencys.cuda() 
        sailencys = sailencys.permute(0, 3, 1, 2) / 255.0
        label_1hot = sailencys
        LABXY_feat_tensor = build_LABXY_feat(
        label_1hot, XY_feat_stack)  # B* (50+2 )* H * W

        reloss = Qlossfn(prob, LABXY_feat_tensor, imgids, pos_weight=0.003)
        loss_s = torch.mean(reloss[0])
        press_loss = torch.mean(reloss[1])
        diff_loss = torch.mean(reloss[2])
        loss = loss_s
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.add({
            'loss': loss.item(),
            'sem_loss': loss_s.item(),
            'pos_loss': press_loss.item(),
            'diff_loss': diff_loss.item(),
        })

        # For Log
        if (iteration + 1) % log_iteration == 0:
            loss, sem_loss, pos_loss, diff_loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))

            data = {
                'iteration': iteration + 1,
                'learning_rate': learning_rate,
                'loss': loss,
                'sem_loss': sem_loss,
                'pos_loss': pos_loss,
                'diff_loss': diff_loss,
                'time': train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)

            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                sem_loss={sem_loss:.4f}, \
                pos_loss={pos_loss:.4f}, \
                diff_loss={diff_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
                     )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)

        # Evaluation
        if (iteration + 1) % (1*val_iteration) == 0:
            save_model_fn()
            mIoU=0
            mIoU, _ = evaluate(valid_loader, _dataset=args.dataset)
            log_func("dwdwd"+str(mIoU))
    
    write_json(data_path, data_dic)
    writer.close()
    print(args.tag)


if __name__ == '__main__':
    params = get_params()
    main(params)

