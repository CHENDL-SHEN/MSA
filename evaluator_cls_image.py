from operator import mod
import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import dataset_root

from core.networks import *
from core.datasets import *
import core.models as fcnmodel

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.Q_util import *
from tools.general.visualization import *
from tools.dataset.voc_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.augment_utils import *
from tools.ai.randaugment import *

parser = argparse.ArgumentParser()


def get_params():
    # Dataset
    parser.add_argument('--dataset', default='voc12', type=str, choices=['voc12', 'coco'])
    parser.add_argument('--domain', default='train', type=str)
    parser.add_argument('--Qmodel_path', default='./experiments/models/train_Q_oriimg_lab/2024-xx-xx xx_xx_xx.pth', type=str)  #
    parser.add_argument('--Cmodel_path', default='./experiments/models/train_MSA_VOC_image/2024-xx-xx xx_xx_xx.pth', type=str)  #
    parser.add_argument('--savepng', default=False, type=str2bool)
    parser.add_argument('--savenpy', default=False, type=str2bool)
    parser.add_argument('--sp_cam', default=True, type=str2bool)
    parser.add_argument('--tag', default='evaluate_IMG', type=str)
    parser.add_argument('--tagA', default='DCR_AP_IPC_Q', type=str)
    parser.add_argument('--curtime', default='', type=str)

    args = parser.parse_args()

    return args

class evaluator:
    def __init__(self, dataset='voc12', domain='train', SP_CAM=True, save_np_path=None, savepng_path=None,
                 muti_scale=False, th_list=list(np.arange(0.05, 0.4, 0.05)), refine_list=range(0, 50, 10)) -> None:
        
        args = get_params()
        
        self.C_model = None
        self.Q_model = None
        self.SP_CAM = SP_CAM
        if (muti_scale):
            self.scale_list = [0.5, 1.0, 1.5, 2.0, -0.5, -1, -1.5, -2.0]  # - is flip
        else:
            self.scale_list = [1]  # - is flip

        self.th_list = th_list              
        self.refine_list = refine_list      
        self.parms = []

        for renum in self.refine_list:
            for th in self.th_list:
                self.parms.append((renum, th))  

        class_num = 21 if dataset == 'voc12' else 81

        self.meterlist = [Calculator_For_mIoU(class_num) for x in self.parms]   
        self.save_png_path = savepng_path
        self.save_np_path = save_np_path

        if (self.save_png_path != None):
            if not os.path.exists(self.save_png_path):
                os.mkdir(self.save_png_path)

        if args.dataset == 'voc2012':
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
        else:
            imagenet_mean = [0.471, 0.448, 0.408]
            imagenet_std = [0.234, 0.239, 0.242]

        test_transform = transforms.Compose([
            Normalize_For_Segmentation(imagenet_mean, imagenet_std),
            Transpose_For_Segmentation()
        ])

        if (dataset == 'voc12'):
            valid_dataset = Dataset_For_Evaluation(dataset_root.VOC_ROOT, domain, test_transform, dataset)
        else:
            valid_dataset = Dataset_For_Evaluation(dataset_root.COCO_ROOT, domain, test_transform, 'coco')

        self.valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=True)

    def get_cam(self, images, ids, Qs, beta=10, it=0.6):    
        with torch.no_grad():
            cam_list = []
            _, _, h, w = images.shape   
                                             
            for s, q in zip(self.scale_list, Qs):  
                target_size = (round(h * abs(s)), round(w * abs(s)))    
                scaled_images = F.interpolate(images, target_size, mode='bilinear', align_corners=False)      
                H_, W_ = int(np.ceil(target_size[0] / 16.) * 16), int(np.ceil(target_size[1] / 16.) * 16)      
                scaled_images = F.interpolate(scaled_images, (H_, W_), mode='bilinear', align_corners=False)    

                if (s < 0):
                    scaled_images = torch.flip(scaled_images, dims=[3])
                if (self.SP_CAM):
                    logits, x4 = self.C_model(inputs=scaled_images, probs=q, pcm=beta, th=it)   
                else:
                    logits, x4 = self.C_model(scaled_images, pcm=12)

                cam_list.append(logits)

        return cam_list

    def get_Q(self, images, ids):
        _, _, h, w = images.shape  
        Q_list = []
        affmat_list = []

        for s in self.scale_list:      
            target_size = (round(h * abs(s)), round(w * abs(s)))
            H_, W_ = int(np.ceil(target_size[0] / 16.) * 16), int(np.ceil(target_size[1] / 16.) * 16)  
            scaled_images = F.interpolate(images, (H_, W_), mode='bilinear', align_corners=False)   

            if (s < 0):
                scaled_images = torch.flip(scaled_images, dims=[3])  
            pred = self.Q_model(scaled_images)      

            Q_list.append(pred)
            affmat_list.append(None)

        return Q_list, affmat_list      
    
    def get_mutiscale_cam(self, cam_list, Q_list, affmat_list, refine_time=0):      
        _, _, h, w = cam_list[-1].shape    
        h *= 16
        w *= 16     
                                                                                            
        refine_cam_list = []                                                                
        for cam, Q, affmat, s in zip(cam_list, Q_list, affmat_list, self.scale_list):        
            if (self.SP_CAM):                                                           
                for i in range(refine_time):
                    cam = refine_with_affmat(cam, affmat)
                cam = upfeat(cam, Q, 16, 16)                                              
                # cam=make_cam(cam)
            cam = F.interpolate(cam, (int(h), int(w)), mode='bilinear', align_corners=False) 
            if(s < 0):
                cam = torch.flip(cam, dims=[3])  # ?dims
            refine_cam_list.append(cam)
        refine_cam = torch.sum(torch.stack(refine_cam_list), dim=0)                       

        return refine_cam                                                              
    
    def getbest_miou(self, clear=True):
        iou_list = []
        for parm, meter in zip(self.parms, self.meterlist):
            cur_iou, mIoU_foreground, IoU_list, FP, FN = meter.get(clear=clear, detail=True)
            iou_list.append((cur_iou, parm))
        iou_list.sort(key=lambda x: x[0], reverse=True)
        return iou_list

    def evaluate(self, C_model, Q_model=None, beta=10, ite=0.6):
        self.C_model, self.Q_model = C_model, Q_model
        self.C_model.eval()
        if (self.SP_CAM):
            self.Q_model.eval()
        with torch.no_grad():
            length = len(self.valid_loader)    

            for step, (images, image_ids, tags, gt_masks) in enumerate(self.valid_loader):  
                images = images.cuda()     
                gt_masks = gt_masks.cuda()  
                _, _, h, w = images.shape  

                if (self.SP_CAM):
                    Qs, affmats = self.get_Q(images, image_ids)     
                else:
                    Qs = [images for x in range(len(self.scale_list))]
                    affmats = [None for x in range(len(self.scale_list))]

                cams_list = self.get_cam(images, image_ids, Qs, beta, ite)
                mask = tags.unsqueeze(2).unsqueeze(3).cuda()    
                
                for renum in self.refine_list:  
                    refine_cams = self.get_mutiscale_cam(cams_list, Qs, affmats, renum)  
                    cams = (make_cam(refine_cams) * mask)      
                    cams = F.interpolate(cams, (int(h), int(w)), mode='bilinear', align_corners=False)  
                                                                                             
                    if (self.save_np_path != None):     
                        cams2 = F.interpolate(cams, (int(h), int(w)), mode='bilinear', align_corners=False)
                        np.save(os.path.join(self.save_np_path, image_ids[0] + '.npy'), cams2.cpu().numpy())
                        img_8 = convert_to_tf(images[0])
                        cams[0, 0] = cams[0, 1:].max(0, True)[0]
                        nnn = cams[0, 1:].max(0, True)[0]
                        cams[0, 0] = nnn

                        ttt = cams.argmax(dim=1)[0]
                        ttt = get_colored_mask(ttt.cpu().numpy())
                        ttt = cv2.cvtColor(ttt, cv2.COLOR_RGB2HLS)  #
                        ttt[:, :, 1] = (230 * nnn[0].cpu().numpy()).astype(np.uint8)
                        ttt = cv2.cvtColor(ttt, cv2.COLOR_HLS2BGR)  #

                        saveimg = None
                        aa = True

                        for i in range(1, 21):
                            if (tags[0][i] == 1):
                                ttt = torch.zeros(cams.shape)[0][0]
                                ttt[:, :] = i
                                ttt = get_colored_mask(ttt.cpu().numpy())
                                ttt = cv2.cvtColor(ttt, cv2.COLOR_RGB2HSV)  #
                                ttt[:, :, 2] = (230 * cams[0][i].cpu().numpy()).astype(np.uint8)
                                ttt = cv2.cvtColor(ttt, cv2.COLOR_HSV2BGR)  #

                                if (aa):
                                    saveimg = ttt.astype(np.float32)
                                    aa = False
                                else:
                                    saveimg += ttt.astype(np.float32)

                        saveimg[saveimg > 255] = 255
                        saveimg = saveimg.astype(np.uint8)
                        cv2.imwrite(os.path.join(self.save_np_path, image_ids[0] + '_' + str(i) + '.png'), saveimg)

                        CAM = generate_vis(cams[0].cpu().numpy(), None, img_8, func_label2color=VOClabel2colormap, threshold=None, norm=False)

                        for i in range(21):
                            if (tags[0][i] == 1):
                                save_img = CAM[i].transpose(1, 2, 0) * 255
                                save_img = cv2.cvtColor(save_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                                cv2.imwrite(os.path.join(self.save_np_path, image_ids[0] + '_' + str(i) + '.png'), save_img)

                    if (step == 600) or (step == 200) or step == 100 or step == 1450:
                        print(self.getbest_miou(clear=False))

                    for th in self.th_list:     
                        cams[:, 0] = th  

                        predictions = torch.argmax(cams, dim=1)     
                        for batch_index in range(images.size()[0]):
                            pred_mask = get_numpy_from_tensor(predictions[batch_index])
                            gt_mask = get_numpy_from_tensor(gt_masks[batch_index])
                            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                            self.meterlist[self.parms.index((renum, th))].add(pred_mask, gt_mask) 

                            if (self.save_png_path != None):
                                cur_save_path = os.path.join(self.save_png_path, str(th))

                                if not os.path.exists(cur_save_path):
                                    os.mkdir(cur_save_path)
                                cur_save_path = os.path.join(cur_save_path, str(renum))
                                if not os.path.exists(cur_save_path):
                                    os.mkdir(cur_save_path)

                                img_path = os.path.join(cur_save_path, image_ids[batch_index] + '.png')
                                save_colored_mask(pred_mask, img_path)

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        self.C_model.train()

        if (self.save_png_path != None):
            savetxt_path = os.path.join(self.save_png_path, "result.txt")

            with open(savetxt_path, 'wb') as f:
                for parm, meter in zip(self.parms, self.meterlist):
                    cur_iou = meter.get(clear=False)[-2]
                    f.write('{:>10.2f} {:>10.2f} {:>10.2f}\n'.format(cur_iou, parm[0], parm[1]).encode())

        ret = self.getbest_miou()

        return ret

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = get_params()
    
    tag = args.tag + '/gama%s' % args.gama
    tagA = args.tagA + '_AP_%s' % args.beta

    time_string = time.strftime("%Y-%m-%d %H_%M_%S")
    if (args.curtime == ''):
        args.curtime = time_string

    log_tag = create_directory(f'./experiments/logs/{args.tag}/{tagA}/')
    log_path = log_tag + f'/{args.curtime}.txt'

    if (args.savepng or args.savenpy):
        prediction_tag = create_directory(f'./experiments/predictions/{args.tag}/{tagA}/')
        prediction_path = create_directory(prediction_tag + f'{args.curtime}/')

    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format(args.tag))
    log_func(str(args))

    class_num = 21 if args.dataset == 'voc12' else 81

    if (args.sp_cam):
        model = SP_CAM_Model3('resnet50', num_classes=class_num)
    else:
        model = CAM_Model('resnet50', 21)

    model = model.cuda()
    model.train()
    model.load_state_dict(torch.load(args.Cmodel_path))

    if (args.sp_cam):
        network_data = torch.load("models_ckpt/SpixelNet_bsd_ckpt.tar")
        Q_model = fcnmodel.SpixelNet1l_bn(data=network_data).cuda()
        Q_model = nn.DataParallel(Q_model)
        Q_model.load_state_dict(torch.load(args.Qmodel_path))
        Q_model.eval()
    else:
        Q_model = None

    _savepng_path = None
    _savenpy_path = None

    if (args.savepng):
        _savepng_path = create_directory(prediction_path + 'pseudo/')
    if (args.savenpy):
        _savenpy_path = create_directory(prediction_path + 'camnpy/')

    evaluatorA = evaluator(dataset='voc12', domain=args.domain, muti_scale=True, SP_CAM=args.sp_cam,
                           save_np_path=_savenpy_path, savepng_path=_savepng_path, refine_list=[0],
                           th_list=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])

    ret = evaluatorA.evaluate(model, Q_model, args.beta, args.gama)
    log_func(str(ret))
