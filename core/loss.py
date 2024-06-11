
from weakref import ref
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.Q_util import *
from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from torch.nn.modules.loss import _Loss


class QLoss(_Loss):

  def __init__(self,
               args,
               size_average=None,
               reduce=None,
               relu_t=0.9,
               reduction='mean'):
    super(QLoss, self).__init__(size_average, reduce, reduction)
    self.relu_t=relu_t
    self.relufn =nn.ReLU()
    self.args=args
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()

  def forward(self,prob,labxy_feat,imgids,pos_weight= 0.003, kernel_size=16):


            # this wrt the slic paper who used sqrt of (mse)

            # rgbxy1_feat: B*50+2*H*W
            # output : B*9*H*w
            # NOTE: this loss is only designed for one level structure

            # todo: currently we assume the downsize scale in x,y direction are always same
            S = kernel_size
            m = pos_weight

            b, c, h, w = labxy_feat.shape
            pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
            reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

            loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]

            if (labxy_feat.shape[1]==5):
                loss_map_sem = reconstr_feat[:, :-2, :, :] - labxy_feat[:, :-2, :, :]
                loss_sem = torch.norm(loss_map_sem, p=2, dim=1).sum() / (b * S)
            else:
                # self def cross entropy  -- the official one combined softmax
                logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
                loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b    

            loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

            # empirically we find timing 0.005 tend to better performance
            loss_sem_sum =   0.005 * loss_sem
            loss_pos_sum = 0.005 * loss_pos
            loss_sum =   loss_pos_sum +loss_sem_sum


            return loss_sum, loss_sem_sum, loss_pos_sum
   

class SP_CAM_Loss2(_Loss):
  def __init__(self, args, size_average=None, reduce=None, reduction='mean'):
    super(SP_CAM_Loss2, self).__init__(size_average, reduce, reduction)
    self.args = args
    self.fg_c_num = 20 if args.dataset == 'voc12' else 80
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()

  def forward(self, fg_cam, sailencys):

        b, c, h, w = fg_cam.size()                  
        imgmin_mask = sailencys.sum(1, True) != 0   # sailencys:(b*p,3,h/p,w/p); imgmin_mask:(b*p,1,h/p,w/p);
        sailencys = F.interpolate(sailencys.float(), size=(h, w))  

        bg = 1-torch.max(fg_cam, dim=1, keepdim=True)[0] ** 1     

        nnn = torch.max((1 - bg.detach() * imgmin_mask).view(b, 1, -1), dim=2)[0] > self.args.ig_th   
        nnn2 = torch.max((bg.detach() * imgmin_mask).view(b, 1, -1), dim=2)[0] > self.args.ig_th     
        nnn = nnn * nnn2       
        if (nnn.sum() == 0):
          nnn = torch.ones(nnn.shape).cuda()
        imgmin_mask = nnn.view(b, 1, 1, 1) * imgmin_mask   

        probs = torch.cat([bg, fg_cam], dim=1)             
        probs1 = probs * imgmin_mask                        

        origin_f = F.normalize(sailencys.detach(), dim=1)  
        origin_f = origin_f * imgmin_mask                  

        f_min = pool_feat_2(probs1, origin_f)       
        up_f = up_feat_2(probs1, f_min)                

        sal_loss = F.mse_loss(up_f, origin_f, reduce=False)           
        sal_loss = (sal_loss * imgmin_mask).sum() / (torch.sum(imgmin_mask) + 1e-3)  
        return sal_loss

 