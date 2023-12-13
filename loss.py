import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.nn import functional as F


class BoundaryBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, edge, target, boundary):
        # edge = edge.squeeze(dim=1)
        mask = target
        pos = torch.count_nonzero(boundary)
        num = mask.numel()

        neg_weight = pos / num+0.0
        pos_weight = 1-neg_weight
        weight = torch.zeros_like(boundary)
        weight[boundary!=0.0] = pos_weight
        weight[boundary==0.0] = neg_weight
        loss = F.binary_cross_entropy(edge, boundary, weight, reduction='sum') / num
        return loss


class DualTaskLoss(nn.Module):
    def __init__(self, threshold=0.8):
        super().__init__()
        self.threshold = threshold

    def forward(self, seg, edge, target):
        # edge = edge.squeeze(dim=1)
        logit = F.binary_cross_entropy(seg, target,reduction='none')
        mask = target
        num = (mask[edge > self.threshold]).numel()
        loss = (logit[edge > self.threshold].sum()) / num
        return loss
def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode =='con_ce':
            return self.ConLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, #ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def ConLoss(self, logit, target):
        # loss = torch.mean(torch.sum(-target * torch.log(F.softmax(logit, dim=1)), dim=1))
        # loss = torch.mean(torch.sum(-target * nn.LogSoftmax()(logit), dim=1))
        loss = nn.BCEWithLogitsLoss()(logit, target)
        # loss = nn.BCELoss()(logit, target)
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.25):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, #ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0 
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def resize(self, y_true, h, w):
        b = y_true.shape[0]
        y = np.zeros((b, h,w ,y_true.shape[1]))
        
        y_true = np.array(y_true.cpu())
        for id in range(b):
            y1 = y_true[id,:,:,:].transpose(1,2,0)
            a = cv2.resize(y1, (h, w))
            if a.ndim == 2:
                a=np.expand_dims(a,axis=-1)
            y[id, :, :, :]=a
        y=y.transpose(0,3,1,2)
        return torch.Tensor(y)
        
    def __call__(self, y_true, y_pred,edge_map):
        # the ground_truth map is resized to the resolution of the predicted map during training
        if y_true.shape[2] != y_pred.shape[2] or y_true.shape[3] != y_pred.shape[3]:
            
            y_true = self.resize(y_true, y_pred.shape[2], y_pred.shape[3]).cuda()
#             print(y_true.shape)
        mask = y_true.clone()[:, 0, :, :].unsqueeze(1)
        edge = y_true.clone()[:, 1, :, :].unsqueeze(1)
        a = self.bce_loss(y_pred, mask)
        b = self.soft_dice_loss(mask, y_pred)
        # edge =F.interpolate(edge, scale_factor=0.25, mode='bilinear', align_corners=False)
        edge_=edge_map.detach().clone()
        edge[edge>0]=1
        plt.subplot(2,3,1)
        plt.imshow(edge_.cpu()[0][0].numpy())
        plt.subplot(2, 3, 2)
        plt.imshow(edge.cpu()[0][0].numpy())
        plt.subplot(2, 3, 3)
        plt.imshow(y_pred.detach().cpu()[0][0].numpy())
        plt.show()
        c = BoundaryBCELoss()(edge_map,mask, edge)
        print('loss:',a+b)
        print('edge',c)
        d = DualTaskLoss()(y_pred,edge_map,mask)
        print('task',d)
        return a + b + c+ d
