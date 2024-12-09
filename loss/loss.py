'''
Description: 
Author: Xiongjun Guan
Date: 2024-12-09 14:55:57
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-12-09 14:56:19

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-6


class FinalLoss(torch.nn.Module):

    def __init__(self):
        super(FinalLoss, self).__init__()
        self.focal_loss = BinaryCEFocalLoss(gamma=2, eps=eps)

    def forward(self, y_pred, y_true):
        # clip
        y_pred = y_pred.clamp(eps, 1 - eps)

        # focal loss
        focal_loss = self.focal_loss(y_pred, y_true)

        return focal_loss


class BinaryCEFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, eps=1e-6):
        super(BinaryCEFocalLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def forward(self, input, target, mask=None, need_sigmoid=False):
        if need_sigmoid:
            p = torch.sigmoid(input, dim=1)
        else:
            p = input.clamp(self.eps, 1.0 - self.eps)

        p = torch.cat((p, 1 - p), 1)
        target = torch.cat((target, 1 - target), 1)

        pt = (target * p).sum(dim=1, keepdim=True)
        log_pt = (target * torch.log(p)).sum(dim=1, keepdim=True)

        loss = -torch.pow(1 - pt, self.gamma) * log_pt

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp_min(self.eps)
        else:
            loss = loss.mean()
        return loss
