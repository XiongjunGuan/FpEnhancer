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


class CodeLoss(torch.nn.Module):

    def __init__(self, beta=0.25):
        super(CodeLoss, self).__init__()
        self.beta = beta

    def forward(self, z_q, z):
        loss = torch.mean((z_q - z.detach())**2) + self.beta * torch.mean(
            (z_q.detach() - z)**2)

        return loss


class SmoothLoss(nn.Module):

    def __init__(self, device=None, eps=1e-6):
        super(SmoothLoss, self).__init__()
        self.eps = eps
        self.smooth_kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).view(1, 1, 3, 3) / 8.0
        if device is not None:
            self.smooth_kernel = self.smooth_kernel.to(device)

    def forward(self, input, mask=None):
        # smoothness
        if mask is None:
            loss = F.conv2d(input,
                            self.smooth_kernel.type_as(input).repeat(
                                1, input.shape[1], 1, 1),
                            padding=1).abs().mean()
        else:
            loss = F.conv2d(input,
                            self.smooth_kernel.type_as(input).repeat(
                                1, input.shape[1], 1, 1),
                            padding=1)
            loss = (loss * mask).sum() / mask.sum().clamp_min(self.eps)
        return loss


class CrossEntropyLoss():

    def __init__(self, eps=1e-6):
        super(CrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, y_true, y_pred, mask=None):
        y_pred = y_pred.clamp(self.eps, 1 - self.eps)
        # weighted cross entropy loss
        lamb_pos, lamb_neg = 1., 1.
        logloss = lamb_pos * y_true * torch.log(y_pred) + lamb_neg * (
            1 - y_true) * torch.log(1 - y_pred)

        if mask is not None:
            logloss = -torch.sum(logloss * mask) / (torch.sum(mask) + self.eps)
        else:
            logloss = -torch.mean(logloss)

        return logloss


class MultiCEFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, eps=1e-6):
        super(MultiCEFocalLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def forward(self, input, target, mask=None, need_softmax=False):
        if need_softmax:
            p = torch.softmax(input, dim=1)
        else:
            p = input.clamp(self.eps, 1.0 - self.eps)

        pt = (target * p).sum(dim=1, keepdim=True)
        log_pt = (target * torch.log(p)).sum(dim=1, keepdim=True)

        loss = -torch.pow(1 - pt, self.gamma) * log_pt

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp_min(self.eps)
        else:
            loss = loss.mean()
        return loss


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
