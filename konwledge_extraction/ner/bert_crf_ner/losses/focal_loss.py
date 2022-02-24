# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/23 10:14 AM
# @Organization: YQN
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        :param output: [N, CLASS]
        :param target: [N,]
        :return:
        """
        logit = F.softmax(output, dim=1)  # [N,CLASS]
        pt = torch.exp(logit)
        logit = (1 - pt) ** self.gamma * logit  # [N, CLASS]
        loss = F.nll_loss(logit, target, self.weight, ignore_index=self.ignore_index)
        return loss
