# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/23 10:28 AM
# @Organization: YQN
# @Email: mlshenkai@163.com
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntry(nn.Module):
    def __init__(self, eps=0.1, reduction="mean", ignore_index=-100):
        super(LabelSmoothingCrossEntry, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        :param output: [N,C]
        :param target: [N,]
        :return:
        """
        c = output.size()[-1]
        logit = F.log_softmax(output, dim=-1)
        if self.reduction == "sum":
            loss = -logit.sum()
        else:
            loss = -logit.sum(dim=-1)
            if self.reduction == "mean":
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            logit, target, reduction=self.reduction, ignore_index=self.ignore_index
        )
