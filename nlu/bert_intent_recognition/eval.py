# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/22 1:36 PM
# @Organization: YQN
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
from torch.utils import data

def eval(model: nn.Module, eval_dataloader: data.DataLoader):
    model.eval()

