# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/2/22 7:18 PM
# @Organization: YQN
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.dropout(
            F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training
        )
        x = self.linear2(x)
        return x


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.layer_normal = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_position=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_position], dim=-1))
        x = self.activation(x)
        x = self.layer_normal(x)
        x = self.dense_1(x)
        return x
