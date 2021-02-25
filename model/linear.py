#!/home/chaofan/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2021/02/05 14:57:09
@Author      :chaofan
@version      :1.0
'''
import torch
import torch.nn as nn


# 单层线性网络
class SingleLayerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleLayerModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = self.linear1(x)
        return output


class DoubleLayerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DoubleLayerModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden1 = self.linear1(x)
        activate1 = torch.relu(hidden1)
        output = self.linear2(activate1)
        return output


class MultiLayerModer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLayerModer, self).__init__()
        self.forward_calc = nn.Sequential(nn.Linear(input_dim, 200), nn.ReLU(),
                                          nn.Linear(200, 200), nn.ReLU(),
                                          nn.Linear(200, output_dim))

    def forward(self, x):
        return self.forward_calc(x)
