import torch as t
import wrapt
import time
from res_block import Residual, FirstBlock, LastBlock
from model import Net
from dataSet import DataSet, Data
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pywt


class ToolWearDiagnosis:
    def __init__(self, net_pth, layers, rows, cols):
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        print('Device in use:', self.device)
        self.net = Net(7, dropout=0).double()
        self.net.load_state_dict(t.load(net_pth, map_location=self.device))
        self.net.to(self.device)
        self.layers = layers
        self.rows = rows
        self.cols = cols

    def pre_process(self, inputs):
        A2, D2, D1 = pywt.wavedec(inputs, 'db4', mode='symmetric', level=2, axis=2)
        return A2

    def forward(self, inputs):
        o = t.tensor(inputs, dtype=t.float64)
        o.to(self.device)
        if 0 <= self.layers <= len(self.net):
            for i in range(self.layers):
                o = self.net.rcnn[i](o)
            return o.detach().numpy()
        else:
            print('self.layers out of range.')

    def afterward(self, inputs):
        o = t.tensor(inputs, dtype=t.float64)
        o = o.reshape(1, self.rows, self.cols)
        o.to(self.device)
        if 0 <= self.layers <= len(self.net):
            for i in range(len(self.net)-self.layers):
                o = self.net.rcnn[i+self.layers](o)
            return o.detach().squeeze().numpy()
        else:
            print('self.layers out of range.')


if __name__ == '__main__':
    pro = ToolWearDiagnosis('net2.pth', 13, 128, 10)
    x = t.rand((1, 7, 5000))
    x = pro.pre_process(x)
    print(x.shape[1], x.shape[2])
    x = pro.forward(x)
    print(x.shape[1], x.shape[2])
    x = x.ravel()
    x = pro.afterward(x)
    print(x)