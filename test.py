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

if __name__ == '__main__':
    # 载入训练数据
    res_dat_train = '.cache/res_train.npy'
    sample_dat_train = '.cache/sample_train.npy'
    res_dat_test = '.cache/res_test.npy'
    sample_dat_test = '.cache/sample_test.npy'
    ds_train = DataSet(res_dat_train, sample_dat_train)
    da_train = Data(ds_train, 0, 755)
    batch_size = 32
    train_loader = DataLoader(da_train, batch_size=batch_size, shuffle=True)

    # 载入测试数据
    ds_test = DataSet(res_dat_train, sample_dat_train)
    # ds_test = DataSet(res_dat_test, sample_dat_test)
    sample_test, d2, d1 = ds_test.gen_data()
    labels_test = ds_test.wear
    labels_test = t.tensor(labels_test, dtype=t.float64).view(-1, 1)
    sample_test = t.tensor(sample_test, dtype=t.float64)

    # 判断GPU是否可用，将数据载入设备

    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
    print('Device in use:', device)
    labels_test = labels_test.to(device)
    sample_test = sample_test.to(device)

    # 定义网络结构，输入通道数为7，层深默认为18个residual
    net = Net(7, dropout=0).double()
    net.load_state_dict(t.load('net2.pth', map_location=device))

    net.to(device)

    # 计算诊断结果并绘图
    result = []
    ti = []
    st = sample_test.unsqueeze(1)
    for s in st:
        t = time.time()
        r = net(s)
        ti.append(time.time() - t)
        r = r.cpu().detach().numpy().squeeze()

        result.append(r)

    x = np.arange(len(result))
    print(np.mean(ti))
    plt.plot(x, result)
    plt.plot(x, labels_test.cpu().detach().numpy())
    plt.show()