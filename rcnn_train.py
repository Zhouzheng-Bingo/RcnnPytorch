import torch as t
import wrapt
import time
import tqdm
from res_block import Residual, FirstBlock, LastBlock
from model import Net
from dataSet import DataSet, Data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 载入训练数据
    res_dat_train = '.cache/res_train.npy'
    sample_dat_train = '.cache/sample_train.npy'
    res_dat_test = '.cache/res_test.npy'
    sample_dat_test = '.cache/sample_test.npy'
    ds_train = DataSet(res_dat_train, sample_dat_train)
    da_train = Data(ds_train, 0, 755)
    batch_size = 128
    train_loader = DataLoader(da_train, batch_size=batch_size, shuffle=True)

    # 载入测试数据
    ds_test = DataSet(res_dat_test, sample_dat_test)
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
    net = Net(7, dropout=0.3).double()
    net.load_state_dict(t.load('net2.pth', map_location=device))
    net.to(device)

    # 训练
    optimizer = t.optim.Adam(net.parameters(), lr=0.01)
    criterion = t.nn.MSELoss()
    num_epochs = 2
    print('Total epochs:', num_epochs)
    min_test_loss = 1000
    for epoch in range(num_epochs):
        train_loss_sum, batch = 0.0, 0
        for idx, (data, label) in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            label = label.view(-1, 1).to(device)  # 输出为(batch_size, 1)
            data = data.to(device)
            predictions = net(data)  # 输出为(batch_size, 1)
            loss = criterion(predictions, label)
            train_loss_sum = train_loss_sum + loss.item()
            batch = idx
            loss.backward()
            optimizer.step()
        train_loss_avg = train_loss_sum / (batch + 1)
        test_loss = criterion(net(sample_test), labels_test)
        min_test_loss = min(test_loss, min_test_loss)
        print(
            'Train epoch:{}, train_loss_avg:{}, test_loss:{}, min_test_loss:{}'.format(epoch, train_loss_avg, test_loss,
                                                                                       min_test_loss))
        if test_loss > min_test_loss:
            print('Exit training, train epochs = {}'.format(epoch))
            break
    t.save(net.state_dict(), 'net2.pth')
