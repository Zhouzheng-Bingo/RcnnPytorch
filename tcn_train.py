import torch as t
import wrapt
import time
import tqdm
from TCN import TemporalBlock, TemporalConvNet, TCN
from dataSet import TCNData
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 载入训练数据
    res_dat = '.cache/all_res_dat_num_5000.npy'
    res_data = np.load(res_dat)
    res_data = np.mean(res_data, axis=1)
    res_data = res_data.reshape(3, 315)
    res_train = np.zeros((301, 1, 15))
    for i in range(301):
        res_train[i, 0, :] = res_data[0, i:i + 15]
    data_train = TCNData(res_train[:290], res_train[11:])
    train_loader = DataLoader(data_train, batch_size=4)

    # 判断GPU是否可用，将数据载入设备
    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
    print('Device in use:', device)

    # 定义神经网络
    tcn = TCN(1, 1, [4, 8, 16], kernel_size=3, dropout=0.2).double()
    tcn.load_state_dict(t.load('net3.pth', map_location=device))
    tcn.to(device)

    # 测试

    result = []
    pre = []
    present = []
    for idx, (input_res, output_res) in enumerate(tqdm.tqdm(train_loader)):
        input_res.to(device)
        output_res.to(device)
        print(input_res.size)
        predictions = tcn(input_res)
        for i in range(predictions.size(0)):
            result.append(predictions[i, 0, -1].detach().numpy())
            pre.append(output_res[i, 0, -1])
            present.append(input_res[i, 0, -1])

    result = np.array(result).astype(float)
    pre = np.array(pre).astype(float)
    criterion = t.nn.MSELoss()
    loss = criterion(t.from_numpy(result), t.from_numpy(pre))
    # np.save('.cache/pres.npy', present)
    x = np.arange(290)

    plt.plot(x, result, 'r')
    plt.plot(x, pre, 'b')
    plt.plot(x, present, 'y')
    plt.show()


    # 训练
    '''
    optimizer = t.optim.Adam(tcn.parameters(), lr=0.0005)
    criterion = t.nn.MSELoss()
    num_epochs = 200
    min_train_loss = 100000
    for epoch in range(num_epochs):
        train_loss_sum, batch = 0.0, 0
        for idx, (input_res, output_res) in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            input_res.to(device)
            output_res.to(device)
            predictions = tcn(input_res)
            loss = criterion(predictions, output_res)
            train_loss_sum = train_loss_sum + loss.item()
            batch = idx
            loss.backward()
            optimizer.step()
        train_loss_avg = train_loss_sum / (batch + 1)
        print('Train epoch:{}, train_loss_avg:{}'.format(epoch, train_loss_avg))
        if (train_loss_avg < 1) & (min_train_loss < train_loss_avg):
            print('Exit epoch:{}, train_min_loss:{}'.format(epoch, min_train_loss))
            break
        min_train_loss = min(min_train_loss, train_loss_avg)
    t.save(tcn.state_dict(), 'net3.pth')
    '''