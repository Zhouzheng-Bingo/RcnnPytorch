import torch as t
import torch.nn as nn


class FirstBlock(nn.Module):
    def __init__(self, channels_1, channels_2, channels_3, kernel_size=3, padding=1, dropout=0.5):
        super(FirstBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels_1, channels_3, kernel_size=5),
            nn.BatchNorm1d(channels_3),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels_3, channels_2, kernel_size=1),
            nn.BatchNorm1d(channels_2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels_2, channels_3, kernel_size, stride=2, padding=padding)
        )
        self.max_pool = nn.MaxPool1d(kernel_size=1, stride=2)

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        out = self.max_pool(o1) + o2
        return out


class Residual(nn.Module):
    def __init__(self, channels_1, channels_2, channels_3, kernel_size=3, stride=1, padding=1, dropout=0.5):
        super(Residual, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm1d(channels_1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels_1, channels_2, kernel_size, padding=padding),
            nn.BatchNorm1d(channels_2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels_2, channels_3, kernel_size, stride=stride, padding=padding)
        )
        if (channels_1 != channels_3) or (stride != 1):
            self.conv1x1 = nn.Conv1d(channels_1, channels_3, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, x):
        out = self.residual(x)
        if self.conv1x1:
            x = self.conv1x1(x)
        out = out + x
        return out


class LastBlock(nn.Module):
    def __init__(self, in_channel, data_size, output_dim):
        super(LastBlock, self).__init__()
        self.bn = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.dense = nn.Sequential(
            nn.Linear(data_size, output_dim)
        )

    def forward(self, x):
        o1 = self.bn(x)
        o2 = t.flatten(o1, 1)
        out = self.dense(o2)
        return out


if __name__ == '__main__':
    net1 = FirstBlock(7, 32, 64)
    net2 = Residual(64, 64, 128, stride=2)
    net3 = Residual(128, 64, 128)

    x = t.rand((1, 7, 100))
    y = net1(x)
    z = net2(y)
    z = net3(z)
    net4 = LastBlock(128, z.size(-1)*128, 1)
    z1 = net4(z)
    print(net1, net2,  net4)
    print(y.size(), z.size(), z1.size())
