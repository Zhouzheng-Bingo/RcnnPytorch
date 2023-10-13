import torch as t
from res_block import Residual, FirstBlock, LastBlock


class Net(t.nn.Module):
    def __init__(self, in_channels, loop_depth=18, dropout=0.5):
        super(Net, self).__init__()
        self.layers = [FirstBlock(in_channels, 64, 128, dropout=dropout)]
        for i in range(loop_depth):
            if i % 2 == 0:
                self.layers.append(Residual(128, 64, 128, stride=2, dropout=dropout))
            else:
                self.layers.append(Residual(128, 64, 128, dropout=dropout))
        self.layers.append(LastBlock(128, 128 * 2, 1))
        self.rcnn = t.nn.Sequential(*self.layers)

    def __len__(self):
        return len(self.layers)
    
    def per_layer_time(self, input_size=[7, 1255], repeated=10):
        pl_time = []
        import time
        for i in range(repeated):
            p_time = []
            x = t.rand((1, input_size[0], input_size[1]))
            for j in range(len(self.layers)):
                start = time.time()
                x = self.rcnn[j](x)
                p_time.append(time.time() - start)
            pl_time.append(p_time)
        return pl_time

    def forward(self, input_tensor):
        out = self.rcnn(input_tensor)
        return out


if __name__ == '__main__':
    net1 = Net(7, dropout=0)
    x = t.rand((1, 7, 1255))
    for i in range(len(net1)):
        x = net1.rcnn[i](x)
        print(x.size())
    '''  
    import matplotlib.pyplot as plt
    import numpy as np
    ti = net1.per_layer_time()
    ti = np.array(ti)
    print(len(net1))
    print(ti.mean(0))
    plt.plot(ti.mean(0))
    plt.show()
    '''