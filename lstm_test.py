from torch import nn
import torch as t
import time
from thop import profile

if __name__ == '__main__':
    model = nn.LSTM(1, 1, 3)
    input = t.randn(15, 1, 1)
    h0 = t.randn(3, 1, 1)
    c0 = t.randn(3, 1, 1)
    t = time.time()
    for i in range(20):
        output, _ = model(input, (h0, c0))
    t = (time.time()-t)/20
    flops, params = profile(model, inputs=(input, (h0, c0),))

