from TCN import TCN
from model import Net
import numpy as np
import torch
import time
import pywt


if __name__ == '__main__':
    tcn = TCN(1, 1, [4, 8, 16], kernel_size=3, dropout=0)
    res_net = Net(7, dropout=0)

    t0 = []
    for i in range(10):
        x = torch.rand((1, 7, 20000))
        t = time.time()
        x = x[:, :, ::4]
        A2, D2, D1 = pywt.wavedec(x, 'db4', mode='symmetric', level=2, axis=2)
        t0.append(time.time() - t)
    t0 = np.array(t0)
    t0 = list([np.mean(t0)])

    t1_ = tcn.per_layer_time(repeated=50)
    t1_ = list(np.array(t1_).mean(0))
    t2_ = res_net.per_layer_time(repeated=50)
    t2_ = list(np.array(t2_).mean(0))

    # Compute data transmission latencies
    data_size = [7 * 1255, 128 * 626, 128 * 313, 128 * 313, 128 * 157, 128 * 157, 128 * 79, 128 * 79, 128 * 40,
                 128 * 40, 128 * 20, 128 * 20,
                 128 * 10, 128 * 10, 128 * 5, 128 * 5, 128 * 3, 128 * 3, 128 * 2, 128 * 2, 1, 4 * 15, 8 * 15, 16 * 15,
                 1]

    # print("data size:", data_size)
    data_t = np.array(data_size, dtype=np.float64) * 4 * 8 * 1000 / (40 * 1024 * 1024)
    data_t_ = np.array(data_size, dtype=np.float64) * 4 * 8 * 1000 / (50 * 1024 * 1024)

    # Combine all latencies
    total_repeated_latencies = t0 + t1_ + t2_

    # Compute throughput for each layer
    throughputs = [size / latency if latency != 0 else float('inf') for size, latency in zip(data_size, total_repeated_latencies)]

    # Print the latencies for each task
    print("Data preprocessing latencies:", t0)
    print("Layer-wise latencies for TCN (repeated):", t1_)
    print("Layer-wise latencies for ResNet (repeated):", t2_)

    # Print the Data transmission latencies for each task
    print("Data transmission latencies (edge):", data_t)
    print("Data transmission latencies (server):", data_t_)

    # Print the throughput for each layer
    print("Throughput:", throughputs)
