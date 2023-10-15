import numpy as np
import pywt
from TCN import TCN
from model import Net
import torch
import time
from MeasureEnergyFunction import get_cpu_usage, get_memory_usage, get_gpu_power

import numpy as np
import torch
import time
import pywt


# Assume get_cpu_usage, get_memory_usage, get_gpu_power are defined as per your previous code snippet

def measure_energy(func, *args, **kwargs):
    cpu_start, mem_start, gpu_start = get_cpu_usage(), get_memory_usage(), get_gpu_power()
    start_time = time.time()

    result = func(*args, **kwargs)

    end_time = time.time()
    cpu_end, mem_end, gpu_end = get_cpu_usage(), get_memory_usage(), get_gpu_power()

    duration = end_time - start_time
    cpu_usage = (cpu_end - cpu_start) * duration
    mem_usage = (mem_end - mem_start) * duration
    gpu_power = (gpu_end - gpu_start) * duration

    return result, (cpu_usage, mem_usage, gpu_power)


def preprocess_data(x):
    x = x[:, :, ::4]
    A2, D2, D1 = pywt.wavedec(x, 'db4', mode='symmetric', level=2, axis=2)
    return A2, D2, D1


if __name__ == '__main__':
    tcn = TCN(1, 1, [4, 8, 16], kernel_size=3, dropout=0)
    res_net = Net(7, dropout=0)

    e0 = []
    for i in range(10):
        x = torch.rand((1, 7, 20000))
        _, energy = measure_energy(preprocess_data, x)
        e0.append(energy)
    e0 = list(np.array(e0).mean(0))

    e1_ = tcn.per_layer_energy(repeated=10)
    e1_ = list(np.array(e1_).mean(0))
    e2_ = res_net.per_layer_energy(repeated=10)
    e2_ = list(np.array(e2_).mean(0))

    # Compute data transmission latencies
    data_size = [7 * 1255, 128 * 626, 128 * 313, 128 * 313, 128 * 157, 128 * 157, 128 * 79, 128 * 79, 128 * 40,
                 128 * 40, 128 * 20, 128 * 20,
                 128 * 10, 128 * 10, 128 * 5, 128 * 5, 128 * 3, 128 * 3, 128 * 2, 128 * 2, 1, 4 * 15, 8 * 15, 16 * 15,
                 1]

    # Combine all energy measurements
    total_repeated_energies = [e0] + e1_ + e2_

    # Compute energy per bit for each layer [optional]
    # Ensure that data_size and total_repeated_energies are of the same length before performing element-wise division.
    energy_per_bit_cpu_usage = [energy[0] / size if size != 0 else float('inf') for energy, size in
                                zip(total_repeated_energies, data_size)]
    energy_per_bit_mem_usage = [energy[1] / size if size != 0 else float('inf') for energy, size in
                                zip(total_repeated_energies, data_size)]
    energy_per_bit_gpu_power = [energy[2] / size if size != 0 else float('inf') for energy, size in
                                zip(total_repeated_energies, data_size)]

    # Print the energy consumption for each task
    print("Data preprocessing energy:", e0)
    print("Layer-wise energy for TCN (repeated):", e1_)
    print("Layer-wise energy for ResNet (repeated):", e2_)

    # Print the total energy for each task [optional]
    print("Total energy per task:", total_repeated_energies)

    # Print the energy per bit for each layer [optional]
    print("Energy per bit (cpu usage) : ", energy_per_bit_cpu_usage)
    print("Energy per bit (mem usage) : ", energy_per_bit_mem_usage)
    print("Energy per bit (gpu usage) : ", energy_per_bit_gpu_power)