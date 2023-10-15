import subprocess
import pywt
from TCN import TCN
from model import Net
import numpy as np
import torch
import time

def get_cpu_usage():
    ps_command = "Get-Counter -Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 1 -MaxSamples 1 | " \
                 "Select-Object -ExpandProperty countersamples | Select-Object -ExpandProperty cookedvalue"
    result = subprocess.run(["powershell", "-Command", ps_command], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').strip())


def get_memory_usage():
    ps_command = "$memory = Get-WmiObject Win32_OperatingSystem;"\
                 "return [math]::round(($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory)*100/$memory.TotalVisibleMemorySize, 2)"
    result = subprocess.run(["powershell", "-Command", ps_command], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').strip())


def get_gpu_power():
    ps_command = "& 'C:\\Windows\\System32\\nvidia-smi' --query-gpu=power.draw --format=csv,noheader,nounits"
    result = subprocess.run(["powershell", "-Command", ps_command], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').strip())


def get_average_metric(metric_function, samples=50):
    return sum(metric_function() for _ in range(samples)) / samples


# Function to measure energy for a particular task
def measure_energy_during_task(task, *args, **kwargs):
    start_time = time.time()
    cpu_usages, memory_usages, gpu_powers = [], [], []
    for _ in range(50):  # Repeat the task 50 times
        cpu_usages.append(get_cpu_usage())
        memory_usages.append(get_memory_usage())
        gpu_powers.append(get_gpu_power())
        task(*args, **kwargs)  # Execute the task
    end_time = time.time()
    duration = end_time - start_time
    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
    avg_memory_usage = sum(memory_usages) / len(memory_usages)
    avg_gpu_power = sum(gpu_powers) / len(gpu_powers)

    print(f"Duration: {duration} seconds")
    print(f"Average CPU Usage: {avg_cpu_usage}%")
    print(f"Average Memory Usage: {avg_memory_usage}%")
    print(f"Average GPU Power: {avg_gpu_power} W")

# Example task: Data Preprocessing
def data_preprocessing(x):
    x = x[:, :, ::4]
    A2, D2, D1 = pywt.wavedec(x, 'db4', mode='symmetric', level=2, axis=2)

# Example task: Model Inference (Using TCN as an example)
def model_inference(model, x):
    _ = model(x)


if __name__ == '__main__':
    # Initialization and data generation
    tcn = TCN(1, 1, [4, 8, 16], kernel_size=3, dropout=0)
    res_net = Net(7, dropout=0)
    x = torch.rand((1, 7, 20000))
    x_tcn = torch.rand((1, 1, 15))  # Input for TCN: [batch_size, input_size,  sequence_length]
    x_net = torch.rand((1, 7, 1255))  # Input for Net: [batch_size, in_channels, sequence_length]

    # Measure energy during data preprocessing
    print("Measuring during Data Preprocessing...")
    measure_energy_during_task(data_preprocessing, x)

    # Measure energy during TCN model inference
    print("Measuring during TCN Model Inference...")
    measure_energy_during_task(model_inference, tcn, x_tcn)

    # Measure energy during ResNet model inference
    print("Measuring during ResNet Model Inference...")
    measure_energy_during_task(model_inference, res_net, x_net)