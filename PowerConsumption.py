import subprocess
import time
from TCN import TCN
from model import Net
import torch
import pywt

def get_cpu_usage():
    ps_command = "Get-Counter -Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 1 -MaxSamples 1 | " \
                 "Select-Object -ExpandProperty countersamples | Select-Object -ExpandProperty cookedvalue"
    result = subprocess.run(["powershell", "-Command", ps_command], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').strip())


def get_gpu_power():
    ps_command = "& 'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi' --query-gpu=power.draw --format=csv,noheader,nounits"
    result = subprocess.run(["powershell", "-Command", ps_command], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').strip())


def get_memory_usage():
    ps_command = "$memory = Get-WmiObject Win32_OperatingSystem;"\
                 "return [math]::round(($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory)*100/$memory.TotalVisibleMemorySize, 2)"
    result = subprocess.run(["powershell", "-Command", ps_command], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').strip())


def get_average_metric(metric_function, samples=50):
    return sum(metric_function() for _ in range(samples)) / samples


def measure_energy_usage_during_task(task_function, *args, **kwargs):
    start_energy_cpu = get_average_metric(get_cpu_usage)
    start_energy_gpu = get_average_metric(get_gpu_power)
    start_energy_mem = get_average_metric(get_memory_usage)

    task_function(*args, **kwargs)

    end_energy_cpu = get_average_metric(get_cpu_usage)
    end_energy_gpu = get_average_metric(get_gpu_power)
    end_energy_mem = get_average_metric(get_memory_usage)

    energy_usage_cpu = end_energy_cpu - start_energy_cpu
    energy_usage_gpu = end_energy_gpu - start_energy_gpu
    energy_usage_mem = end_energy_mem - start_energy_mem

    return energy_usage_cpu, energy_usage_gpu, energy_usage_mem


if __name__ == '__main__':
    tcn = TCN(1, 1, [4, 8, 16], kernel_size=3, dropout=0)
    res_net = Net(7, dropout=0)

    # Example task function for data pre-processing
    def data_preprocessing_task():
        x = torch.rand((1, 7, 20000))
        x = x[:, :, ::4]
        A2, D2, D1 = pywt.wavedec(x, 'db4', mode='symmetric', level=2, axis=2)

    # Example task function for TCN layer-wise time
    def tcn_layer_time_task():
        tcn.per_layer_time(repeated=50)

    # Example task function for ResNet layer-wise time
    def resnet_layer_time_task():
        res_net.per_layer_time(repeated=50)

    # Measure energy usage during the tasks
    energy_data_prep = measure_energy_usage_during_task(data_preprocessing_task)
    energy_tcn_layer = measure_energy_usage_during_task(tcn_layer_time_task)
    energy_resnet_layer = measure_energy_usage_during_task(resnet_layer_time_task)

    # Printing the energy usage
    print(f"Data preprocessing CPU usage: {energy_data_prep[0]}%, GPU power: {energy_data_prep[1]}W, Memory usage: {energy_data_prep[2]}%")
    print(f"TCN layer-wise time CPU usage: {energy_tcn_layer[0]}%, GPU power: {energy_tcn_layer[1]}W, Memory usage: {energy_tcn_layer[2]}%")
    print(f"ResNet layer-wise time CPU usage: {energy_resnet_layer[0]}%, GPU power: {energy_resnet_layer[1]}W, Memory usage: {energy_resnet_layer[2]}%")
