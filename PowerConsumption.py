import subprocess
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
    ps_command = "& 'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi' --query-gpu=power.draw --format=csv,noheader,nounits"
    result = subprocess.run(["powershell", "-Command", ps_command], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').strip())


def get_average_metric(metric_function, samples=50):
    return sum(metric_function() for _ in range(samples)) / samples


# Measure power while running your neural network task
start_time = time.time()
# ... Your neural network task starts here ...
end_time = time.time()

average_cpu_usage = get_average_metric(get_cpu_usage)
average_memory_usage = get_average_metric(get_memory_usage)
average_gpu_power = get_average_metric(get_gpu_power)

print(f"Duration: {end_time - start_time} seconds")
print(f"Average CPU Usage: {average_cpu_usage}%")
print(f"Average Memory Usage: {average_memory_usage}%")
print(f"Average GPU Power: {average_gpu_power} W")
