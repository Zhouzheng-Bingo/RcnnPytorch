import subprocess


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
