import subprocess


# 温度暂时没用
def get_cpu_temp():
    result = subprocess.run(['vcgencmd', 'measure_temp'], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').split('=')[1].split('\'')[0])


def get_cpu_usage():
    result = subprocess.run(['awk', '-F\'[ :%]*\'', '\'NF{if ($2>0) print $3}\'', '/proc/stat'], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').strip())


def get_memory_usage():
    result = subprocess.run(['free', '|', 'awk', '\'FNR == 3 {print $3/($3+$4)*100}\''], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').strip())


def get_gpu_power():
    return -1.0
