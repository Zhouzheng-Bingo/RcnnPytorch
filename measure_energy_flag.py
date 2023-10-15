flag = 1  # 1：在云上 0：在边缘

if flag == 1:
    from measure_energy_function_3090 import get_cpu_usage, get_memory_usage, get_gpu_power
elif flag == 0:
    from measure_energy_function_rasp import get_cpu_usage, get_memory_usage
else:
    print("flag setting error!")