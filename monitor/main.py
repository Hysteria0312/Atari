import time
import pynvml
import matplotlib.pyplot as plt
import numpy as np

# 初始化NVML库
pynvml.nvmlInit()

# 获取GPU数量
device_count = pynvml.nvmlDeviceGetCount()

monitor_duration = 60
sampling_interval = 0.5  # 每隔1秒采样一次

# 存储GPU利用率数据
gpu_utilizations = {i: [] for i in range(device_count)}


def monitor_gpu_utilization():
    start_time = time.time()
    while time.time() - start_time < monitor_duration:
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            gpu_utilizations[i].append(utilization.gpu)

        time.sleep(sampling_interval)

    plot_gpu_utilization()


def plot_gpu_utilization():
    plt.figure(figsize=(10, 6))

    # For each GPU, calculate the moving average of every 5 points
    for i in range(device_count):
        utilization_data = gpu_utilizations[i]

        # Create a simple moving average (5-point average)
        moving_avg = np.convolve(utilization_data, np.ones(5)/5, mode='valid')

        # Plot the moving average
        plt.plot(moving_avg, label=f"GPU {i} (5-point Avg)")

    plt.title("GPU Utilization Over Time (5-point Average)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("GPU Utilization (%)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        monitor_gpu_utilization()
    except KeyboardInterrupt:
        print("Monitoring stopped.")
    finally:
        pynvml.nvmlShutdown()
