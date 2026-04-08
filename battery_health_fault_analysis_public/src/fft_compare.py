import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

# 读取数据
df = pd.read_csv('Task3_Raw_Battery_Signal_Data.csv')

# 获取所有可用的 sample_id 列表
available_samples = df['sample_id'].unique()

# 让用户输入要查看的 sample_id
while True:
    try:
        user_input = input("请输入要查看的 sample_id (0-99 或 100-199): ")
        target_sample = int(user_input)
        
        if not (0 <= target_sample <= 199):
            print("错误：请输入 0-199 之间的整数！")
            continue
        
        # 确定要查询的 sample_id 对
        if 0 <= target_sample <= 99:
            sample_id_pair = [target_sample, target_sample + 100]
        else:
            sample_id_pair = [target_sample - 100, target_sample]
        
        # 检查这两个 sample_id 是否都存在
        valid_samples = [sid for sid in sample_id_pair if sid in available_samples]
        if len(valid_samples) < 2:
            print(f"错误：需要两个 sample_id 进行对比，但 {sample_id_pair} 中只有 {valid_samples} 存在！")
            continue
        
        break
    except ValueError:
        print("错误：请输入一个有效的整数 sample_id！")

print(f"查询的 sample_id 对: {sample_id_pair}")

# 提取两个样本的数据
sample1 = df[df['sample_id'] == sample_id_pair[0]].sort_values('time_step')
sample2 = df[df['sample_id'] == sample_id_pair[1]].sort_values('time_step')

# 获取数据（将time_step转换为秒）
time1 = sample1['time_step'].values * 1e-4  # 转换为秒
voltage1, current1 = sample1['voltage'].values, sample1['current'].values
time2 = sample2['time_step'].values * 1e-4  # 转换为秒
voltage2, current2 = sample2['voltage'].values, sample2['current'].values

# 计算采样率
def get_sampling_rate(time):
    return 1.0 / np.mean(np.diff(time))

sampling_rate1 = get_sampling_rate(time1)
sampling_rate2 = get_sampling_rate(time2)

# FFT计算函数
def compute_fft(signal, sampling_rate):
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / sampling_rate)[:n//2]
    return xf, 2.0/n * np.abs(yf[0:n//2])

# 计算FFT
v_xf1, v_yf1 = compute_fft(voltage1, sampling_rate1)  # 样本1电压FFT
c_xf1, c_yf1 = compute_fft(current1, sampling_rate1)   # 样本1电流FFT
v_xf2, v_yf2 = compute_fft(voltage2, sampling_rate2)  # 样本2电压FFT
c_xf2, c_yf2 = compute_fft(current2, sampling_rate2)   # 样本2电流FFT

# 创建对比图表
plt.figure(figsize=(16, 12))

# 1. 电压时域对比
plt.subplot(2, 2, 1)
plt.plot(time1, voltage1, 'b-', label=f'Sample {sample_id_pair[0]}')
plt.plot(time2, voltage2, 'r--', label=f'Sample {sample_id_pair[1]}')
plt.title('Voltage Time Domain Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)

# 2. 电压频域对比
plt.subplot(2, 2, 2)
plt.plot(v_xf1, v_yf1, 'b-', label=f'Sample {sample_id_pair[0]}')
plt.plot(v_xf2, v_yf2, 'r--', label=f'Sample {sample_id_pair[1]}')
plt.title('Voltage Frequency Domain Comparison')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.xlim(0, min(np.max(v_xf1), np.max(v_xf2)))  # 限制x轴范围

# 3. 电流时域对比
plt.subplot(2, 2, 3)
plt.plot(time1, current1, 'b-', label=f'Sample {sample_id_pair[0]}')
plt.plot(time2, current2, 'r--', label=f'Sample {sample_id_pair[1]}')
plt.title('Current Time Domain Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend()
plt.grid(True)

# 4. 电流频域对比
plt.subplot(2, 2, 4)
plt.plot(c_xf1, c_yf1, 'b-', label=f'Sample {sample_id_pair[0]}')
plt.plot(c_xf2, c_yf2, 'r--', label=f'Sample {sample_id_pair[1]}')
plt.title('Current Frequency Domain Comparison')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.xlim(0, min(np.max(c_xf1), np.max(c_xf2)))  # 限制x轴范围

plt.tight_layout()
plt.show()
