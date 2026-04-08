import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

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

print(f"对比的 sample_id 为: {sample_id_pair}")

# 获取两个样本的数据
sample1 = df[df['sample_id'] == sample_id_pair[0]].sort_values('time_step')
sample2 = df[df['sample_id'] == sample_id_pair[1]].sort_values('time_step')

# 定义计算统计特征的函数
def calculate_statistical_features(signal):
    features = {}
    
    # 基本统计量
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))  # 均方根
    features['peak_to_peak'] = np.max(signal) - np.min(signal)  # 峰峰值
    
    # 高阶统计量
    features['skewness'] = skew(signal)  # 偏度
    features['kurtosis'] = kurtosis(signal)  # 峭度
    
    # 波形指标
    features['crest_factor'] = np.max(np.abs(signal)) / features['rms']  # 波峰因子
    features['impulse_factor'] = np.max(np.abs(signal)) / np.mean(np.abs(signal))  # 脉冲因子
    features['margin_factor'] = np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal))))**2  # 裕度因子
    
    return features

# 计算 sample1 和 sample2 的电压和电流特征
voltage_features_sample1 = calculate_statistical_features(sample1['voltage'].values)
current_features_sample1 = calculate_statistical_features(sample1['current'].values)

voltage_features_sample2 = calculate_statistical_features(sample2['voltage'].values)
current_features_sample2 = calculate_statistical_features(sample2['current'].values)

# 打印结果
print("\nSample 1 (ID: {}) Voltage Features:".format(sample_id_pair[0]))
for key, value in voltage_features_sample1.items():
    print(f"{key}: {value:.4f}")

print("\nSample 1 (ID: {}) Current Features:".format(sample_id_pair[0]))
for key, value in current_features_sample1.items():
    print(f"{key}: {value:.4f}")

print("\nSample 2 (ID: {}) Voltage Features:".format(sample_id_pair[1]))
for key, value in voltage_features_sample2.items():
    print(f"{key}: {value:.4f}")

print("\nSample 2 (ID: {}) Current Features:".format(sample_id_pair[1]))
for key, value in current_features_sample2.items():
    print(f"{key}: {value:.4f}")

# 绘制波形对比
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(sample1['time_step'], sample1['voltage'], label=f'Sample {sample_id_pair[0]} Voltage')
plt.plot(sample2['time_step'], sample2['voltage'], label=f'Sample {sample_id_pair[1]} Voltage')
plt.xlabel('Time Step')
plt.ylabel('Voltage')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sample1['time_step'], sample1['current'], label=f'Sample {sample_id_pair[0]} Current')
plt.plot(sample2['time_step'], sample2['current'], label=f'Sample {sample_id_pair[1]} Current')
plt.xlabel('Time Step')
plt.ylabel('Current')
plt.legend()

plt.tight_layout()
plt.show()
