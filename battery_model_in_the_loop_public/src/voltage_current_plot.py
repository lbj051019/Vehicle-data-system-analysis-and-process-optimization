import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# 计算电压差
voltage_diff = np.abs(sample1['voltage'].values - sample2['voltage'].values)
time_points = sample1['time_step'].values

# 找出差异大于0.001的时间点
significant_diff_indices = np.where(voltage_diff > 0.001)[0]
significant_times = time_points[significant_diff_indices]

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 绘制电压曲线
ax1.plot(sample1['time_step'], sample1['voltage'], label=f'Sample {sample_id_pair[0]}')
ax1.plot(sample2['time_step'], sample2['voltage'], label=f'Sample {sample_id_pair[1]}')

# 绘制电流曲线
ax2.plot(sample1['time_step'], sample1['current'], '--', label=f'Sample {sample_id_pair[0]}')
ax2.plot(sample2['time_step'], sample2['current'], '--', label=f'Sample {sample_id_pair[1]}')

# 标记电压差异大的点
for t in significant_times:
    ax1.axvline(x=t, color='r', alpha=0.3, linestyle=':')
    ax2.axvline(x=t, color='r', alpha=0.3, linestyle=':')

# 添加图例和标签
ax1.set_title(f'Voltage and Current Comparison (Sample {sample_id_pair[0]} vs {sample_id_pair[1]})')
ax1.set_ylabel('Voltage (V)')
ax2.set_ylabel('Current (A)')
ax2.set_xlabel('Time Step')
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)

plt.tight_layout()
plt.show()
