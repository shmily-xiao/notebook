import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# 1. 生成模拟数据（100个时间点的CPU使用率）
np.random.seed(42)  # 保证每次结果一致
normal_data = np.random.normal(loc=60, scale=10, size=100)  # 正常数据（均值60，标准差10）
data = normal_data.copy()
anomaly_indices = [20, 50, 70, 90]  # 随机选4个异常点位置
anomalies = [85, 92, 88, 95]        # 人工插入的异常点
for idx, val in zip(anomaly_indices, anomalies):
    data[idx] = val
timestamps = pd.date_range(start="2025-01-01", periods=100, freq="h")

# 2. 异常检测（Z-Score 算法）
mean = np.mean(data)
std = np.std(data)
z_scores = [(x - mean) / std for x in data]
detected_anomaly_indices = [i for i, z in enumerate(z_scores) if abs(z) > 3]

# 3. 可视化
plt.figure(figsize=(12, 6))
plt.plot(timestamps, data, color='blue', alpha=0.7, label='CPU Usage')
plt.scatter(timestamps[detected_anomaly_indices], data[detected_anomaly_indices], 
            color='red', s=100, zorder=5, label='Anomaly Detected')
plt.title('AIOPS Demo: Real-time CPU Anomaly Detection', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('CPU Usage (%)', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# 4. 添加演讲备注（在演示时口头说明）
print("\n" + "="*50)
print("演讲备注：")
print("1. 模拟了服务器CPU使用率数据（正常范围50-70%）")
print("2. AIOPS通过Z-Score算法自动检测到4个异常峰值（>90%）")
print("3. 实际场景中，系统会自动告警并触发根因分析")
print("="*50)

plt.show()
