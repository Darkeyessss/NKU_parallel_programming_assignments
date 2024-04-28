import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 根据用户提供的新数据定义和处理来绘制热力图
N_new = [20, 50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000]
times_new = {
    "Serialx86": [0.002, 0.113, 2.698, 11.338, 67.208, 278.932, 656.163, 980.281, 1234, 2254, 2908, 10097],
    "SerialARM": [0.002, 0.375, 5.042, 24.786, 170.389, 298.932, 476.372, 980, 1576, 3626, 6763, 17892],
    "NEON": [0.014, 1.953, 7.332, 29.308, 169.900, 279.009, 356.163, 900, 1498, 3359, 6421, 14556],
    "SSE": [0.001, 0.096, 1.343, 4.854, 29.533, 99.908, 309.142, 437.296, 805.309, 1101.723, 2077.19, 6295.11],
    "SSE(alignment)": [0.301, 0.696, 1.736, 5.012, 30.576, 98.878, 300.476, 411.87, 790.30, 1066.7, 2050, 6187],
    "AVX": [0.004, 0.15, 0.986, 1.738, 20.535, 45.635, 111.702, 207.252, 459.13, 682.334, 1039.172, 3921.98],
}

data_new = np.array([times_new[key] for key in times_new]).T
log_data_new = np.log(np.clip(data_new, a_min=0.001, a_max=None))

# 绘制新的热力图
plt.figure(figsize=(12, 9))
ax = sns.heatmap(log_data_new, annot=True, fmt=".3f", cmap="viridis", xticklabels=list(times_new.keys()), yticklabels=N_new)
plt.title("Running Time (Log Scale)")
plt.xlabel("Optimization Methods")
plt.ylabel("Array Size (N)")
plt.xticks(rotation=45)  # 横坐标标签斜着写
plt.show()
