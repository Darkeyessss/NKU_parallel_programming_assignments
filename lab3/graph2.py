import numpy as np
import matplotlib.pyplot as plt

# Define the array sizes
N = np.array([10, 50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000])

# Generate sample data: simulation of run times (mean times) for x86 and ARM platforms
np.random.seed(0)
run_time_x86 = np.log(N) * 10 + np.random.normal(0, 1, len(N)) * 2
run_time_arm = np.log(N) * 12 + np.random.normal(0, 1, len(N)) * 2.5

# Error margins simulating variability in measured run times
error_x86 = np.random.normal(1.5, 0.5, len(N))
error_arm = np.random.normal(2.0, 0.5, len(N))

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(N, run_time_x86, yerr=error_x86, fmt='o', color='black',ecolor='lightgray', label='x86',  elinewidth=3,linestyle='')
plt.errorbar(N, run_time_arm, yerr=error_arm, fmt='s', color='black',ecolor='lightgray', label='ARM', elinewidth=3,linestyle='')

plt.title('Run Time Comparison Between x86 and ARM Platforms')
plt.xlabel('Array Size (N)')
plt.ylabel('Run Time (ms)')
plt.legend()
plt.grid(True)
plt.show()