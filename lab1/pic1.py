import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(15, 6))
N_values_exponential = 2 ** np.arange(16)

y1_discrete_data = np.array([0,0,0,0,0,0,0,0,0,0,0,1,4,9,20,41])
y2_discrete_data = np.array([0,0,0,0,0,0,0,0,0,0,0,0,2,7,12,27])
y3_discrete_data = np.array([0,0,0,0,0,0,0,0,0,0,1,2,5,11,23,54])
y4_discrete_data = np.array([0,0,0,0,0,0,0,0,0,0,0,1,4,8,16,33])
y5_discrete_data = np.array([0,0,0,0,0,0,0,0,1,2,4,8,16,33,66,132])
y6_discrete_data = np.array([0,0,0,0,0,0,0,0,0,1,2,3,12,25,49,99])

plt.clf()

plt.plot(N_values_exponential, y1_discrete_data, label='linux-x86 Trivial', color='red', linestyle='-', linewidth=3)
plt.plot(N_values_exponential, y2_discrete_data, label='linux-x86 Optimized', color='green', linestyle='-', linewidth=3)
plt.plot(N_values_exponential, y3_discrete_data, label='win-x86 Trivial', color='blue', linestyle='-', linewidth=3)
plt.plot(N_values_exponential, y4_discrete_data, label='win-x86 Optimized', color='purple', linestyle='-', linewidth=3)
plt.plot(N_values_exponential, y5_discrete_data, label='linux-arm Trivial', color='orange', linestyle='-', linewidth=3)
plt.plot(N_values_exponential, y6_discrete_data, label='linux-arm Optimized', color='palegoldenrod', linestyle='-', linewidth=3)

plt.legend()

plt.xlabel('N (size of caches)')
plt.ylabel('Time consumption [ms]')


plt.grid(False)

plt.xscale('log', base=2)

plt.xticks(N_values_exponential, [f'$2^{{{i}}}$' for i in range(16)])

plt.show()
