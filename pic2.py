import matplotlib.pyplot as plt
import numpy as np

N = np.array([8192,16384, 32768, 65536, 131072])  
times_a = np.array([23.6, 47.1, 76.4, 195.6, 480.4])        
times_b = np.array([10.6, 24.3, 47.9, 93.8, 252.6])       
times_c = np.array([0.05, 0.1, 0.2, 0.35, 0.55])     
times_d = np.array([0.2, 0.25, 0.45, 0.65, 0.8])     


def plot_graph(N, times_a, times_b, times_c, times_d):
    plt.figure(figsize=(10, 6))
    plt.plot(N, times_a, marker='o', label='multichain',  linewidth=4)
    plt.plot(N, times_b, marker='s', label='multichain unrolled',  linewidth=4)

    plt.xlabel('N (dataset size)')
    plt.ylabel('Time consumption [ms]')

    plt.legend()
    plt.grid(True)
    plt.show()

plot_graph(N, times_a, times_b, times_c, times_d)
