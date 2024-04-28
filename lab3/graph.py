import matplotlib.pyplot as plt
import pandas as pd

# Data provided by the user
data_combined = {
    "N": [10, 50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000],
    #"Serialx86": [0.002, 0.113, 2.698, 11.338, 67.208, 278.932, 656.163, 980.281, 1234, 2254, 2908, 10097],
    "SerialARM": [0.002, 0.375, 5.042, 24.786, 170.389, 298.932, 476.372,980, 1576, 3626, 6763,17892],
    "NEON": [0.014, 1.953, 7.332, 29.308, 169.900, 279.009, 356.163, 900, 1498, 3359, 6421,14556],
    #"SSE": [0.001, 0.096, 1.343, 4.854, 29.533, 99.908, 309.142, 437.296,805.309, 1101.723, 2077.19, 6295.11],
    #"SSE(alignment)": [0.301, 0.696, 1.736, 5.012, 30.576, 98.878, 300.476, 411.87,790.30, 1066.7, 2050, 6187],
    #"AVX": [0.004, 0.15, 0.986, 1.738, 20.535, 45.635, 111.702, 207.252, 459.13, 682.334, 1039.172, 3921.98],
    # "Serial_groebner": [0.001, 0.202, 0.439, 1.418, 5.824, 13.625, 21.897, 32.475, 49.853, 60.071, 70.874, 104.09],
    "NEON_groebner": [0.078, 0.121, 0.345, 1.765, 4.785, 11.112, 17.238, 25.86, 30.837, 35.286, 47.982, 60.888],
}

# Convert to DataFrame
df_combined = pd.DataFrame(data_combined)

# Plotting settings
colors_combined = [ 'lightgreen', 'lightpink', 'lightsteelblue','lightcoral',  'lightgoldenrodyellow']  
line_width = 3
marker_style = 'o'

# Create a new figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each algorithm on the same figure
for idx, algorithm in enumerate(df_combined.columns[1:]):  # Exclude 'N' which is the x-axis
    ax.plot(df_combined['N'], df_combined[algorithm], marker=marker_style, color=colors_combined[idx], linewidth=line_width, label=algorithm)

# Title and labels
# ax.set_title('Algorithm Execution Time vs Problem Size N')
ax.set_xlabel('Matrix Size N')
ax.set_ylabel('time/ms')

# Legend and grid
ax.legend(title='Algorithm')
ax.grid(True)
ax.set_yscale('log')
# Setting the x-axis ticks to be the problem sizes
ax.set_xticks(df_combined['N'])

# Saving the plot to a file
plt.show()
