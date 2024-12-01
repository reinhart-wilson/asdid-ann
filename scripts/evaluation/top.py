# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:21:15 2024

@author: rafae
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:08:55 2024

@author: rafae
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# List of CSV file paths
csv_files = ["file1.csv", "file2.csv", "file3.csv"]
labels = ["Experiment 1", "Experiment 2", "Experiment 3"]  # Labels for each plot

# Maximum step to include
max_steps = 99

# Create subplots: 1 row, 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Loop through each file and plot the data in individual subplots
for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
    # Load CSV file
    data = pd.read_csv(csv_file)
    
    # Extract Step and Value columns, filter by max_steps
    filtered_data = data[data['Step'] <= max_steps]
    steps = filtered_data['Step']
    values = filtered_data['Value']
    
    # Plot the data in the respective subplot
    axes[i].plot(steps, values, label=label)
    axes[i].set_title(label)  # Title for each subplot
    axes[i].set_xlabel("Steps (Epoch - 1)")
    axes[i].grid(True)

# Set shared y-axis label
fig.text(0.06, 0.5, "Value", va='center', rotation='vertical')

# Add combined legend
lines, labels = [], []
for ax in axes:
    ax_lines, ax_labels = ax.get_legend_handles_labels()
    lines.extend(ax_lines)
    labels.extend(ax_labels)
fig.legend(lines, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend

# Show the plot
plt.show()





