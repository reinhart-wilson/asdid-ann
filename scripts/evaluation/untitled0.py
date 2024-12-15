# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:21:15 2024

@author: rafae
"""
import sys, os

working_dir = os.path.abspath(os.path.dirname(__file__))
configs_dir = os.path.join(working_dir, '..', '..', 'config')
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)
sys.path.append(configs_dir)

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

mpl.rcParams['figure.dpi'] = 300 # Set resolusi plot

monitored_param = '5.transfer'
metric = 'Loss'
tensorboard_data_path = f'../../tensorboard_data/{monitored_param}/{metric}'
csv_files = ["tanpa transfer learning.csv", 
             "dengan transfer learning.csv"]
labels = ["Tanpa Transfer Learning", 
          "Dengan Transfer Learning"]

# Loop through each file and plot the data
for csv_file, label in zip(csv_files, labels):
    # Load CSV file
    data = pd.read_csv(os.path.join(tensorboard_data_path, csv_file))
    # data = data[data['Step'] <= 99]
    
    # Extract Step and Value columns
    steps = data['Step'] + 1  # Epoch - 1
    values = data['Value']  # Metric to plot
    
    # Plot the data
    plt.plot(steps, values, label=label)
# plt.gca().yaxis.set_major_locator(MultipleLocator(0.1)) 
# Customize the plot
plt.xlabel("Epoch")
plt.ylabel(metric)
plt.title("")
plt.legend()  # Add a legend
# plt.grid(True)  # Optional: Add a grid
plt.tight_layout()

# Show the plot
plt.show()



