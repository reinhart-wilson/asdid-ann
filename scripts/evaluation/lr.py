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
from utils import evaluation_utils as eutils

mpl.rcParams['figure.dpi'] = 300 # Set resolusi plot

monitored_param = '3.augment'
tensorboard_data_path = f'../../tensorboard_data/{monitored_param}'
csv_files = [
    "cosdecay_validation.csv",
    "augment_validation.csv" 
             ]
labels = ["Tanpa Augmentasi Data", 
          "Dengan Augmentasi Data"]

eutils.show_tensorboard_plots(tensorboard_data_path, csv_files, labels, 200,
                              show_legend=False, line_colors=['orange', 'seagreen'])




