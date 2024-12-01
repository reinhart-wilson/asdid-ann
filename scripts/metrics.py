# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:37:30 2024

@author: reinh
"""

import os
import sys

working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', 'src')
sys.path.append(src_dir)

from utils import evaluation_utils as eutils
import numpy as np

exp1_path = '../training_result/exp1'
for folder in os.listdir(exp1_path):
    metrics = []
    for i in range(1, 4):
        logs_dir = os.path.join(
            exp1_path, folder, f'{i}', 'logs', 'train'
        )
        best = eutils.extract_metrics_from_logs(logs_dir)
        metrics.append(best)

    # Initialize dictionaries to store sums and lists for std calculation
    totals = {'lowest_loss': 0, 'highest_accuracy': 0, 'highest_recall': 0}
    metric_lists = {'lowest_loss': [], 'highest_accuracy': [], 'highest_recall': []}

    # Sum each metric and collect values for std calculation
    for metric in metrics:
        for key in metric:
            totals[key] += metric[key]
            metric_lists[key].append(metric[key])

    # Compute averages
    averages = {key: value / len(metrics) for key, value in totals.items()}

    # Compute standard deviations
    stds = {key: np.std(values) for key, values in metric_lists.items()}

    # Round results
    rounded_results = {
        'lowest_loss': (round(averages['lowest_loss'], 3), round(stds['lowest_loss'], 3)),
        'highest_accuracy': (round(averages['highest_accuracy'] * 100, 3), round(stds['highest_accuracy'] * 100, 3)),  # Convert to percentage
        'highest_recall': (round(averages['highest_recall'], 3), round(stds['highest_recall'], 3)),
    }

    # Format results as LaTeX-friendly strings
    formatted_results = {
        key: f"\({avg} \pm {std}\)"
        for key, (avg, std) in rounded_results.items()
    }

    # Print results
    print(f"Results for {folder}:")
    print("LaTeX Format:")
    for key, value in formatted_results.items():
        print(f"{key}: {value}")
    print()
