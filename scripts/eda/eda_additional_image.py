import sys
import os

# Tambahkan path proyek ke environment variable PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils import eda_utils

dataset_path = '../dataset/additional_data_2'

# Sampel data
eda_utils.show_sample_images(dataset_path, num_samples=3)

# Visualisasi persebaran data
class_counts = eda_utils.count_data_per_class(dataset_path)
sorted_counts = eda_utils.sort_counts(class_counts)
counts = list(sorted_counts.values())
color = [
    'skyblue' if v >= 700 
    else 'deepskyblue' if v >= 700 
    else 'deepskyblue' for v in counts
    ]
eda_utils.visualize_class_counts(sorted_counts, 
                                 plot_size=(8,6), 
                                 threshold=800, 
                                 in_percentage=True,
                                 rotation=0,
                                 horizontal = True,
                                 color=color
                                 )

# Visualisasi persebaran ukuran citra
eda_utils.eda_scatter_plot(dataset_path)



