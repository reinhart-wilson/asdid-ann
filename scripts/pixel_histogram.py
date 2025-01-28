# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:59:09 2025

@author: reinh
"""

import os, sys
# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir,'..','src')
sys.path.append(src_dir)

import matplotlib.pyplot as plt
from eda import eda_utils as eutils

# Folder dataset
dataset_path = "../dataset/histogram_data"
prepped_path = "../dataset/prepped_data"

# Hitung histogram
original_histogram = eutils.calculate_combined_histogram_rgb(dataset_path)
prepped_histogram = eutils.calculate_combined_histogram_rgb(prepped_path)
# noriginal_histogram=normalize_histogram(original_histogram)
# nprepped_histogram=normalize_histogram(prepped_histogram)
noriginal_histogram=(original_histogram)
nprepped_histogram=(prepped_histogram)

# # Histogram G
x = range(len(original_histogram['G']))
plt.bar(x, noriginal_histogram['G'], color='lightgreen', alpha=1, label='Sebelum Pemotongan')
plt.bar(x, nprepped_histogram['G'], color='seagreen', alpha=1, label='Setelah Pemotongan')

plt.title("Frekuensi Nilai Piksel Kanal Green")
plt.xlabel("Nilai Piksel")
plt.ylabel("Rata-Rata Kemunculan")
plt.legend()
plt.show()

# # Histogram R
x = range(len(original_histogram['R']))
plt.bar(x, noriginal_histogram['R'], color='salmon', alpha=1, label='Sebelum Pemotongan')
plt.bar(x, nprepped_histogram['R'], color='firebrick', alpha=1, label='Setelah Pemotongan')

plt.title("Frekuensi Nilai Piksel Kanal Red")
plt.xlabel("Nilai Piksel")
plt.ylabel("Rata-Rata Kemunculan")
plt.legend()
plt.show()

# # Histogram B
x = range(len(original_histogram['B']))
plt.bar(x, noriginal_histogram['B'], color='lightskyblue', alpha=1, label='Sebelum Pemotongan')
plt.bar(x, nprepped_histogram['B'], color='dodgerblue', alpha=1, label='Setelah Pemotongan')

plt.title("Frekuensi Nilai Piksel Kanal Blue")
plt.xlabel("Nilai Piksel")
plt.ylabel("Rata-Rata Kemunculan")
plt.legend()
plt.show()


# # # Histogram setelah pemotongan
# # plt.subplot(1, 2, 2)
# # plt.title("Histogram Gabungan Setelah Pemotongan")
# # plt.bar(range(len(cropped_histogram)), cropped_histogram, color='green')
# # plt.xlabel("Pixel Value")
# # plt.ylabel("Average Frequency")

# # plt.tight_layout()
# plt.show()