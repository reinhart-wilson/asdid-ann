# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 21:58:16 2025

@author: reinh
"""

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Folder dataset
dataset_path = "../dataset/original_data"

# Fungsi untuk menghitung histogram gabungan dari semua gambar
def calculate_combined_histogram(dataset_path):
    combined_histogram = np.zeros(256)  # Untuk gambar grayscale
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            for image_path in os.listdir(class_path):
                # Baca gambar
                image_full_path = os.path.join(dataset_path, class_folder, image_path)
                if image_full_path.endswith((".jpg", ".png")):
                    image = Image.open(image_full_path).convert("L")
                    
                    # Hitung histogram dan tambahkan ke histogram gabungan
                    histogram = np.array(image.histogram())
                    combined_histogram += histogram
    
    # Normalisasi histogram gabungan
    combined_histogram /= len(dataset_path)
    return combined_histogram

# Hitung histogram untuk gambar asli
original_histogram = calculate_combined_histogram(dataset_path)

# # Hitung histogram untuk gambar setelah dipotong
# cropped_histogram = calculate_combined_histogram(dataset_path, crop=True)

# Visualisasi histogram gabungan
plt.figure(figsize=(12, 6))

# Histogram asli
plt.subplot(1, 2, 1)
plt.title("Histogram Gabungan Sebelum Pemotongan")
plt.bar(range(len(original_histogram)), original_histogram, color='blue')
plt.xlabel("Pixel Value")
plt.ylabel("Average Frequency")

# # Histogram setelah pemotongan
# plt.subplot(1, 2, 2)
# plt.title("Histogram Gabungan Setelah Pemotongan")
# plt.bar(range(len(cropped_histogram)), cropped_histogram, color='green')
# plt.xlabel("Pixel Value")
# plt.ylabel("Average Frequency")

# plt.tight_layout()
plt.show()