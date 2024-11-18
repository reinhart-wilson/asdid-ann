# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:25:19 2023

@author: reinh
"""

import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import random

mpl.rcParams['figure.dpi'] = 150 # Set resolusi plot

def get_image_resolution(image_path):
    """
    Fungsi untuk mengambil dan mengembalikan ukuran gambar.
    
    Parameters:
    - image_path (str): Path penyimpanan gambar.
    
    Returns:
    tuple: Tuple berisi resolusi gambar dengan urutan height lalu width.
    """
    with Image.open(image_path) as img:
        return img.size
    
def get_set_resolutions(dataset_path):
    resolutions = []
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            resolution = get_image_resolution(image_path)
            resolutions.append(resolution)            
    return resolutions

def eda_scatter_plot(dataset_path):
    """
    Fungsi untuk membuat scatter plot resolusi gambar dalam set data.
    
    Parameters:
    - dataset_path (str): Path folder set data.
    """
    
    resolutions = get_set_resolutions(dataset_path)

    resolutions = list(set(resolutions))  # Menghilangkan resolusi duplikat
    width, height = zip(*resolutions)

    plt.scatter(width, height, alpha=0.5)
    plt.title('Persebaran Resolusi Gambar dalam Dataset')
    plt.xlabel('Lebar (Piksel)')
    plt.ylabel('Tinggi (Piksel)')
    plt.show()

def count_classes(dataset_path):
    """
    Fungsi untuk menghitung banyak kelas dengan melihat banyak folder dalam path
    yang ditentukan.
    
    Parameters:
    - dataset_path (str): Path folder set data.
    """
    classes = [class_folder for class_folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, class_folder))]
    return len(classes), classes

def count_data_per_class(dataset_path):
    """
    Fungsi untuk membuat scatter plot resolusi gambar dalam set data.
    
    Parameters:
        - dataset_path (str): Path folder set data.
    
    Returns:
        Sebuah dictionary dengan key nama kelas dan value banyak data dalam 
        kelas tsb.
    """
    class_counts = {}
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            class_counts[class_folder] = len(os.listdir(class_path))
    return class_counts

def sort_counts (class_counts, descending = True):
    sorted_counts = dict(sorted(class_counts.items(), 
                                key=lambda item: item[1], 
                                reverse=descending))
    return sorted_counts


def visualize_class_counts(class_counts, plot_size=None, show_label=True, 
                           threshold = 0, in_percentage=False, rotation = 0,
                           horizontal=False, color = 'skyblue'):
    """
    Menampilkan visualisasi bar chart dari banyaknya data per kelas.

    Parameters:
    - class_counts (dict): Dictionary berisi banyaknya data per kelas.
    - plot_size (tuple, optional): Ukuran plot sebagai tuple (lebar, tinggi).
    - show_label (bool, optional): Menentukan apakah menampilkan nilai di atas bar.

    Returns:
    None
    """
    if plot_size:
        plt.figure(figsize=plot_size)

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    bar_colors = color

    xlabel = 'Kelas'
    ylabel = 'Banyak Data'

    if horizontal:
        bars = plt.barh(classes, counts, color=bar_colors )
        xlabel, ylabel = ylabel, xlabel 
        max_val = max(counts)
    else:
        bars = plt.bar(classes, counts, color=bar_colors )

    if show_label:
        
        if in_percentage:
            total_data = calculate_total_data(class_counts)
        for bar in bars:
            
            # Atur posisi teks sesuai orientasi
            if horizontal:
                val = bar.get_width()
                x = val-(0.06*max_val)
                y = bar.get_y() + bar.get_height() / 2
            else:
                val = bar.get_height()
                x = bar.get_x() + bar.get_width()/2
                y = val
            
            # Bagi banyak data dengan total data jika diminta dalam persentase
            if in_percentage :
                ratio = val/total_data
                value = f'{ratio:.2%}'
            else: 
                value = round(val, 2)
            
            plt.text(x, 
                     y, 
                     value, 
                     ha='center', 
                     va='bottom')

    
    plt.xticks(rotation=rotation)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Banyak Data per Kelas')
    plt.show()
    
    
    
def calculate_total_data(class_counts):
    """
    Menghitung total anggota dari semua kelas.

    Parameters:
    - class_counts (dict): Dictionary berisi banyaknya data per kelas.

    Returns:
    int: Total anggota dari semua kelas.
    """
    total_data = sum(class_counts.values())
    return total_data

def calculate_data_ratio(class_counts):
    """
    Menghitung persebaran data dengan menhitung rasio banyak data per kelas 
    dengan jumlah total data

    Parameters
    ----------
    class_counts : DICT
        Sebuah dictionary dengan key nama kelas dan value banyak data dalam 
        kelas tsb. Gunakan fungsi count_data_per_class() jika perlu.

    Returns
    -------
    new_dict : DICT
        Dictionary dengan key = nama kelas dan value = persentase data di kelas
        tsb dibanding total data.

    """
    
    total_data = calculate_total_data(class_counts)
    new_dict = {k: v / total_data for k, v in class_counts.items()}
    return new_dict


def check_channels_used(dataset_path):
    channels_used = set()
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            with Image.open(image_path) as img:
                channels_used.update(img.getbands())
    return channels_used

def show_sample_images(dataset_path, num_samples=3):
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(len(os.listdir(dataset_path)), num_samples)

    for i, class_folder in enumerate(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            sample_images = os.listdir(class_path)[:num_samples]
            for j, image_name in enumerate(sample_images):
                image_path = os.path.join(class_path, image_name)
                img = Image.open(image_path)
                plt.subplot(gs[i, j])
                plt.imshow(img)
                plt.axis('off')
                plt.title(f'Class: {class_folder}')                
    plt.show()
    
def plot_resolution_heatmap(resolutions):
    # Membuat histogram
    plt.figure(figsize=(10, 6))
    plt.hist2d(*zip(*resolutions), bins=(50, 50), cmap='viridis')
    plt.title('Persebaran Ukuran Gambar')
    plt.xlabel('Lebar Gambar')
    plt.ylabel('Tinggi Gambar')
    plt.colorbar()
    plt.show()
    
def create_overall_size_histogram(data_directory):
    # Inisialisasi list untuk menyimpan ukuran file dari seluruh dataset
    all_sizes = []

    # Mendapatkan daftar kelas (sub-direktori) dalam direktori data
    classes = os.listdir(data_directory)

    # Iterasi melalui setiap kelas
    for class_name in classes:
        class_path = os.path.join(data_directory, class_name)

        # Memastikan bahwa yang diakses adalah direktori
        if os.path.isdir(class_path):
            # Iterasi melalui setiap file dalam kelas
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)

                # Memastikan bahwa yang diakses adalah file
                if os.path.isfile(file_path):
                    # Mendapatkan ukuran file dalam kilobytes dan menambahkannya ke list
                    file_size_kb = os.path.getsize(file_path) / 1024
                    all_sizes.append(file_size_kb)

    # Membuat histogram persebaran ukuran file
    plt.figure(figsize=(10, 6))
    plt.hist(all_sizes, bins=50, alpha=0.5, color='blue', edgecolor='black')

    plt.title('Persebaran Ukuran File Citra (KB) - Keseluruhan Dataset')
    plt.xlabel('Ukuran File (KB)')
    plt.ylabel('Frekuensi')
    plt.show()
    
def create_class_samples(num_samples = 2, data_dir="./data", 
                         output_dir='./output', seed=None):
    # Membuat direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Mengatur seed untuk random jika diberikan
    if seed is not None:
        random.seed(seed)

    # Iterasi melalui setiap kelas
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        # Mengambil daftar file gambar dalam kelas
        image_files = [file for file in os.listdir(class_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Mengambil sampel acak dari kelas
        random_samples = random.sample(image_files, min(num_samples, len(image_files)))

        # Menggabungkan sampel gambar menjadi satu gambar
        combined_image = Image.new("RGB", (400 * num_samples, 300), (255, 255, 255))  # Latar belakang putih

        for i, sample_file in enumerate(random_samples):
            sample_path = os.path.join(class_path, sample_file)
            sample_image = Image.open(sample_path)

            # Menambahkan latar belakang putih jika ukuran gambar kecil
            if sample_image.size[0] < 400 or sample_image.size[1] < 300:
                sample_image = ImageOps.expand(sample_image, border=(0, 0, 400 - sample_image.size[0], 300 - sample_image.size[1]), fill="white")

            # Menyesuaikan ukuran gambar agar sesuai dengan layout
            sample_image.thumbnail((400, 300))

            # Menghitung posisi untuk meletakkan gambar dalam satu gambar
            position = (400 * i, 0)
            combined_image.paste(sample_image, position)

        # Menyimpan gambar gabungan ke direktori output
        output_path = os.path.join(output_dir, f"{class_name}_sample_collage.jpg")
        combined_image.save(output_path)

        # Menampilkan gambar gabungan menggunakan Matplotlib
        plt.imshow(combined_image)
        plt.title(f"Class: {class_name}")
        plt.axis("off")
        plt.show()
