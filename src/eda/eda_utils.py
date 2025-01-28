# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:25:19 2023

@author: reinh
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
import random
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor

mpl.rcParams['figure.dpi'] = 150 # Set resolusi plot

def calculate_combined_histogram_rgb(dataset_path):
    combined_histogram = {
        "R": np.zeros(256),
        "G": np.zeros(256),
        "B": np.zeros(256),
    }
    image_count = 0
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for class_entry in os.scandir(dataset_path):
            if class_entry.is_dir():
                for image_entry in os.scandir(class_entry.path):
                    if image_entry.name.endswith((".jpg", ".png")):
                        futures.append(executor.submit(process_image_rgb, image_entry.path))
                        image_count += 1
        
        for future in futures:
            histograms = future.result()
            combined_histogram["R"] += histograms["R"]
            combined_histogram["G"] += histograms["G"]
            combined_histogram["B"] += histograms["B"]
    
    # Normalisasi histogram gabungan
    if image_count > 0:
        for key in combined_histogram:
            combined_histogram[key] /= image_count
    return combined_histogram

def process_image_rgb(image_path):
    image = Image.open(image_path).convert("RGB")
    r, g, b = image.split()
    return {
        "R": np.array(r.histogram()),
        "G": np.array(g.histogram()),
        "B": np.array(b.histogram()),
    }

def normalize_histogram(histograms):
    # Cari nilai maksimum global dari semua kanal
    global_max = max(max(values) for values in histograms.values())
    print(global_max)
    
    # Normalisasi setiap kanal berdasarkan nilai maksimum global
    normalized_histograms = {
        key: values / global_max if global_max > 0 else values
        for key, values in histograms.items()
    }
    return normalized_histograms

def get_image_resolution(image_path):
    """
    Mengambil dan mengembalikan ukuran gambar.
    
    Parameters
    ----------
    image_path : str
        Path ke file gambar.
    
    Returns
    -------
    tuple
        Resolusi gambar dalam bentuk (lebar, tinggi).
    """
    with Image.open(image_path) as img:
        return img.size
    
def get_set_resolutions(dataset_path):
    """
    Mengambil resolusi semua gambar dalam folder dataset.

    Parameters
    ----------
    dataset_path : str
        Jalur ke folder dataset.

    Returns
    -------
    list
        Daftar resolusi gambar dalam format (lebar, tinggi).
    """
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
    Membuat scatter plot resolusi gambar dalam dataset.

    Parameters
    ----------
    dataset_path : str
        Path ke folder dataset.

    Returns
    -------
    None
        Scatter plot ditampilkan menggunakan Matplotlib.
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
    Menghitung jumlah kelas dalam dataset berdasarkan jumlah sub-folder.

    Parameters
    ----------
    dataset_path : str
        Jalur ke folder dataset.

    Returns
    -------
    tuple
        Jumlah kelas dan daftar nama kelas.
    """
    classes = [class_folder for class_folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, class_folder))]
    return len(classes), classes

def count_data_per_class(dataset_path):
    """
    Menghitung jumlah data per kelas dalam dataset.

    Parameters
    ----------
    dataset_path : str
        Jalur ke folder dataset.

    Returns
    -------
    dict
        Dictionary berisi jumlah data untuk setiap kelas.
    """
    class_counts = {}
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            class_counts[class_folder] = len(os.listdir(class_path))
    return class_counts

def visualize_class_counts(class_counts, plot_size=None, show_label=False):
    """
    Menampilkan visualisasi bar chart jumlah data per kelas.

    Parameters
    ----------
    class_counts : dict
        Dictionary berisi jumlah data untuk setiap kelas.
    plot_size : tuple, optional
        Ukuran plot dalam format (lebar, tinggi).
    show_label : bool, optional
        Jika True, menampilkan nilai di atas setiap bar.

    Returns
    -------
    None
        Bar chart ditampilkan menggunakan Matplotlib.
    """
    if plot_size:
        plt.figure(figsize=plot_size)

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    bars = plt.bar(classes, counts, color='skyblue')

    if show_label:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    plt.xlabel('Kelas')
    plt.ylabel('Banyak Data')
    plt.title('Banyak Data per Kelas')
    plt.show()
    
def calculate_total_data(class_counts):
    """
    Menghitung total jumlah data dari semua kelas.

    Parameters
    ----------
    class_counts : dict
        Dictionary berisi jumlah data untuk setiap kelas.

    Returns
    -------
    int
        Total jumlah data dari semua kelas.
    """
    total_data = sum(class_counts.values())
    return total_data

def check_channels_used(dataset_path):
    """
    Memeriksa kanal warna yang digunakan dalam semua gambar di dalam set data.

    Parameters
    ----------
    dataset_path : str
        ath ke folder dataset.

    Returns
    -------
    channels_used : set
        Kanal warna yang ditemukan dalam dataset.

    """
    channels_used = set()
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            with Image.open(image_path) as img:
                channels_used.update(img.getbands())
    return channels_used
    
def plot_resolution_heatmap(resolutions):
    """
    Membuat heatmap untuk memperlihatkan persebaran resolusi gambar dalam dataset.

    Parameters
    ----------
    resolutions : list
        List resolusi gambar. Format sama dengan keluaran fungsi get_set_resolutions

    Returns
    -------
    None.

    """
    plt.figure(figsize=(10, 6))
    plt.hist2d(*zip(*resolutions), bins=(50, 50), cmap='viridis')
    plt.title('Persebaran Ukuran Gambar')
    plt.xlabel('Lebar Gambar')
    plt.ylabel('Tinggi Gambar')
    plt.colorbar()
    plt.show()
    
def create_overall_size_histogram(data_directory):
    """
    Membuat histogram untuk memperlihatkan distribusi ukuran file gambar dalam 
    dataset.

    Parameters
    ----------
    data_directory : str
        Path ke direktori dataset.

    Returns
    -------
    None.
    """
    all_sizes = []
    classes = os.listdir(data_directory)

    for class_name in classes:
        class_path = os.path.join(data_directory, class_name)

        # Memastikan bahwa yang diakses adalah direktori
        if os.path.isdir(class_path):
            # Iterasi melalui setiap file dalam kelas
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)

                # Memastikan bahwa yang diakses adalah file
                if os.path.isfile(file_path):
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
    """
    Membuat kolase gambar sampel dari setiap kelas dalam dataset.
    
    Parameters
    ----------
    num_samples : int, opsional
        Jumlah gambar sampel per kelas. Default adalah 2.
    data_dir : str, optional
        Path direktori dataset. Default adalah './data'.
    output_dir : str, optional
        Path direktori output untuk menyimpan kolase. Default adalah './output'.
    seed : int, optional 
        Seed untuk memastikan randomisasi yang konsisten. Default adalah None.
    
    Returns
    ----------
    None.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if seed is not None:
        random.seed(seed)

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        image_files = [file for file in os.listdir(class_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random_samples = random.sample(image_files, min(num_samples, len(image_files)))
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

        output_path = os.path.join(output_dir, f"{class_name}_sample_collage.jpg")
        combined_image.save(output_path)

        plt.imshow(combined_image)
        plt.title(f"Class: {class_name}")
        plt.axis("off")
        plt.show()


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