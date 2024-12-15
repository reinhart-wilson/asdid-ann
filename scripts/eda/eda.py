# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:26:38 2024

@author: reinh
"""
import sys, os

working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)

from PIL import Image
from utils import eda_utils as ef
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150 
dataset_path = '../../dataset/additional_data_2'

# ==============
## EDA

# Lihat jumlah kelas dan nama kelas
num_classes, class_names = ef.count_classes(dataset_path)
print(f"Jumlah kelas: {num_classes}")
print(f"Nama kelas: {class_names}")

# Lihat persebaran data
class_counts = ef.count_data_per_class(dataset_path)
print("Banyak data per kelas:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")
ef.visualize_class_counts(class_counts, (16, 6), True)

# Jumlah data
total_data = ef.calculate_total_data(class_counts)
print("Total Anggota dari Semua Kelas:", total_data)

# ==============
## Implementasi CNN
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Params
batch_size = 32
img_height = 180
img_width = 180
seed = 42

# Load data train dan validasi
train_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_path,
  validation_split=0.2,
  subset="training",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_path,
  validation_split=0.2,
  subset="validation",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Tampilkan sampel dari set data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    # Shuffle dataset dengan seed
    train_ds_shuffled = train_ds.shuffle(buffer_size=len(train_ds), seed=seed)    
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# num_classes = 5

# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(num_classes)
# ])

# model.compile(
#   optimizer='adam',
#   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#   metrics=['accuracy'])

# model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=25
# )


# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Tentukan path dataset
# dataset_path = "flower_photos"

# # Tentukan ukuran batch dan dimensi gambar
# batch_size = 32
# image_size = (180, 180)

# # Inisialisasi objek ImageDataGenerator untuk augmentasi dan normalisasi
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2  # Proporsi data untuk validasi
# )

# # Persiapan dataset pelatihan
# train_generator = datagen.flow_from_directory(
#     dataset_path,
#     target_size=image_size,
#     batch_size=batch_size,
#     class_mode='categorical',  # Sesuaikan dengan jenis masalah Anda
#     subset='training'  # Menunjukkan ini adalah dataset pelatihan
# )

# # Persiapan dataset validasi
# validation_generator = datagen.flow_from_directory(
#     dataset_path,
#     target_size=image_size,
#     batch_size=batch_size,
#     class_mode='categorical',  # Sesuaikan dengan jenis masalah Anda
#     subset='validation'  # Menunjukkan ini adalah dataset validasi
# )

# # Jumlah kelas dalam dataset
# num_classes = len(train_generator.class_indices)

# # Contoh penggunaan dataset pelatihan dan validasi dalam model
# model = tf.keras.models.Sequential([
#     # Tambahkan layer-layer model di sini
# ])

# # Kompilasi model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Pelatihan model dengan dataset pelatihan dan validasi
# model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=validation_generator
# )

# # Evaluasi model pada dataset pengujian
# test_generator = datagen.flow_from_directory(
#     dataset_path,
#     target_size=image_size,
#     batch_size=batch_size,
#     class_mode='categorical',  # Sesuaikan dengan jenis masalah Anda
#     subset='validation'  # Menunjukkan ini adalah dataset pengujian
# )

# test_loss, test_acc = model.evaluate(test_generator)
# print(f'Test accuracy: {test_acc}')
