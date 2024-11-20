# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
import sys
import os
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', 'src')
sys.path.append(src_dir)

# Panggil pustaka-pustaka yang digunakan
from mymodels.abstract_cnn import AbstractCNN
import numpy as np
import tensorflow as tf
import time

# Kelas dummy untuk pengujian
class DummyModel(AbstractCNN):
    def build_model(self):
        pass  # Tidak diperlukan karena model langsung dimuat dari file

# Muat model
arc_name = 'lenet'
training_result_folder = '../training_result/exp1'
input_shape = (224, 224, 3)
cnn_model = DummyModel(input_shape, 8)
cnn_model.load_model(
    os.path.join(
    training_result_folder, 
    arc_name,
    '1',
    'model_at_epoch100.keras'
    ))

# Uji waktu inferensi
target_size = (224, 224)
img_folder = '../dataset/split_prepped_data/test/bacterial_blight'
inference_times = []
for filename in os.listdir(img_folder):
    img = tf.keras.utils.load_img(
        os.path.join(img_folder, filename), 
        target_size=target_size)
    
    single_input = tf.keras.utils.img_to_array(img)  # Shape: (224, 224, 3)
    single_input = single_input / 255.0  # Normalisasi jika diperlukan
    single_input = np.expand_dims(single_input, axis=0)  # Tambahkan batch dimension: (1, 224, 224, 3)
    
    start_time = time.time()
    prediction = cnn_model.predict(single_input)
    end_time = time.time()
    inference_times.append(end_time-start_time)

# lenet_model = tf.keras.models.load_model(os.path.join(
#     training_result_folder, 
#     'lenet',
#     '1',
#     'model_at_epoch100.keras'
#     ))
# alexnet_model = tf.keras.models.load_model(os.path.join(
#     training_result_folder, 
#     'alexnet',
#     '1',
#     'model_at_epoch100.keras'
#     ))
# efficientnetv2_model = tf.keras.models.load_model(os.path.join(
#     training_result_folder, 
#     'efficientnetv2b20',
#     '1',
#     'model_at_epoch100.keras'
#     ))
# models = []

