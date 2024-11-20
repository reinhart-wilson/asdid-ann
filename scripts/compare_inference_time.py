# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
import sys
import os
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', 'src')
sys.path.append(src_dir)

# Panggil pustaka-pustaka yang digunakan
from mymodels.abstract_cnn import AbstractCNN
from utils import general_utils as gutils
from utils import evaluation_utils as eutils
import gc
import numpy as np
import signal
import tensorflow as tf
import time

# Kelas dummy untuk pengujian
class DummyModel(AbstractCNN):
    def build_model(self):
        pass  # Tidak diperlukan karena model langsung dimuat dari file

# Muat model
arc_name = 'mobilenetv3large'
training_result_folder = os.path.join(
    '../training_result/exp1', 
    arc_name)
attempt_no = str(1)

# Periksa metrik
min_val_loss = None
max_val_acc = None
for no in range(1,4,1):
    print(no)
    attempt_no = str(no)
    history = eutils.load_history(
        os.path.join(
            training_result_folder,
            attempt_no,
            'history_at_epoch100.pkl'
            )
        )

    # Train loss
    checked_epochs = [10, 30, 50, 80, 100]
    print("Loss pelatihan")
    print( '\n'.join(f'{history["loss"][epoch-1]:.3f}' 
                     for epoch in checked_epochs) + '\n')
    
    
    # Val loss
    print('Loss validasi\n' + 
          '\n'.join(f'{history["val_loss"][epoch-1]:.3f}' 
                    for epoch in checked_epochs), 
          '\n')
    
    # Val acc
    print('Akurasi pelatihan\n' + 
          '\n'.join(f'{history["accuracy"][epoch-1]*100:.2f}' 
                    for epoch in checked_epochs), 
          '\n')
    
    # Val acc
    print('Akurasi validasi\n' + 
          '\n'.join(f'{history["val_accuracy"][epoch-1]*100:.2f}' 
                    for epoch in checked_epochs), 
          '\n')
    
    cur_min_loss = min(history['val_loss'])
    cur_max_acc = max(history['val_accuracy'])
    idx = history['val_loss'].index(cur_min_loss)
    print(f"Terbaik dalam percobaan ke-{no}: epoch ke-{idx+1}")
    print(f'- val_loss: {cur_min_loss}')
    print(f'- val_acc: {history["val_accuracy"][idx]}')
    
    
    max_val_acc = max(max_val_acc, cur_max_acc) if max_val_acc is not None else cur_max_acc
    min_val_loss = min(min_val_loss, cur_min_loss) if min_val_loss is not None else cur_min_loss

    eutils.plot_loss(history)
# input_shape = (224, 224, 3)
# cnn_model = DummyModel(input_shape, 8)
# cnn_model.load_model( 
#     os.path.join(
#         training_result_folder,
#         'model_at_epoch100.keras'
#         )
#     )

# target_size = (224, 224)
# img_folder = '../dataset/split_prepped_data/test/bacterial_blight'
# inference_times = []

# #warm up
# dummy_input = tf.constant(np.random.randn(1,224,224,3))
# for _ in range(10):  # Warm-up runs
#     cnn_model.predict(dummy_input)

# for filename in os.listdir(img_folder):
#     img = tf.keras.utils.load_img(
#         os.path.join(img_folder, filename), 
#         target_size=target_size)
    
#     single_input = tf.keras.utils.img_to_array(img)  # Shape: (224, 224, 3)
#     single_input = single_input / 255.0  # Normalisasi jika diperlukan
#     single_input = np.expand_dims(single_input, axis=0)  # Tambahkan batch dimension: (1, 224, 224, 3)
    
#     # Uji waktu inferensi
#     start_time = time.time()
#     # with tf.device('/CPU:0'):
#     prediction = cnn_model.predict(single_input)
#     end_time = time.time()
#     inference_times.append(end_time-start_time)
    
# # Hitung FLOPs
# print(eutils.get_flops(cnn_model.get_tf_model(), [dummy_input]))



# gc.collect()

