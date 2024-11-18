import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', 'src')
sys.path.append(src_dir)

import tensorflow as tf

# Muat model-model
training_result_folder = '../training_result/exp1'
lenet_model = tf.keras.models.load_model(os.path.join(
    training_result_folder, 
    'lenet',
    '1',
    'model_at_epoch100.keras'
    ))
alexnet_model = tf.keras.models.load_model(os.path.join(
    training_result_folder, 
    'alexnet',
    '1',
    'model_at_epoch100.keras'
    ))
efficientnetv2_model = tf.keras.models.load_model(os.path.join(
    training_result_folder, 
    'efficientnetv2b20',
    '1',
    'model_at_epoch100.keras'
    ))