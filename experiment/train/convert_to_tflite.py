# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:17:54 2024

@author: reinh
"""

import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)

train_dir = os.path.join(working_dir, '..', 'train')
sys.path.append(train_dir)


from tensorflow import lite as tflite
import tensorflow as tf
from keras.models import load_model
from configs.mobilenetv2_cfg import config_imagenet1_augment2_7_2 as config

# Memuat model yang sudah dilatih
at_epoch = 1
# model_filename = config.MODEL_FILENAME.format(epoch=at_epoch)
# model_path = os.path.join(config.RESULT_PATH, model_filename)
# model = load_model(model_path)
saved_model_path = os.path.join(
    config.RESULT_PATH, 
    f"model_at_epoch{at_epoch}")
# tf.saved_model.save(model, saved_model_path)
# Ubah model ke model TFLite
converter = tflite.TFLiteConverter.from_saved_model(saved_model_path)
converter.experimental_new_converter = True
# converter.target_spec.supported_ops = [
#     tflite.OpsSet.TFLITE_BUILTINS,  # TFLite operations
#     tflite.OpsSet.SELECT_TF_OPS     # Enables Flex compatibility
# ]
tflite_model = converter.convert()
tflite_filename = os.path.join(
    config.RESULT_PATH, 
    f"model_at_epoch_{at_epoch}.tflite")
with open(tflite_filename,'wb') as f:
    f.write(tflite_model)