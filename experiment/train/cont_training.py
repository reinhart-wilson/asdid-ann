# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:36:04 2024

@author: reinh
"""

import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)


import logging
from configs.mobilenetv2_cfg import config_imagenet1_augment2_10 as config
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
from utils import general_utils as gutils
from utils import training_utils as tutils


# Enable GPU (harus dilakukan sebelum import TensorFlow)
# gutils.use_gpu(True)


#Disable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)


# Load data
train_data_dir = os.path.join(config.DATA_PATH, 'train')
# train_data_dir = dataloc.ADDITIONAL_DATA_PATH
val_data_dir = os.path.join(config.DATA_PATH, 'validation')
# val_data_dir=dataloc.ADDITIONAL_DATA_PATH
train_datagen = gutils.make_datagen(train_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE,
                                    augment=config.AUGMENT)
val_datagen = gutils.make_datagen(val_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE)


#Continue training
at_epoch=31
model_path = os.path.join(
    config.RESULT_PATH, 
    config.MODEL_FILENAME.format(epoch=at_epoch)
    )
model = load_model(model_path) 
model_callbacks = tutils.generate_callbacks(config.CALLBACKS_CONFIG)
tensorboard_callback = callbacks.TensorBoard(log_dir=config.LOGDIR)
model_checkpoint = callbacks.ModelCheckpoint(
    os.path.join(config.RESULT_PATH, config.MODEL_FILENAME),
    verbose=0
    )
model_callbacks.append(tensorboard_callback)
model_callbacks.append(model_checkpoint)
tutils.train_model(model, train_datagen, val_datagen, config.EPOCHS, 
                   config.RESULT_PATH, last_epoch=at_epoch)