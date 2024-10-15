# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:01:44 2024

@author: reinh
"""
"""
Konfigurasi yang terinspirasi oleh konfigurasi MobileNetV2 yang dipakai untuk 
tugas klasifikasi ImageNet.  menggunakan augmentasi
"""

from os import path
from tensorflow.keras import optimizers

# SEED = 42
CFG_NUM = 'imagenet1_augment2.4'
USE_GPU= True
DATA_PATH = "../../dataset/split_prepped_data"

# Parameter umum
IMAGE_SIZE      = (224, 224)  
INPUT_SHAPE     = tuple(list(IMAGE_SIZE) + [3])
NUM_CLASSES     = 8
BATCH_SIZE      = 12
EPOCHS          = 200 
LEARNING_RATE   = 0.045/8
DECAY_STEPS     = EPOCHS
AUGMENT         = True
LR_ALPHA        = 1e-3

# Parameter spesifik arsitektur
MODEL_CONFIG = {
    'model_name': 'mobilenetv2',
    'alpha'     : 1.0,
    'dense'     : 1024,
    'dropout'   : 0,
    'weight_decay':4e-5
    }

RESULT_PATH = f"./training_result/{MODEL_CONFIG['model_name']}/config{CFG_NUM}"

# Args untuk callbacks
SAVE_INTERVAL   = 10
HISTORY_FILENAME= 'history_at_epoch{epoch}.pkl'
MODEL_FILENAME= 'model_at_epoch{epoch}.keras'

# Callbacks 
CALLBACKS_CONFIG = {
    'history_saver' : {
        'interval' : SAVE_INTERVAL,
        'save_path' : path.join(RESULT_PATH, HISTORY_FILENAME)
    },
    'model_checkpoint' : {
        'interval' : SAVE_INTERVAL,
        'save_path' : path.join(RESULT_PATH, MODEL_FILENAME)
    },
    'learning_rate_logger':{}
}
