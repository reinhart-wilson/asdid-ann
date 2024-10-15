# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:24:48 2024

@author: reinh
"""

from os import path

SEED = 42
CFG_NUM = 2
USE_GPU= True
DATA_PATH = "../../dataset/split_prepped_data"

# Parameter umum
BATCH_SIZE      = 10
EPOCHS          = 100 
IMAGE_SIZE      = (224, 224)  
INPUT_SHAPE     = tuple(list(IMAGE_SIZE) + [3])
LEARNING_RATE   = 1e-4
NUM_CLASSES     = 8
AUGMENT = False

# Parameter spesifik arsitektur
MODEL_CONFIG = {
    'model_name': 'mobilenetv3',
    'alpha'     : 1.0,
    'dense'     : 1024,
    'variant':'large'
    }

RESULT_PATH = f"./training_result/exp1/{MODEL_CONFIG['model_name']}{MODEL_CONFIG['variant']}/{CFG_NUM}"

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
    }
}
