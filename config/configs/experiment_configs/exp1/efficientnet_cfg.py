# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:20:22 2024

@author: rafae
"""

from os import path
import time

SEED = 42
CFG_NUM = 1
USE_GPU = True
DATA_PATH = "../../dataset/split_prepped_data"

# Parameter umum
BATCH_SIZE      = 10
EPOCHS          = 100 
IMAGE_SIZE      = (224, 224)  
INPUT_SHAPE     = tuple(list(IMAGE_SIZE) + [3])
LEARNING_RATE   = 0.0001
NUM_CLASSES     = 8
AUGMENT=False


# Parameter spesifik arsitektur
MODEL_CONFIG = {
    'model_name' : 'efficientnet'
    }
RESULT_PATH = f"./training_result/exp1/{MODEL_CONFIG['model_name']}/{CFG_NUM}"


# Args untuk callbacks
SAVE_INTERVAL   = 5
HISTORY_FILENAME= 'history_at_epoch{epoch}.pkl'
MODEL_FILENAME  = 'model_at_epoch{epoch}.keras'

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