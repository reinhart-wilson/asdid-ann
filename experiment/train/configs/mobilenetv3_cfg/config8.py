# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:30:56 2024

@author: reinh
"""

from os import path

USE_DETERMINISM = False
SEED = 1
CFG_NUM = 8
USE_GPU= False
DATA_PATH = "../../dataset/split_prepped_data"

# Parameter umum
BATCH_SIZE      = 16
EPOCHS          = 100 
IMAGE_SIZE      = (224, 224)  
INPUT_SHAPE     = tuple(list(IMAGE_SIZE) + [3])
LEARNING_RATE   = 1e-4
NUM_CLASSES     = 8


# Parameter spesifik arsitektur
MODEL_CONFIG = {
    'model_name': 'mobilenetv3',
    'alpha'     : 1.0,
    'dense'     : 1024,
    'variant'   : 'large'
    }

RESULT_PATH = f"./training_result/{MODEL_CONFIG['model_name']}/config{CFG_NUM}"

# Args untuk callbacks
SAVE_INTERVAL   = 5
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
