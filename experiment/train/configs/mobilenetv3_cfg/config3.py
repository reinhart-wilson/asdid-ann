# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:13:19 2024

@author: reinh
"""

"""
Menggunakan dataset vanilla namun sudah di-resize dan di-crop
"""


from os import path

SEED = 42
CFG_NUM = 3
USE_GPU= True
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
    'variant'   : 'small'
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
    }
}
