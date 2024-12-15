# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:38:21 2024

@author: reinh
"""


from os import path

SEED = 42
CFG_NUM = 4
USE_GPU= True
DATA_PATH = "../dataset/split_prepped_data"

# Parameter umum
BATCH_SIZE      = 10
EPOCHS          = 100 
IMAGE_SIZE      = (224, 224)  
INPUT_SHAPE     = tuple(list(IMAGE_SIZE) + [3])
LEARNING_RATE   = 1e-4
NUM_CLASSES     = 8
AUGMENT = False
N_GRADIENTS     = 0


# Parameter spesifik arsitektur
MODEL_CONFIG = {
    'model_name' : 'originalnasnetmobile'
    }
RESULT_PATH = f"../training_result/exp1/{MODEL_CONFIG['model_name']}/{CFG_NUM}"

LOGDIR = path.join(RESULT_PATH, "logs")
# Args untuk callbacks
SAVE_INTERVAL   = 10
HISTORY_FILENAME= 'history_at_epoch{epoch}.pkl'
MODEL_FILENAME= 'latest_model.tf'
BEST_MODEL_FILENAME= 'best_model.tf'

# Callbacks 
CALLBACKS_CONFIG = {
    # 'history_saver' : {
    #     'interval' : SAVE_INTERVAL,
    #     'save_path' : path.join(RESULT_PATH, HISTORY_FILENAME)
    # },
    # # 'model_checkpoint' : {
    # #     'interval' : SAVE_INTERVAL,
    # #     'save_path' : path.join(RESULT_PATH, MODEL_FILENAME)
    # # },
    'save_best':{
        'save_path' : path.join(RESULT_PATH, BEST_MODEL_FILENAME),
        'monitor': 'val_loss'
    }
}
