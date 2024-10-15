"""
Konfigurasi yang terinspirasi oleh konfigurasi MobileNetV2 yang dipakai untuk 
tugas klasifikasi ImageNet
"""

from os import path
from tensorflow.keras import optimizers

SEED = 42
CFG_NUM = 'imagenet1'
USE_GPU= False
DATA_PATH = "../../dataset/split_prepped_data"

# Parameter umum
IMAGE_SIZE      = (224, 224)  
INPUT_SHAPE     = tuple(list(IMAGE_SIZE) + [3])
NUM_CLASSES     = 8
BATCH_SIZE      = 16
EPOCHS          = 100 
LEARNING_RATE   = 0.045
DECAY_STEPS     = EPOCHS



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
SAVE_INTERVAL   = 2
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
