from os import path

SEED = 42
CFG_NUM = 3
USE_GPU= True
DATA_PATH = "../../dataset/split_augmented_data"

# Parameter umum
BATCH_SIZE      = 10
EPOCHS          = 300 
IMAGE_SIZE      = (224, 224)  
INPUT_SHAPE     = tuple(list(IMAGE_SIZE) + [3])
LEARNING_RATE   = 0.0001
NUM_CLASSES     = 8


# Parameter spesifik arsitektur
MODEL_CONFIG = {
    'model_name': 'mobilenetv2',
    'alpha'     : 1.0,
    'dense'     : 1024
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
