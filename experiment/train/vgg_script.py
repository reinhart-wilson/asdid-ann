# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:03:54 2024

@author: reinh
"""

import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)

# Import file konfigurasi
from configs.vgg16_cfg import config1 as config

# Set seed untuk beberapa library python agar hasil deterministik
from utils import general_utils as gutils
from utils import training_utils as tutils
gutils.use_gpu(config.USE_GPU)
gutils.set_determinism(config.SEED)

# Import packages lainnya yang diperlukan
import signal
import pickle
from callbacks.callbacks_factory import create_callback
from keras import optimizers
from keras.models import load_model
from mymodels.model_factory import create_model


# ===============================

def generate_callbacks(callback_configs):
    callbacks = []
    for cb_name, cfg_dict in callback_configs.items():
        callback = create_callback(cb_name, cfg_dict)
        callbacks.append(callback)
    if len(callbacks) == 0:
        return None
    return callbacks

def load_data():
    train_data_dir = os.path.join(config.DATA_PATH, 'train')
    val_data_dir = os.path.join(config.DATA_PATH, 'validation')
    train_datagen = gutils.make_datagen(train_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE)
    val_datagen = gutils.make_datagen(val_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE)
    return train_datagen, val_datagen

def train():    
    #
    callbacks = generate_callbacks(config.CALLBACKS_CONFIG)
    
    #
    optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE)

    # Muat data
    train_datagen, val_datagen = load_data()
    
    # Training
    model = create_model(config.INPUT_SHAPE, config.NUM_CLASSES, config.MODEL_CONFIG)
    model.build_model()
    model.compile_model(optimizer=optimizer)
    # model.show_summary()
    model.train(train_datagen, val_datagen, epochs = config.EPOCHS, 
                batch_size=config.BATCH_SIZE,callbacks=callbacks)
    
    # Release memory
    signal.signal(signal.SIGINT, gutils.clean_memory)
    
def resume_train(at_epoch):
    #
    callbacks = generate_callbacks(config.CALLBACKS_CONFIG)
    
    # Muat data
    train_datagen, val_datagen = load_data()
    
    model_path = os.path.join(config.RESULT_PATH, f"model_at_epoch{at_epoch}.keras")
    model = load_model(model_path)    
    interrupted_history_path = os.path.join(config.RESULT_PATH, 
                                      f"history_at_epoch{at_epoch}.pkl")
    with open(interrupted_history_path,'rb') as file:
        interrupted_history = pickle.load(file)
        
    tutils.train_model(model, train_datagen, val_datagen, 100, 
                       config.RESULT_PATH, last_epoch=at_epoch, 
                       last_history=interrupted_history,callbacks=callbacks)
    
if __name__ == "__main__":
    train()
    # resume_train(60)
