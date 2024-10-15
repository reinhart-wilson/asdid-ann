# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:17:21 2024

@author: reinh
"""

import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)

# Import file konfigurasi
from configs.mobilenetv3_cfg import config_large_v2imagenet1 as config

# Set seed untuk beberapa library python agar hasil deterministik
from utils import general_utils as gutils
from utils import training_utils as tutils
import tensorflow as tf

# Configure session with allow_soft_placement
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
session = tf.compat.v1.Session(config=tfconfig)
gutils.use_gpu(config.USE_GPU)
gutils.set_determinism(config.SEED)

# Import packages lainnya yang diperlukan
import signal
import pickle
from callbacks.callbacks_factory import create_callback
from mymodels.model_factory import create_model
from keras.models import load_model

#temp
from tensorflow.keras import optimizers

# ===============================

def generate_callbacks(callback_configs):
    callbacks = []
    for cb_name, cfg_dict in callback_configs.items():
        callback = create_callback(cb_name, cfg_dict)
        callbacks.append(callback)
    if len(callbacks) == 0:
        return None
    return callbacks

def load_data(augment=False):
    train_data_dir = os.path.join(config.DATA_PATH, 'train')
    val_data_dir = os.path.join(config.DATA_PATH, 'validation')
    train_datagen = gutils.make_datagen(train_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE,
                                        augment=augment)
    val_datagen = gutils.make_datagen(val_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE)
    return train_datagen, val_datagen

def compute_decay_steps(num_samples, batch_size, epochs):
    # Calculate the number of steps per epoch
    steps_per_epoch = num_samples // batch_size
    
    # Total number of steps (across all epochs)
    decay_steps = steps_per_epoch * epochs
    
    return decay_steps 

def train():    
    #
    callbacks = generate_callbacks(config.CALLBACKS_CONFIG)
    

    # Muat data
    train_data_dir = os.path.join(config.DATA_PATH, 'train')
    val_data_dir = os.path.join(config.DATA_PATH, 'validation')
    train_datagen = gutils.make_datagen(train_data_dir, 
                                        config.IMAGE_SIZE, 
                                        config.BATCH_SIZE,
                                        augment=config.AUGMENT)
    val_datagen = gutils.make_datagen(val_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE)
    
    #temp
    LR_SCHEDULE = optimizers.schedules.CosineDecay(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=compute_decay_steps(train_datagen.samples, config.BATCH_SIZE, 
                                        config.EPOCHS)
    )
    OPTIMIZER = optimizers.SGD(learning_rate=LR_SCHEDULE, momentum=0.9)
    
    
    # Training
    model = create_model(config.INPUT_SHAPE, config.NUM_CLASSES, config.MODEL_CONFIG)
    model.build_model()
    model.compile_model(optimizer=OPTIMIZER)
    model.train(train_datagen, val_datagen, epochs = config.EPOCHS, 
                batch_size=config.BATCH_SIZE,callbacks=callbacks)
    
    # Release memory
    signal.signal(signal.SIGINT, gutils.clean_memory)
    
def resume_train(at_epoch):
    #
    callbacks = generate_callbacks(config.CALLBACKS_CONFIG)
    
    # Muat data
    train_datagen, val_datagen = load_data(augment=config.AUGMENT)
    
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
    # resume_train(4)