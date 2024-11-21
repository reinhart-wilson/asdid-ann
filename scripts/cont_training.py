# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:36:04 2024

@author: reinh
"""

import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', 'src')
configs_dir = os.path.join(working_dir, '..', 'config')
sys.path.append(src_dir)
sys.path.append(configs_dir)


import logging
from configs.experiment_configs.exp1 import mobilenetv2_cfg as config
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
import tensorflow as tf
from utils import general_utils as gutils
from utils import training_utils as tutils
from tensorflow.keras import optimizers


# Enable GPU (harus dilakukan sebelum import TensorFlow)
# gutils.use_gpu(True)


#Disable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

# Load data
train_data_dir = os.path.join(config.DATA_PATH, 'train')
# train_data_dir = dataloc.ADDITIONAL_DATA_PATH
val_data_dir = os.path.join(config.DATA_PATH, 'validation')
# val_data_dir=dataloc.ADDITIONAL_DATA_PATH
train_datagen = gutils.make_datagen(train_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE,
                                    augment=config.AUGMENT)
val_datagen = gutils.make_datagen(val_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE)


# Mendapatkan learning rate sebelum training terputus
# log_dir = os.path.join(config.RESULT_PATH, 'logs','train')
# events_filename = 'events.out.tfevents.1731377552.LAPTOP-I8SDHTDM.11400.3.v2'
# last_lr = float(tutils.get_last_lr_from_tensorboard(os.path.join(
#     log_dir,
#     events_filename
#     )))

# Menghitung sisa step berdasarkan sisa epoch
at_epoch=28
remaining_epochs = config.EPOCHS - at_epoch
# decay_steps = tutils.compute_decay_steps(
#             train_datagen.samples,
#             config.BATCH_SIZE,
#             remaining_epochs
#         )

# Adjust the learning rate scheduler to start from the current learning rate
# current_learning_rate = last_lr
# LR_SCHEDULE_RESUME = optimizers.schedules.CosineDecay(
#     initial_learning_rate=current_learning_rate,
#     alpha=config.LR_ALPHA,
#     decay_steps=decay_steps
# )
# optimizer_resume = optimizers.SGD(learning_rate=LR_SCHEDULE_RESUME, momentum=0.9)

#Continue training
model_path = os.path.join(
    config.RESULT_PATH, 
    config.MODEL_FILENAME.format(epoch=at_epoch)
    )
model = load_model(model_path) 
# model.compile(optimizer=optimizer_resume, 
#               loss=model.loss, 
#               metrics=['accuracy'])
generated_callbacks = tutils.generate_callbacks(config.CALLBACKS_CONFIG)
model_callbacks = generated_callbacks if generated_callbacks is not None else [] 
tensorboard_callback = callbacks.TensorBoard(log_dir=config.LOGDIR)
model_checkpoint = callbacks.ModelCheckpoint(
    os.path.join(config.RESULT_PATH, config.MODEL_FILENAME),
    verbose=0
    )
model_callbacks.append(tensorboard_callback)
model_callbacks.append(model_checkpoint)
tutils.train_model(model, train_datagen, val_datagen, config.EPOCHS, 
                    config.RESULT_PATH, last_epoch=at_epoch, callbacks=model_callbacks)