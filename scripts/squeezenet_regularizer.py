# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:25:56 2024

@author: rafae
"""

import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', 'src')
configs_dir = os.path.join(working_dir, '..', 'config')
sys.path.append(src_dir)
sys.path.append(configs_dir)

# Set seed
from utils import general_utils as gutils
# gutils.set_determinism(42)
gutils.use_mixed_precision()

import time
from configs.other_configs import data_info as dinfo
from tensorflow.keras import optimizers, metrics, callbacks, layers, regularizers 
from mymodels.squeezenet import MySqueezeNet
from utils import training_utils as tutils

# Params
epoch       = 500
lr          = 1e-4
batch_size  = 16*6
wdecay      = 2e-4

# Paths
PARAM_VAL               = f'l2+decay{wdecay}'
TUNED_PARAM             = 'regularizer'
SAVE_PATH               = f'../training_result/squeezenet/{TUNED_PARAM}/{PARAM_VAL}'
BEST_MODEL_FILENAME     = 'best_model_epoch_{epoch}.tf'
LAST_MODEL_FILENAME     = 'model_at_{epoch}.tf'
LATEST_MODEL_FILENAME   = 'latest_model.tf'
LOGDIR                  = os.path.join(SAVE_PATH, f"logs/run_{int(time.time())}")


# Load data
train_data_dir = os.path.join(dinfo.DATA_PATH, 'train')
val_data_dir = os.path.join(dinfo.DATA_PATH, 'validation')
train_datagen = gutils.make_datagen(train_data_dir, 
                                    dinfo.IMG_RES, 
                                    batch_size,
                                    augment=False)
val_datagen = gutils.make_datagen(val_data_dir, 
                                  dinfo.IMG_RES, 
                                  batch_size,
                                  shuffle=False)

# Definisi model
model = MySqueezeNet(dinfo.IMG_DIM, dinfo.NUM_CLASSES)
model.build_model(include_classification_head=False, include_top=False,
                  pooling=None)
top_layers = [
    layers.Dropout(0.5, name='drop9'),
    layers.Convolution2D(dinfo.NUM_CLASSES, (1, 1), padding='valid', 
                         kernel_regularizer=regularizers.l2(wdecay),
                          name='conv10'),
    layers.Activation('relu', name='relu_conv10'),
    layers.GlobalAveragePooling2D(),
    layers.Activation('softmax', name='loss')
]
model.add_layers(top_layers)

# Callbacks
model_callbacks = [
    callbacks.TensorBoard(log_dir=LOGDIR),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(SAVE_PATH, BEST_MODEL_FILENAME),
        monitor='val_loss',
        save_best_only=True,
        mode='min', 
        save_weights_only=False
    ),
    # callbacks.ModelCheckpoint(
    #     filepath=os.path.join(SAVE_PATH, LATEST_MODEL_FILENAME),
    #     save_best_only=False,
    #     save_weights_only=False
    # )
]

# Compile
optimizer = optimizers.Adam(learning_rate=lr)
metrics = [
    'accuracy',
    metrics.Recall(name='recall')    
    ]
model.compile_model(optimizer=optimizer, metrics=metrics)


# Train
model.train(train_datagen,
            val_datagen, 
            epochs=epoch,
            batch_size=batch_size,
            callbacks=model_callbacks)

model.save_model(os.path.join(SAVE_PATH, LAST_MODEL_FILENAME.format(epoch=epoch)))

# # Train lagi ke 500
# prev_epoch = epoch
# epoch = 500
# model.train(train_datagen,
#             val_datagen, 
#             epochs=epoch,
#             batch_size=batch_size,
#             callbacks=model_callbacks,
#             initial_epoch=prev_epoch
#             )

# model.save_model(os.path.join(SAVE_PATH, LAST_MODEL_FILENAME.format(epoch=epoch)))

# # Tambahkan layer dense
# model.add_layers