# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:59:47 2024

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

# Set seed
from utils import general_utils as gutils
# gutils.set_determinism(42)
gutils.use_mixed_precision()

from configs.other_configs import data_info as dinfo
from tensorflow.keras import optimizers, metrics, callbacks
from mymodels.squeezenet import MySqueezeNet
from utils import training_utils as tutils

# Paths
EXP_NUM                 = 1
SAVE_PATH               = f'../training_result/squeezenet/{EXP_NUM}'
BEST_MODEL_FILENAME     = 'best_model.tf'
LATEST_MODEL_FILENAME   = 'last_model.tf'
LOGDIR                  = os.path.join(SAVE_PATH, 'logs')

# Params
epoch       = 100
lr          = 1e-4
batch_size  = 64
dense       = 128 * 1# Banyak unit di dense layer

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
model.build_model(include_classification_head=True)

# Callbacks
model_callbacks = [
    callbacks.TensorBoard(log_dir=LOGDIR),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(SAVE_PATH, BEST_MODEL_FILENAME),
        monitor='val_loss',
        save_best_only=True,
        mode='min', 
        save_weights_only=False
    )
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


# # Tambahkan layer dense
# model.add_layers

