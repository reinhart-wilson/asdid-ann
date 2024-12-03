# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:10:18 2024

@author: reinh
"""

import sys, os

working_dir = os.path.abspath(os.path.dirname(__file__))
configs_dir = os.path.join(working_dir, '..', '..', 'config')
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)
sys.path.append(configs_dir)

# Set mixed precision
from utils import general_utils as gutils
gutils.use_mixed_precision()

from callbacks.save_latest_model import SaveLatestModel
from callbacks.learning_rate_logger import LearningRateLogger
from configs.other_configs import data_info as dinfo
from mymodels.squeezenet import MySqueezeNet
from tensorflow.keras import optimizers, metrics, callbacks, layers, regularizers 
from utils import training_utils as tutils

# Params
mode        = 0 # 0 cont 1 train
last_epoch  = 108
epoch       = 200
batch_size  = 96
dense       = 512*1 
dropout     = 0
weights     = None
wdecay      = 4e-5
alpha       = 1.0 # Bukan learning rate
augment     = True
optimizer_name = 'adam'
momentum    = None
lr_config   = {
    'init_value' : 1e-4,
    'scheduler_config' : {
        'name' : 'cosine_decay',
        'lr_alpha' : 1e-2,
        'epochs_to_decay' : epoch
    }
}

# Paths
PARAM_VAL               = 1
TUNED_PARAM             = 'final'
SAVE_PATH               = f'../../training_result/shufflenet/{TUNED_PARAM}/{PARAM_VAL}'
BEST_MODEL_FILENAME     = 'best_model_epoch.tf'
LAST_MODEL_FILENAME     = 'model_at_{epoch}.tf'
LATEST_MODEL_FILENAME   = 'latest_model.tf'
LOGDIR                  = os.path.join(SAVE_PATH, "logs")

# Load data
train_data_dir = os.path.join(dinfo.EXTENDED_DATA_PATH, 'train')
val_data_dir = os.path.join(dinfo.EXTENDED_DATA_PATH, 'validation')
train_datagen = gutils.make_datagen(train_data_dir, 
                                    dinfo.IMG_RES, 
                                    batch_size,
                                    augment=augment)
val_datagen = gutils.make_datagen(val_data_dir, 
                                  dinfo.IMG_RES, 
                                  batch_size,
                                  shuffle=False)

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
    SaveLatestModel(os.path.join(SAVE_PATH, LATEST_MODEL_FILENAME)),
    LearningRateLogger()
]

# Metrics
model_metrics = [
    'accuracy',
    metrics.Recall(name='recall')    
    ]

# Definisi model
model = MySqueezeNet(dinfo.IMG_DIM, dinfo.NUM_CLASSES)

if mode == 1:
    # Definisi model
    model = MySqueezeNet(dinfo.IMG_DIM, dinfo.NUM_CLASSES)
    model.build_model(include_classification_head=True, include_top=True,
                      pooling=None)
    # top_layers = [
    #     layers.Dropout(0.5, name='drop9'),
    #     layers.Convolution2D(dinfo.NUM_CLASSES, (1, 1), padding='valid', 
    #                          kernel_regularizer=regularizers.l2(wdecay),
    #                           name='conv10'),
    #     layers.Activation('relu', name='relu_conv10'),
    #     layers.GlobalAveragePooling2D(),
    #     layers.Activation('softmax', name='loss')
    # ]
    # model.add_layers(top_layers)
    
    
    # Learning Rate
    if 'scheduler_config' in lr_config:
        lr_scheduler_config = lr_config['scheduler_config']
        if lr_scheduler_config['name'] == 'cosine_decay':
            decay_steps = tutils.compute_decay_steps(
                train_datagen.samples,                                                  
                batch_size,                                                  
                lr_scheduler_config['epochs_to_decay'])
            lr = optimizers.schedules.CosineDecay(
                initial_learning_rate=lr_config['init_value'],
                alpha = lr_scheduler_config['lr_alpha'],
                decay_steps = decay_steps
            )
    else:
        lr = lr_config['init_value']
    
    # Compile
    optimizer = optimizers.Adam(learning_rate=lr)

    model.compile_model(optimizer=optimizer, metrics=model_metrics)
    
    # Train
    model.train(train_datagen,
                val_datagen, 
                epochs=epoch,
                batch_size=batch_size,
                callbacks=model_callbacks)
    model.save_model(os.path.join(SAVE_PATH, LAST_MODEL_FILENAME.format(epoch=epoch)))
if mode == 0:
    lr_scheduler_config = lr_config['scheduler_config']
    
    # Calculate total decay steps
    total_decay_steps = tutils.compute_decay_steps(
        train_datagen.samples,
        batch_size,
        lr_scheduler_config['epochs_to_decay']
    )

    # Calculate the number of completed steps
    completed_steps = last_epoch * (train_datagen.samples // batch_size)

    # Resume CosineDecay schedule
    if lr_scheduler_config['name'] == 'cosine_decay':
        LR_SCHEDULE_RESUME = optimizers.schedules.CosineDecay(
            initial_learning_rate=lr_config['init_value'],
            alpha=lr_scheduler_config['lr_alpha'],
            decay_steps=total_decay_steps
        )
        # Update the learning rate to the current step
        for _ in range(completed_steps):
            current_learning_rate = LR_SCHEDULE_RESUME(completed_steps)
        learning_rate = current_learning_rate

    # Define optimizer
    if optimizer_name == 'sgd':
        optimizer_resume = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name == 'adam':
        optimizer_resume = optimizers.Adam(learning_rate=learning_rate)

    # Load the latest model and continue training
    model.load_model(os.path.join(SAVE_PATH, LATEST_MODEL_FILENAME))
    model.compile_model(optimizer=optimizer_resume, metrics=model_metrics)
    model.train(
        train_datagen,
        val_datagen,
        epochs=epoch,
        batch_size=batch_size,
        callbacks=model_callbacks,
        initial_epoch=last_epoch
    )
