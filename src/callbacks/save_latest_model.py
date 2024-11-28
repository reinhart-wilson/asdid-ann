# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:17:36 2024

@author: reinh
"""

import keras
from keras import callbacks 

class SaveLatestModel(callbacks.Callback):
    def __init__(self, modelSaveLocation):
        super(keras.callbacks.Callback, self).__init__()
        self.modelSaveLocation = modelSaveLocation

    def on_epoch_end(self, epoch, logs={}):
        self.model.save(self.modelSaveLocation, 
                        overwrite=True)