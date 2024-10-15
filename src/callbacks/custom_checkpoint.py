# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:22:14 2024

@author: reinh
"""

from keras.callbacks import ModelCheckpoint

class CustomCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, save_interval, *args, **kwargs):
        super(CustomCheckpoint, self).__init__(filepath, *args, **kwargs)
        self.save_interval = save_interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_interval == 0:  # Check if epoch is a multiple of save_interval
            super(CustomCheckpoint, self).on_epoch_end(epoch, logs)
