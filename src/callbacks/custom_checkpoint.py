# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:22:14 2024

@author: reinh
"""

from keras.callbacks import ModelCheckpoint

"""
Subkelas ModelCheckpoimt yang digunakan khusus untuk menyimpan model setiap
interval yang ditentukan. Dengan callback ini, pelatihan dapat dihentikan 
lalu dilanjutkan kemudian dengan memuat model terakhir.
"""
class CustomCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, save_interval, *args, **kwargs):
        super(CustomCheckpoint, self).__init__(filepath, *args, **kwargs)
        self.save_interval = save_interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_interval == 0:
            super(CustomCheckpoint, self).on_epoch_end(epoch, logs)
