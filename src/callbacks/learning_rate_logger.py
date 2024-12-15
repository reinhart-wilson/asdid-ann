# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:00:15 2024

@author: reinh
"""


from keras import backend, callbacks
from tensorflow.keras import optimizers
from tensorflow import summary

"""
Merupakan subkelas dari Callback keras untuk mencatat learning rate di setiap
akhir epoch. Berguna untuk memantau learning rate ketika scheduler digunakan.
"""
class LearningRateLogger(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        logs['lr'] = backend.get_value(lr)
        summary.scalar('learning rate', data=lr, step=epoch)