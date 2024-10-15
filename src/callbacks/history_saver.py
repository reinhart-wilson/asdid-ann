# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:15:40 2024

@author: reinh
"""

import pickle
import os
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers, backend

class HistorySaver(Callback):
    def __init__(self, save_interval, history_path_template, initial_epoch=0, 
                 save_lr = False, read_only=True):
        super(HistorySaver, self).__init__()
        self.save_interval = save_interval
        self.history_path_template = history_path_template
        self.initial_epoch = initial_epoch
        self.combined_history = {'loss': [], 'val_loss': [], 'accuracy': [], 
                                 'val_accuracy': [], 'lr': []}
        self.read_only=read_only
        self.save_lr=save_lr
        directory = os.path.dirname(history_path_template)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            if self.save_lr:
                lr = self.model.optimizer.learning_rate
                if isinstance(lr, optimizers.schedules.LearningRateSchedule):
                    lr = lr(self.model.optimizer.iterations)
                logs['lr'] = backend.get_value(lr)
                
            for key, value in logs.items():
                if key in self.combined_history:
                    self.combined_history[key].append(value)
                else:
                    self.combined_history[key] = [value]
        
        if (epoch + 1) % self.save_interval == 0:
            # Save history
            history_path = self.history_path_template.format(epoch=epoch + 1)
            with open(history_path, 'wb') as file:
                pickle.dump(self.combined_history, file)
                if self.read_only:
                    os.chmod(history_path, 0o444)

    def on_train_begin(self, logs=None):
        history_path = self.history_path_template.format(epoch=self.initial_epoch)
        if os.path.exists(history_path):
            with open(history_path, 'rb') as file:
                self.combined_history = pickle.load(file)
