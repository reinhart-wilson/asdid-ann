# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 22:05:18 2024

@author: reinh
"""

from abc import ABC, abstractmethod
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

class AbstractCNN(ABC):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    @abstractmethod
    def build_model(self):
        pass
    
    def compile_model(self, 
                      optimizer=Adam(), 
                      loss='categorical_crossentropy',
                      metrics=['accuracy']):
        if self.model is None:
            self.build_model()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train(self, train_data, validation_data, epochs=10, initial_epoch = 0, 
              batch_size=16, callbacks=None, class_weights=None):
        if self.model is None:
            self.build_model()
        history = self.model.fit(train_data, 
                       validation_data=validation_data, 
                       epochs=epochs, 
                       batch_size=batch_size, 
                       initial_epoch=initial_epoch,
                       callbacks=callbacks,
                       class_weight=class_weights)
        return history
    
    def evaluate(self, test_data):
        if self.model is None:
            raise ValueError("Model has not been built.")
        return self.model.evaluate(test_data)
    
    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Model has not been built.")
        return self.model.predict(input_data)
    
    def save_model(self, file_path):
        if self.model is None:
            raise ValueError("Model has not been built.")
        self.model.save(file_path)
    
    def load_model(self, file_path):
        self.model = load_model(file_path)
        
    def show_summary(self):
        self.model.summary()
        
    def get_tf_model (self):
        return self.model

        
