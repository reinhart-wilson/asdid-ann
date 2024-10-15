# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 22:00:12 2024

@author: reinh
"""

from keras.applications import MobileNet
from keras import models, layers
from .abstract_cnn import AbstractCNN

class MyMobileNetV1(AbstractCNN):
    def __init__(self, input_shape, num_classes, alpha=1.0, 
                 dense_neuron_num=1024):
        super(MyMobileNetV1, self).__init__(input_shape, 
                                            num_classes)
        self.dense_neuron_num = dense_neuron_num
        self.alpha = alpha
    
    def build_model(self):
        # Arsitektur MobileNet dari Keras sebagai feature extractor
        base_model = MobileNet(include_top=False,
                                weights=None,
                                input_shape=self.input_shape, 
                                alpha=self.alpha)
        
        # Classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.dense_neuron_num, activation='relu')(x) 
        x = layers.Dropout(0.5)(x)  
        predictions = layers.Dense(self.num_classes, 
                                    activation='softmax')(x) 

        self.model = models.Model(inputs=base_model.input, 
                                  outputs=predictions)

        
class OriginalMobileNet(AbstractCNN):
    
    def __init__(self, input_shape, num_classes, alpha=1.0, 
                 dense_neuron_num=1024):
        super(OriginalMobileNet, self).__init__(input_shape, 
                                            num_classes)
        self.dense_neuron_num = dense_neuron_num
        self.alpha = alpha
    
    def build_model(self):
        # Arsitektur MobileNet dari Keras sebagai feature extractor
        base_model = MobileNet(include_top=False,
                               weights=None,
                               input_shape=self.input_shape, 
                               alpha=self.alpha)
        
        # Classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        predictions = layers.Dense(self.num_classes, activation='softmax')(x)


        self.model = models.Model(inputs=base_model.input, 
                                  outputs=predictions)