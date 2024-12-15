# -*- coding: utf-8 -*-
"""
Created on Tue Jun  

@author: reinh
"""

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from .abstract_cnn import AbstractCNN

class EfficientNet(AbstractCNN):
    def build_model(self):
        base_model = EfficientNetB0(weights=None, include_top=False, 
                                    input_shape=self.input_shape)
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)