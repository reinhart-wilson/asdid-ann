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
        # Build the EfficientNetB0 model without pre-trained weights
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=self.input_shape)
        
        # Add Global Average Pooling (as per the original EfficientNet architecture)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Add the final output layer (softmax for classification)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # This is the final model based on the original EfficientNet-B0 structure
        self.model = Model(inputs=base_model.input, outputs=predictions)