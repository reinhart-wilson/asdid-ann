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
        
        # Build the EfficientNet model without pre-trained weights
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        
        # Add your own layers on top of it
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Add global average pooling
        x = Dense(self.dense_neuron_num, activation='relu')(x)  # Add a fully connected layer
        predictions = Dense(self.num_classes, activation='softmax')(x)  # Add the final output layer
        
        # This is the model we will train from scratch
        self.model = Model(inputs=base_model.input, outputs=predictions)