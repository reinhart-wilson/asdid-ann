# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 20:53:27 2024

@author: reinh
"""

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2B0  # Import EfficientNetV2
from .abstract_cnn import AbstractCNN

class EfficientNetV2(AbstractCNN):  # Renamed the class for clarity
    def __init__(self, input_shape, num_classes, alpha=1.0, 
                 dense_neuron_num=1024, model_variant='large'):
        super(EfficientNetV2, self).__init__(input_shape, num_classes)
        self.dense_neuron_num = dense_neuron_num

    def build_model(self):
        
        # Build the EfficientNetV2 model without pre-trained weights
        base_model = EfficientNetV2B0(weights=None, include_top=False, 
                                      input_shape=(224, 224, 3))
        
        # Classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Add global average pooling
        x = Dense(self.dense_neuron_num, activation='relu')(x)  # Add a fully connected layer
        predictions = Dense(self.num_classes, activation='softmax')(x)  # Add the final output layer
        
        # This is the model we will train from scratch
        self.model = Model(inputs=base_model.input, outputs=predictions)
