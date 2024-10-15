# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:09:20 2024

@author: reinh
"""

from keras.applications import MobileNetV3Large, MobileNetV3Small
from keras import models, layers
from .abstract_cnn import AbstractCNN
from keras.regularizers import l2

class MyMobileNetV3(AbstractCNN):
    def __init__(self, input_shape, num_classes, alpha=1.0, 
                 dense_neuron_num=1024, model_variant='large', dropout=0, 
                 weights=None,
                 weight_decay=0):
        super(MyMobileNetV3, self).__init__(input_shape, num_classes)
        self.dense_neuron_num = dense_neuron_num
        self.alpha = alpha
        self.model_variant = model_variant  # Add option to select between MobileNetV3Small and MobileNetV3Large
        if weights not in [None, 'imagenet']:
            raise ValueError("Weights not recognized.")
        self.weights = weights
        self.weight_decay = weight_decay 
        self.dropout = dropout
        
    def build_model(self):
        # Select MobileNetV3 model variant (large or small)
        if self.model_variant == 'large':
            base_model = MobileNetV3Large(include_top=False,
                                          weights=self.weights, 
                                          input_shape=self.input_shape, 
                                          alpha=self.alpha)
        else:
            base_model = MobileNetV3Small(include_top=False,
                                          weights=self.weights, 
                                          input_shape=self.input_shape, 
                                          alpha=self.alpha)
        
        # Classification head (Original MobileNetV3 Architecture)
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)

        # Final classification layer (MobileNetV3 uses softmax for classification)
        predictions = layers.Dense(self.num_classes, activation='softmax')(x)
    
        self.model = models.Model(inputs=base_model.input, outputs=predictions)
        
        # def build_model(self):
        #     # Select MobileNetV3 model variant (large or small)
        #     if self.model_variant == 'large':
        #         base_model = MobileNetV3Large(include_top=False,
        #                                       weights=None, 
        #                                       input_shape=self.input_shape, 
        #                                       alpha=self.alpha)
        #     else:
        #         base_model = MobileNetV3Small(include_top=False,
        #                                       weights=None, 
        #                                       input_shape=self.input_shape, 
        #                                       alpha=self.alpha)
            
        #     # Classification head
        #     x = base_model.output
        #     x = layers.GlobalAveragePooling2D()(x)
        #     x = layers.Dense(self.dense_neuron_num, activation='relu')(x) 
        #     x = layers.Dropout(0.5)(x)  
        #     predictions = layers.Dense(self.num_classes, 
        #                                activation='softmax')(x) 
    
        #     self.model = models.Model(inputs=base_model.input, 
        #                               outputs=predictions)
    
    # def build_model(self):
    #     # Select MobileNetV3 model variant (large or small)
    #     if self.model_variant == 'large':
    #         base_model = MobileNetV3Large(include_top=False,
    #                                       weights=None, 
    #                                       input_shape=self.input_shape, 
    #                                       alpha=self.alpha)
    #     else:
    #         base_model = MobileNetV3Small(include_top=False,
    #                                       weights=None, 
    #                                       input_shape=self.input_shape, 
    #                                       alpha=self.alpha)
        
    #     # Classification head
    #     x = base_model.output
    #     x = layers.GlobalAveragePooling2D()(x)
    #     x = layers.Dense(self.dense_neuron_num, activation=None, 
    #                      kernel_regularizer=l2(self.weight_decay))(x)  # Apply L2 regularization
    #     x = layers.BatchNormalization()(x)
    #     x = layers.ReLU()(x)
    #     if self.dropout>0:
    #         x = layers.Dropout(self.dropout)(x)  # Adjust dropout rate if needed
    #     predictions = layers.Dense(self.num_classes, activation='softmax')(x)

    #     # Create the model
    #     self.model = models.Model(inputs=base_model.input, outputs=predictions)
