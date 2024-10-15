# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:56:59 2024

@author: reinh
"""

from keras.applications import MobileNetV2
from keras import models, layers
from keras.regularizers import l2
from .abstract_cnn import AbstractCNN
from .custom_gradient_accumulation import CustomGradientAccumulation as CustomGA

class MyMobileNetV2(AbstractCNN):
    def __init__(self, input_shape, num_classes, alpha=1.0, 
                 dense_neuron_num=1024, dropout=0, weights=None,
                 weight_decay=0, n_gradients = 0):  # Added weight_decay parameter
        super(MyMobileNetV2, self).__init__(input_shape, num_classes)
        self.dense_neuron_num = dense_neuron_num
        self.alpha = alpha
        self.dropout = dropout
        if weights not in [None, 'imagenet']:
            raise ValueError("Weights not recognized.")
        self.weights = weights
        self.weight_decay = weight_decay  # Store the weight decay parameter
        self.n_gradients = n_gradients
    
    def build_model(self):
        # MobileNetV2 architecture from Keras as the base model (feature extractor)
        base_model = MobileNetV2(include_top=False,
                                  weights=self.weights, 
                                  input_shape=self.input_shape, 
                                  alpha=self.alpha)
        
        # Classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.dense_neuron_num, activation=None, 
                          kernel_regularizer=l2(self.weight_decay))(x)  # Apply L2 regularization
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if self.dropout>0:
            x = layers.Dropout(self.dropout)(x)  # Adjust dropout rate if needed
        predictions = layers.Dense(self.num_classes, activation='softmax')(x)
        
        if self.n_gradients>0:
        # Menggunakan CustomTrainStep sebagai model kustom dengan akumulasi gradien
            self.model = CustomGA(n_gradients=self.n_gradients, 
                                  inputs=base_model.input, 
                                  outputs=predictions)
        else:
            self.model = models.Model(inputs=base_model.input, outputs=predictions)
        # self.model.summary()  # Print model summary for verification

class OriginalMobileNetV2(AbstractCNN):
    
    def __init__(self, input_shape, num_classes, alpha=1.0, 
                 dense_neuron_num=1024, dropout=0, weights=None,
                 weight_decay=0):  # Added weight_decay parameter
        super(OriginalMobileNetV2, self).__init__(input_shape, num_classes)
        self.dense_neuron_num = dense_neuron_num
        self.alpha = alpha
        if weights not in [None, 'imagenet']:
            raise ValueError("Weights not recognized.")
        self.weights = weights  # Store the weight decay parameter

    def build_model(self):
        # Arsitektur MobileNet dari Keras sebagai base model (feature extractor)
        base_model = MobileNetV2(include_top=False,
                                  weights=None, 
                                  input_shape=self.input_shape, 
                                  alpha=self.alpha)
        
        # Classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        predictions = layers.Dense(self.num_classes, activation='softmax')(x)

        # Create the model
        self.model = models.Model(inputs=base_model.input, outputs=predictions)