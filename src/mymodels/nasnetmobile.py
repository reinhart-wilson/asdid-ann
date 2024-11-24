# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:03:20 2024

@author: reinh
"""

from keras import layers
from keras import models
from .abstract_cnn import AbstractCNN
from .base_models.nasnet import NASNetMobile

class OriginalNasNetMobile(AbstractCNN):   
    
    def build_model(self):            
        base_model = NASNetMobile(
            include_top=False, 
            weights=None,
            input_shape=self.input_shape,
            pooling='avg',
            classes=self.num_classes,
            name='nasnetmobile'
            )
        
        # Classification head
        x = base_model.output
        predictions = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=base_model.input, 
                                  outputs=predictions)
        
        
