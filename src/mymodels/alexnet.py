# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:57:20 2024

@author: reinh
"""

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model
from .abstract_cnn import AbstractCNN

class AlexNet(AbstractCNN):
    def build_model(self):
        input_layer = Input(shape=self.input_shape)
        
        # 1st Convolutional Layer
        x = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu')(input_layer)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)  # Pooling size changed to 3x3

        # 2nd Convolutional Layer
        x = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x)  # Increased filters
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

        # 3rd Convolutional Layer
        x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

        # 4th Convolutional Layer
        x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

        # 5th Convolutional Layer
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)  # Pooling after 5th conv layer

        # Fully Connected Layers
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Output Layer
        output_layer = Dense(self.num_classes, activation='softmax')(x)

        # Create the model
        self.model = Model(inputs=input_layer, outputs=output_layer)


# class AlexNet(AbstractCNN):
#     def build_model(self):
#         input_layer = Input(shape=self.input_shape)
        
#         # Convolutional Layer
#         x = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu')(input_layer)
#         x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
        
#         # Convolutional Layer dengan Group Convolution
#         x1 = Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu')(x[:, :, :, :48])
#         x2 = Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu')(x[:, :, :, 48:])
#         x = concatenate([x1, x2], axis=-1)
#         x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
        
#         # Convolutional Layer
#         x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x)
        
#         # Convolutional Layer dengan Group Convolution
#         x1 = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x[:, :, :, :192])
#         x2 = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x[:, :, :, 192:])
#         x = concatenate([x1, x2], axis=-1)
        
#         # Convolutional Layer dengan Group Convolution
#         x1 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x[:, :, :, :192])
#         x2 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x[:, :, :, 192:])
#         x = concatenate([x1, x2], axis=-1)
#         x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)

#         # Fully Connected layer
#         x = Flatten()(x)
        
#         # Fully Connected Layer
#         x = Dense(4096, activation='relu')(x)
#         # Add Dropout to prevent overfitting
#         x = Dropout(0.5)(x)
        
#         # Fully Connected Layer
#         x = Dense(4096, activation='relu')(x)
#         # Add Dropout
#         x = Dropout(0.5)(x)
        
#         # Output Layer
#         output_layer = Dense(self.num_classes, activation='softmax')(x)
        
#         # Create the model
#         self.model = Model(inputs=input_layer, outputs=output_layer)