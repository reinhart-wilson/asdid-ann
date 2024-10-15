# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:57:58 2024

@author: reinh
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from .abstract_cnn import AbstractCNN

class VGG16 (AbstractCNN):    

    def build_model(self):
        model = Sequential()

        # Lapisan Konvolusi Blok 1
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
           
        # Lapisan Konvolusi Blok 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
           
        # Lapisan Konvolusi Blok 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
           
        # Lapisan Konvolusi Blok 4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
           
        # Lapisan Konvolusi Blok 5
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
           
        # Lapisan Dense
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        self.model=model