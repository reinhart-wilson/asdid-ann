# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:32:41 2024

@author: reinh
"""

from keras import models, layers
import tensorflow as tf
import os
import numpy as np
import random

class MyModel:

    def __init__(self, seed=None, use_gpu=False):
        
        if seed is not None:
            self.set_determinism(seed)
        
        # Cek dan gunakan GPU
        if use_gpu and tf.test.is_gpu_available():
            # Set TensorFlow to use GPU memory growth
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        else:
            print("No GPU set")
            
        
        
    def set_determinism(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)                     
        
        os.environ['TF_DETERMINISTIC_OPS'] = '1'   
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
            
    
    def MobileNet(self, input_shape, num_classes, alpha=1.0, kernel_regularizer=None):
        input_tensor = layers.Input(shape=input_shape)
        
        # Lapisan convolutional inisial
        """
        Mengubah input citra menjadi representasi fitur awal.
        
        Conv2D membuat lapisan konvolusi 2D, kemudian hasilnya 
        dinormalisasi oleh Batch Normalization, dan diaktifkan oleh ReLU.
        """
        x = layers.Conv2D(int(32 * alpha), kernel_size=(3, 3), strides=(2, 2), 
                          padding='same', use_bias=False, 
                          kernel_regularizer=kernel_regularizer)(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Depthwise Separable Convolution Blocks
        x = self.depthwise_separable_block(x, filters=64, alpha=alpha,
                                           strides=(1, 1), 
                                           kernel_regularizer=kernel_regularizer)
        x = self.depthwise_separable_block(x, filters=128, alpha=alpha, 
                                           strides=(2, 2), 
                                           kernel_regularizer=kernel_regularizer)
        x = self.depthwise_separable_block(x, filters=128, alpha=alpha, 
                                           strides=(1, 1), 
                                           kernel_regularizer=kernel_regularizer)
        x = self.depthwise_separable_block(x, filters=256, alpha=alpha, 
                                           strides=(2, 2), 
                                           kernel_regularizer=kernel_regularizer)
        x = self.depthwise_separable_block(x, filters=256, alpha=alpha, 
                                           strides=(1, 1), 
                                           kernel_regularizer=kernel_regularizer)
        x = self.depthwise_separable_block(x, filters=512, alpha=alpha, 
                                           strides=(2, 2), 
                                           kernel_regularizer=kernel_regularizer)        
        for _ in range(5):
            x = self.depthwise_separable_block(x, filters=512, alpha=alpha, 
                                               strides=(1, 1), 
                                               kernel_regularizer=kernel_regularizer)       
        x = self.depthwise_separable_block(x, filters=1024, alpha=alpha, 
                                           strides=(2, 2), 
                                           kernel_regularizer=kernel_regularizer)
        x = self.depthwise_separable_block(x, filters=1024, alpha=alpha, 
                                           strides=(1, 1), 
                                           kernel_regularizer=kernel_regularizer)
        
        
        # Global Average Pooling and Dense Layer
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=input_tensor, outputs=x, name='MobileNet')
        return model
    
    def depthwise_separable_block(self, input_tensor, filters=64, alpha=1.0, 
                                  strides=(1, 1), kernel_regularizer=None):
        """
        Blok konvolusi depthwise separable yang merupakan inti dari arsitektur 
        MobileNet. Setiap blok terdiri dari lapisan-lapisan depthwise separable 
        convolution yang diikuti oleh operasi Batch Normalization dan fungsi 
        aktivasi ReLU.

        Parameters
        ----------
        input_tensor : tensor 
            Tensor input untuk blok konvolusi depthwise separable. 
        filters : integer 
            Jumlah filter yang digunakan.
        alpha : float
            learning rate dalam range [0.0, 1.0].
        strides : tuple integer, opsional
            Besar stride horizontal dan vertikal. Nilai default adalah (1, 1).

        Returns
        -------
        x : tensor, tensor output yang telah melewati lapisan konvolusi 
            depthwise separable, Batch Normalization, dan ReLU activation.

        """
        
        channel_axis = 1 if 'channels_first' else -1
        depthwise_conv = layers.DepthwiseConv2D(kernel_size=(3, 3), 
                                                strides=strides, 
                                                padding='same', 
                                                use_bias=False,
                                                depthwise_regularizer=kernel_regularizer
                                                )(input_tensor)
        x = layers.BatchNormalization(axis=channel_axis)(depthwise_conv)
        x = layers.ReLU()(x)
        
        pointwise_conv = layers.Conv2D(int(filters * alpha), kernel_size=(1, 1), 
                                       strides=(1, 1), padding='same', 
                                       use_bias=False, 
                                       kernel_regularizer=kernel_regularizer
                                       )(x)
        x = layers.BatchNormalization(axis=channel_axis)(pointwise_conv)
        x = layers.ReLU()(x)
        
        return x

    def LeNet(self, input_shape, num_classes):
        model = models.Sequential()

        # Lapisan ke-1: Convolutional + MaxPooling
        model.add(layers.Conv2D(6, kernel_size=(5, 5), 
                                activation='relu', 
                                input_shape=input_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        # Lapisan ke-2: Convolutional + MaxPooling
        model.add(layers.Conv2D(16, kernel_size=(5, 5), 
                                activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())

        # Lapisan ke-3: Fully Connected 
        model.add(layers.Dense(120, activation='relu'))

        # Lapisan ke-4: Fully Connected
        model.add(layers.Dense(84, activation='relu'))

        # Lapisan output
        model.add(layers.Dense(num_classes, activation='softmax'))

        return model     
    
def set_determinism(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)                     
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'   
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
set_determinism(2)