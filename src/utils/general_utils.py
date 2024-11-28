import numpy as np
import os
import random
import sys
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from callbacks.callbacks_factory import create_callback

def set_determinism(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'   
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
def adjust_contrast_and_brightness(image):
    # Adjust brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    # Adjust contrast
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image
    
def make_datagen(directory, image_size, batch_size, seed=None, shuffle=True, 
                 augment=False):
        
    if augment:
        preprocessing_function=adjust_contrast_and_brightness               
        datagen = ImageDataGenerator(rescale=1./255,                                      
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True, 
            fill_mode='constant', 
            cval=0,
            preprocessing_function=preprocessing_function)
    else:
        datagen = ImageDataGenerator(rescale=1./255)
    

    flow = datagen.flow_from_directory(
        directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode="rgb",
        shuffle=shuffle,
        seed=seed
    )
    return flow

def use_gpu(use_gpu):    
    physical_devices = tf.config.list_physical_devices('GPU')
    if use_gpu and physical_devices:
        # Set TensorFlow to use GPU memory growth
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
        for gpu in physical_devices:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])  # Set limit to 2GB (or however much you want)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        print ('GPU Set')
    else:
        tf.config.set_visible_devices([], 'GPU') # matikan penggunaan GPU
        print("No GPU set")
        
def use_mixed_precision():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
def clean_memory (signum, frame):
    print("Cleaning up before exiting...")
    K.clear_session()
    sys.exit(0)

def combine_generators(gen1, gen2, batch_size):
    while True: # While true karena tidak diketahui banyak batch di kedua generator.
        # Mengambil batch dari setiap generator
        batch1 = next(gen1)
        batch2 = next(gen2)
        
        # Gabungkan batch dari dua generator
        combined_batch = (np.concatenate((batch1[0], batch2[0]), axis=0),
                          np.concatenate((batch1[1], batch2[1]), axis=0))
        
        yield combined_batch
        
def generate_callbacks(callback_configs):
    callbacks = []
    for cb_name, cfg_dict in callback_configs.items():
        callback = create_callback(cb_name, cfg_dict)
        callbacks.append(callback)
    if len(callbacks) == 0:
        return None
    return callbacks