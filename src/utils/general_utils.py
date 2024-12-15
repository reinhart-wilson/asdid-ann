import numpy as np
import os
import random
import sys
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from callbacks.callbacks_factory import create_callback

def set_determinism(seed):
    """
    Mengatur determinisme pada eksperimen dengan mengatur seed untuk berbagai 
    library yang digunakan dalam pelatihan model, termasuk NumPy, TensorFlow, 
    dan Python standard library.

    Fungsi ini memastikan bahwa eksperimen yang dilakukan akan menghasilkan hasil 
    yang sama setiap kali dijalankan, asalkan tidak ada faktor eksternal yang 
    mempengaruhi (seperti perangkat keras yang berbeda).

    Parameters
    ----------
    seed : int
        Nilai seed yang digunakan untuk memastikan determinisme dalam eksperimen.

    Returns
    -------
    None
    
    Notes
    -----
    - Fungsi ini akan mendegradasi performa pelatihan secara signifikan.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'   
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
def adjust_contrast_and_brightness(image):
    """
    Menambahkan augmentasi acak pada kontras dan kecerahan gambar.

    Fungsi ini melakukan perubahan acak pada kontras dan kecerahan gambar 
    untuk meningkatkan keberagaman data pelatihan dengan melakukan sedikit 
    perubahan pada gambar input.

    Parameters
    ----------
    image : tf.Tensor
        Gambar input dalam bentuk tensor yang akan diproses.

    Returns
    -------
    tf.Tensor
        Gambar yang telah dimodifikasi dengan perubahan pada kontras dan kecerahan.
    """
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image
    
def make_datagen(directory, image_size, batch_size, seed=None, shuffle=True, 
                 augment=False):
    """
    Membuat generator data untuk melatih model menggunakan gambar dari direktori.
    
    Fungsi ini menggunakan `ImageDataGenerator` untuk menghasilkan batch data 
    gambar yang dapat digunakan dalam pelatihan model, dengan opsional augmentasi 
    gambar jika `augment=True`.
    
    Parameters
    ----------
    directory : str
        Direktori yang berisi subdirektori gambar untuk setiap kelas yang akan 
        digunakan dalam pelatihan.
    
    image_size : tuple of int
        Ukuran gambar yang diubah sebelum dimasukkan ke dalam model (misalnya, 
        `(224, 224)`).
    
    batch_size : int
        Jumlah gambar dalam setiap batch yang akan dihasilkan oleh generator.
    
    seed : int, optional
        Nilai seed untuk memastikan hasil yang dapat diulang. Default adalah None.
    
    shuffle : bool, optional
        Menentukan apakah gambar akan diacak sebelum dibagi menjadi batch. Default 
        adalah True.
    
    augment : bool, optional
        Jika True, akan diterapkan augmentasi pada gambar (perubahan kontras, 
        kecerahan, rotasi, dll). Default adalah False.
    
    Returns
    -------
    flow : DirectoryIterator
        Generator data yang menghasilkan batch gambar beserta label kelas yang sesuai.
        
    """
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

def use_gpu(use_gpu, limit = None):  
    """
    Mengonfigurasi penggunaan GPU dengan TensorFlow.

    Parameters
    ----------
    use_gpu : bool
        Jika True, TensorFlow akan diatur untuk menggunakan GPU jika tersedia. 
        Jika False, GPU tidak akan dipakai.
    
    limit : int, optional
        Batasan jumlah memori (dalam MB) yang akan digunakan oleh TensorFlow pada 
        GPU. Default adalah None, yang berarti tidak ada batasan.

    Returns
    -------
    None        

    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if use_gpu and physical_devices:
        if limit is None:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            for gpu in physical_devices:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        print ('GPU Set')
    else:
        tf.config.set_visible_devices([], 'GPU') # matikan penggunaan GPU
        print("No GPU set")
        
def use_mixed_precision():
    """
    Mengaktifkan penggunaan presisi campuran (mixed precision) pada pelatihan 
    model dengan TensorFlow. Fungsi ini mengatur kebijakan presisi global TensorFlow 
    menjadi `mixed_float16`, yang memungkinkan model untuk menggunakan presisi 
    16-bit (float16) pada beberapa operasi untuk mempercepat pelatihan dan mengurangi
    penggunaan memori GPU, tanpa mengorbankan akurasi model secara signifikan.
    """
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