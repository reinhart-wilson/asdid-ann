import pickle
import os
import numpy as np
from time import time
from callbacks.callbacks_factory import create_callback
from sklearn.utils import class_weight as cw
import tensorflow as tf
    
def compute_decay_steps(num_samples, batch_size, epochs):    
    """
    Menghitung jumlah langkah (steps) untuk decay learning rate berdasarkan 
    jumlah sampel, ukuran batch, dan jumlah epoch. Fungsi ini digunakan untuk 
    menghitung jumlah langkah total yang diperlukan untuk scheduler decay, yang 
    berguna dalam pengaturan learning rate yang dinamis.

    Parameters
    ----------
    num_samples : int
        Jumlah total sampel dalam dataset pelatihan.
    
    batch_size : int
        Ukuran batch untuk pelatihan.
    
    epochs : int
        Jumlah epoch yang akan dilalui selama pelatihan.

    Returns
    -------
    decay_steps : int
        Jumlah langkah yang diperlukan untuk decay learning rate selama pelatihan.
    """
    steps_per_epoch = num_samples // batch_size  # Banyak batch yang diproses pada 1 epoch
    decay_steps = steps_per_epoch * epochs  # Jumlah step total di seluruh epoch
    return decay_steps

def compute_decay_steps_ga(num_samples, batch_size, n_gradients, epochs):
    """
    Menghitung jumlah langkah decay untuk akumulasi gradien.Fungsi ini digunakan 
    untuk mengatur langkah decay dalam eksperimen di mana gradien dihitung secara 
    akumulasi (gradient accumulation) sebelum pembaruan dilakukan pada model, 
    yang berguna untuk menyimulasikan pelatihan dengan ukuran batch yang lebih 
    besar pada lingkungan dengan sumber daya yang terbatas.

    Parameters
    ----------
    num_samples : int
        Jumlah total sampel dalam dataset pelatihan.
    
    batch_size : int
        Ukuran batch yang digunakan untuk perhitungan gradien.

    n_gradients : int
        Jumlah batch yang digunakan untuk menghitung akumulasi gradien sebelum 
        pembaruan model.
    
    epochs : int
        Jumlah epoch yang akan dilalui selama pelatihan.

    Returns
    -------
    decay_steps : int
        Jumlah langkah decay yang sesuai dengan pengaturan gradien akumulasi.
    """
    total_batch = num_samples/batch_size    
    effective_update_steps = total_batch / n_gradients
    decay_steps = effective_update_steps * epochs    
    return decay_steps 

def compute_class_weight(train_generator):
    """    
    Menghitung bobot kelas berdasarkan distribusi kelas dalam dataset pelatihan.

    Fungsi ini digunakan untuk menangani ketidakseimbangan kelas dalam dataset pelatihan, 
    dengan memberikan bobot yang lebih besar pada kelas yang jarang muncul untuk menghindari bias
    terhadap kelas mayoritas selama pelatihan.

    Parameters
    ----------
    train_generator : keras.preprocessing.image.DirectoryIterator
        Generator data pelatihan yang menghasilkan batch gambar dan label kelas.

    Returns
    -------
    class_weight_dict : dict
        Dictionary yang berisi bobot untuk setiap kelas yang ada dalam dataset.

    Parameters
    ----------
    train_generator : TYPE
        DESCRIPTION.

    Returns
    -------
    class_weight_dict : TYPE
        DESCRIPTION.

    """
    class_counts = train_generator.classes  # Mendapatkan array of indeks kelas untuk tiap sampel
       
    class_weights = cw.compute_class_weight(
        class_weight='balanced',  # Mode ini secara otomatis menyesuaikan bobot secara terbalik proporsional dengan frekuensi kelas
        classes=np.unique(class_counts),  # List label kelas
        y=class_counts  # Indeks kelas dari generator data
    )
    
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict

def get_last_lr_from_tensorboard(events_path):
    """
    Mendapatkan learning rate terakhir dari log TensorBoard yang digunakan selama 
    pelatihan.

    Fungsi ini berguna ketika pelatihan dihentikan dan kemudian dilanjutkan. Fungsi 
    ini akan membaca file log TensorBoard dan mengambil nilai learning rate terakhir 
    yang digunakan selama pelatihan.

    Parameters
    ----------
    events_path : str
        Path file events TensorBoard yang berisi riwayat pelatihan dan metrik yang 
        dilaporkan.

    Returns
    -------
    lr_tensor : numpy.ndarray
        Nilai learning rate terakhir yang digunakan selama pelatihan, dikembalikan 
        dalam bentuk array numpy.

    """
    sums = tf.compat.v1.train.summary_iterator(events_path)
    for e in sums:
        for value in e.summary.value:
            if value.tag == 'epoch_learning_rate':  # or the tag name used for your LR
                lr_tensor = tf.make_ndarray(value.tensor)
    return lr_tensor

def generate_callbacks(callback_configs):
    callbacks = []
    for cb_name, cfg_dict in callback_configs.items():
        callback = create_callback(cb_name, cfg_dict)
        callbacks.append(callback)
    if len(callbacks) == 0:
        return None
    return callbacks


def load_history(file_path):
    with open(file_path,'rb') as file:
        history = pickle.load(file)
    return history    
    
def print_parameters(use_gpu, batch_size, image_size, epochs,  
                     seed, alpha, loss_function, learning_rate,
                     ):
    print(f"{'Parameter':<15} {'Value'}")
    print("-" * 30)
    print(f"{'use_gpu':<15} {use_gpu}")
    print(f"{'batch_size':<15} {batch_size}")
    print(f"{'image_size':<15} {image_size}")
    print(f"{'epochs':<15} {epochs}")
    print(f"{'seed':<15} {seed}")
    print(f"{'alpha':<15} {alpha}")
    print(f"{'loss_function':<15} {loss_function}")
    print(f"{'learning_rate':<15} {learning_rate}")
        
def train_model(model, train_datagen, val_datagen, total_epochs, 
                training_result_path, last_epoch=0, class_weights=None,
                last_history=None, callbacks=None):
    os.makedirs(training_result_path, exist_ok=True)
    
    start_time = time()
        
    history = model.fit(train_datagen, 
                        validation_data=val_datagen, 
                        epochs=total_epochs, 
                        initial_epoch=last_epoch,
                        callbacks=callbacks,
                        class_weight=class_weights)\
            
    end_time = time() # Waktu akhir pelatihan
    execution_time_ms = (end_time - start_time) * 1000 # Waktu eksekusi (ms)
    print("\nExec time: ", execution_time_ms,'\n')
    
    return history
        


