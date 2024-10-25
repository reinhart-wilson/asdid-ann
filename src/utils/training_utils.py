import pickle
import os
import numpy as np
from time import time
from callbacks.callbacks_factory import create_callback
from sklearn.utils import class_weight as cw
from collections import Counter

def load_history(file_path):
    with open(file_path,'rb') as file:
        history = pickle.load(file)
    return history
    
def compute_decay_steps(num_samples, batch_size, epochs):    
    steps_per_epoch = num_samples // batch_size # Calculate the number of steps per epoch or basically batch per eppch
    decay_steps = steps_per_epoch * epochs # Total number of steps (across all epochs)
    return decay_steps 

def compute_decay_steps_ga(num_samples, batch_size, n_gradients, epochs):
    total_batch = num_samples/batch_size
    
    effective_update_steps = total_batch / n_gradients
    
    # Total number of steps (across all epochs)
    decay_steps = effective_update_steps * epochs
    
    return decay_steps 

def generate_callbacks(callback_configs):
    callbacks = []
    for cb_name, cfg_dict in callback_configs.items():
        callback = create_callback(cb_name, cfg_dict)
        callbacks.append(callback)
    if len(callbacks) == 0:
        return None
    return callbacks

def compute_class_weight(train_generator):
    # Total number of samples in each class
    class_counts = train_generator.classes  # This gives an array of class indices for each sample
       
    class_weights = cw.compute_class_weight(
        class_weight='balanced',  # This mode automatically adjusts the weights inversely proportional to class frequency
        classes=np.unique(class_counts),  # List of class labels
        y=class_counts  # The class indices from the generator
    )
    
    # Convert to dictionary for Keras
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict
    
    
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
        
