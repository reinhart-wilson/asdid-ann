seed=42 
use_gpu = True

import pickle
import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(src_dir)

import tensorflow as tf
from keras import optimizers, models, layers, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from time import strftime
from utils.general_utils import make_datagen
from utils.training_utils import print_parameters, train_model
from callbacks import LearningRateLogger
from callbacks.schedules import WarmUpCosineDecay
    
    
    
if __name__ == "__main__":
    batch_size = 10
    image_size = (224, 224)  
    input_shape = tuple(list(image_size) + [3])
    epochs = 300
    num_classes = 8
    alpha=1
    learning_rate=0.00001
    loss_function='categorical_crossentropy'
    es_patience = 15
    es_min_delta = 0.01
    lr_patience = 5
    lr_min_delta =0.05
    lr_decay_factor = 0.5
    l2_lambda = 0.005
    
    
    
    print_parameters(use_gpu, batch_size, image_size, epochs, seed, alpha,
                      loss_function, learning_rate)
    print()
    
    #setting lain    
    cur_time = strftime("%Y%m%d-%H%M%S")
    training_result_path = f'./training_result/mobilenetv1-{cur_time}'
    save_interval = 5
    
    # Percobaan menggunakan cosine decay 
    init_lr = 0.001
    warmup_target = 0.01
    
    lr_schedule = WarmUpCosineDecay(
        initial_learning_rate=init_lr,
        decay_steps=epochs,
        warmup_steps=10,
        warmup_target=warmup_target,
        alpha=warmup_target/init_lr
    )
    optimizer=optimizers.Adam(learning_rate=learning_rate)
        
    # Callbacks    
    early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, 
                                   min_delta=es_min_delta)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=lr_decay_factor, 
                                  patience=lr_patience, 
                                  min_delta=lr_min_delta)
    save_on_intervals = ModelCheckpoint(os.path.join(training_result_path, "{epoch:02d}.keras"), 
                                        save_weights_only=False,
                                        save_freq='epoch',
                                        period = save_interval) 
    lr_logger = LearningRateLogger()
    callbacks = [
        early_stopping,
        reduce_lr
        ]
    callbacks=[save_on_intervals, lr_logger] # Comment bagian ini jika callbacks ingin dimatikan 
    
    # regularizer
    l2_reg = regularizers.l2(l2_lambda)
    kernel_regularizer=l2_reg
    kernel_regularizer = None
    
    # Path data
    augmented_data_dir = "..\..\Dataset\split_augmented_data" 
    augmented_train_data_dir = os.path.join(augmented_data_dir, 'train')
    augmented_val_data_dir = os.path.join(augmented_data_dir, 'validation')
    
    # Buat generator data yang akan diumpankan ke model    
    print('Loading augmented dataset')
    print("-" * 30)
    print('Training dataset:')
    aug_train_datagen = make_datagen(augmented_train_data_dir, image_size, batch_size, 
                                seed=seed)    
    print('Validation dataset:')
    aug_val_datagen = make_datagen(augmented_val_data_dir, image_size, batch_size, 
                                seed=seed)
    print()
    
    # # Buat model MobileNet
    # Load pre-trained MobileNetV2 model without top layers
    base_model = tf.keras.applications.MobileNet(include_top=False, 
                                                   weights=None,
                                                   input_shape=input_shape, 
                                                   alpha=alpha)
    
    # Add custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)  # Adjust the number of units based on experimentation
    x = layers.Dropout(0.5)(x)  # Add dropout for regularization
    predictions = layers.Dense(num_classes, activation='softmax')(x)  # Final classification layer

    # Create the model
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    print('Model training')
    print("-" * 30)
    # model = my_model.MobileNet(input_shape, num_classes, alpha=alpha, 
    #                             kernel_regularizer=kernel_regularizer)
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy'])
    
    # Pelatihan model
    history = train_model(model, 
                          aug_train_datagen, 
                          aug_val_datagen, 
                          epochs, 
                          training_result_path, 
                          callbacks=callbacks)
    print()
    
   # # Lanjutkan pelatihan jika terinterupsi
    # print('Model training (continued)')
    # print("-" * 30)
    # last_epoch = 270
    # continued_result_path = './training_result/mobilenetv1-20240531-172228'
    # interrupted_model_path = os.path.join(continued_result_path, 
    #                                   f"{last_epoch}.keras")
    # interrupted_model = models.load_model(interrupted_model_path)
    # # interrupted_history_path = os.path.join(continued_result_path, 
    # #                                   f"history_at_epoch_{last_epoch}.pkl")
    # # with open(interrupted_history_path,'rb') as file:
    # #     interrupted_history = pickle.load(file)
        
    # history = train_model(interrupted_model, 
    #                       optimizer, 
    #                       aug_train_datagen, 
    #                       aug_val_datagen, 
    #                       epochs, 
    #                       training_result_path, 
    #                       interval=save_interval,
    #                       callbacks=callbacks,
    #                       last_epoch=last_epoch)