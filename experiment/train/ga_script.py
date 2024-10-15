import sys
import os
from absl import logging

#Disable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)

# Import file konfigurasi
from configs.mobilenetv2_cfg import config_imagenet1_augment2_10 as config
from configs import data_location as dataloc

# Set seed untuk beberapa library python agar hasil deterministik
from utils import general_utils as gutils
from utils import training_utils as tutils
gutils.use_gpu(config.USE_GPU)
# gutils.use_mixed_precision()
# gutils.set_determinism(config.SEED)

# Import packages lainnya yang diperlukan
import signal
import pickle
from mymodels.model_factory import create_model
from keras.models import load_model

#temp
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

# ===============================

def load_data(augment=False):
    train_data_dir = os.path.join(config.DATA_PATH, 'train')
    # train_data_dir = dataloc.ADDITIONAL_DATA_PATH
    val_data_dir = os.path.join(config.DATA_PATH, 'validation')
    # val_data_dir=dataloc.ADDITIONAL_DATA_PATH
    train_datagen = gutils.make_datagen(train_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE,
                                        augment=augment)
    val_datagen = gutils.make_datagen(val_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE)
    return train_datagen, val_datagen

# def train():    
    # Buat callbacks
model_callbacks = tutils.generate_callbacks(config.CALLBACKS_CONFIG)
tensorboard_callback = callbacks.TensorBoard(log_dir=config.LOGDIR)
model_checkpoint = callbacks.ModelCheckpoint(
    os.path.join(config.RESULT_PATH, config.MODEL_FILENAME),
    verbose=0
    )
model_callbacks.append(tensorboard_callback)
model_callbacks.append(model_checkpoint)

# Muat data
train_datagen, val_datagen = load_data(config.AUGMENT)

# Gunakan class weights
class_weights = tutils.compute_class_weight(train_datagen)

# temp
LR_SCHEDULE = optimizers.schedules.CosineDecay(
    initial_learning_rate=config.LEARNING_RATE,
    alpha = config.LR_ALPHA,
    decay_steps=tutils.compute_decay_steps(train_datagen.samples, config.BATCH_SIZE, 
                                    config.EPOCHS)
)
OPTIMIZER = optimizers.SGD(learning_rate=LR_SCHEDULE, momentum=0.9)
# OPTIMIZER = optimizers.Adam(learning_rate=config.LEARNING_RATE)

# Training
model = create_model(config.INPUT_SHAPE, config.NUM_CLASSES, config.MODEL_CONFIG)
model.build_model()
model.compile_model(optimizer=OPTIMIZER)
history = model.train(train_datagen, val_datagen, epochs = config.EPOCHS, 
            batch_size=config.BATCH_SIZE,callbacks=model_callbacks, 
            class_weights=class_weights)
model_path= os.path.join(config.RESULT_PATH, 'model.tf')
model.save_model(model_path)

# Release memory
signal.signal(signal.SIGINT, gutils.clean_memory)
    
def resume_train(at_epoch):
    #
    callbacks_config = config.CALLBACKS_CONFIG
    callbacks_config['history_saver']['at_epoch']=at_epoch
    callbacks = tutils.generate_callbacks(config.CALLBACKS_CONFIG)
    
    # Muat data
    train_datagen, val_datagen = load_data()
    
    model_path = os.path.join(config.RESULT_PATH, f"model_at_epoch{at_epoch}.keras")
    model = load_model(model_path)    
    interrupted_history_path = os.path.join(config.RESULT_PATH, 
                                      f"history_at_epoch{at_epoch}.pkl")
    with open(interrupted_history_path,'rb') as file:
        interrupted_history = pickle.load(file)
        
    tutils.train_model(model, train_datagen, val_datagen, config.EPOCHS, 
                       config.RESULT_PATH, last_epoch=at_epoch, 
                       last_history=interrupted_history,callbacks=callbacks)
    

    
# if __name__ == "__main__":
#     train()  
#     # resume_train(60)