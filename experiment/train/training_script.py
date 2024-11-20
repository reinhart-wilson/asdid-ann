import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)
# Set seed untuk beberapa library python agar hasil deterministik
from utils import general_utils as gutils
# gutils.set_determinism(42)
gutils.use_mixed_precision()
gutils.use_gpu(True)

from utils import training_utils as tutils
# Import file konfigurasi
# from configs.exp1 import efficientnet_cfg as config
from configs import data_location as dataloc





# Import packages lainnya yang diperlukan
import signal
import pickle
from mymodels.model_factory import create_model
from keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
#temp
from tensorflow.keras import optimizers
from tensorflow.keras import metrics

# ===============================

def load_data(config, augment=False):
    train_data_dir = os.path.join(config.DATA_PATH, 'train')
    # train_data_dir = dataloc.ADDITIONAL_DATA_PATH
    val_data_dir = os.path.join(config.DATA_PATH, 'validation')
    # val_data_dir=dataloc.ADDITIONAL_DATA_PATH
    train_datagen = gutils.make_datagen(train_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE,
                                        augment=augment)
    val_datagen = gutils.make_datagen(val_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE)
    return train_datagen, val_datagen

def train(config):    
    #
    callbacks = tutils.generate_callbacks(config.CALLBACKS_CONFIG)
    
    
    # Muat data
    train_datagen, val_datagen = load_data(config, config.AUGMENT)
    
    # Gunakan class weights
    class_weights = tutils.compute_class_weight(train_datagen)
    
    # temp
    # LR_SCHEDULE = optimizers.schedules.CosineDecay(
    #     initial_learning_rate=config.LEARNING_RATE,
    #     alpha = config.LR_ALPHA,
    #     decay_steps=tutils.compute_decay_steps(train_datagen.samples, config.BATCH_SIZE, 
    #                                     config.EPOCHS)
    # )
    # OPTIMIZER = optimizers.SGD(learning_rate=LR_SCHEDULE, momentum=0.9)
    OPTIMIZER = optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    # Training
    model = create_model(config.INPUT_SHAPE, config.NUM_CLASSES, config.MODEL_CONFIG)
    model.build_model()
    model.compile_model(optimizer=OPTIMIZER,
                        metrics=['accuracy', metrics.Recall()])
    model.train(train_datagen, val_datagen, epochs = config.EPOCHS, 
                batch_size=config.BATCH_SIZE,callbacks=callbacks,
                # class_weights=class_weights
                )
    
    # Release memory
    signal.signal(signal.SIGINT, gutils.clean_memory)
    
# def resume_train(at_epoch):
#     #
#     callbacks_config = config.CALLBACKS_CONFIG
#     callbacks_config['history_saver']['at_epoch']=at_epoch
#     callbacks = tutils.generate_callbacks(config.CALLBACKS_CONFIG)
    
#     # Muat data
#     train_datagen, val_datagen = load_data()
    
#     model_path = os.path.join(config.RESULT_PATH, f"model_at_epoch{at_epoch}.keras")
#     model = load_model(model_path)    
#     interrupted_history_path = os.path.join(config.RESULT_PATH, 
#                                       f"history_at_epoch{at_epoch}.pkl")
#     with open(interrupted_history_path,'rb') as file:
#         interrupted_history = pickle.load(file)
        
#     tutils.train_model(model, train_datagen, val_datagen, config.EPOCHS, 
#                        config.RESULT_PATH, last_epoch=at_epoch, 
#                        last_history=interrupted_history,callbacks=callbacks)
    

    
if __name__ == "__main__":
    from configs.exp1 import mobilenetv2_cfg as config
    train(config)
    # resume_train(30)