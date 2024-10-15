import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)

# Import file konfigurasi
from configs import data_location as dataloc
from configs.alexnet_cfg import config5 as config

# Set seed untuk beberapa library python agar hasil deterministik
from utils import general_utils as gutils
gutils.use_gpu(config.USE_GPU)
gutils.set_determinism(config.SEED)

# Import packages lainnya yang diperlukan
import signal
from callbacks.callbacks_factory import create_callback
from keras import optimizers
from mymodels.model_factory import create_model


# ===============================

def generate_callbacks(callback_configs):
    callbacks = []
    for cb_name, cfg_dict in callback_configs.items():
        callback = create_callback(cb_name, cfg_dict)
        callbacks.append(callback)
    if len(callbacks) == 0:
        return None
    return callbacks

def main():    
    #
    callbacks = generate_callbacks(config.CALLBACKS_CONFIG)
    
    #
    optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE)

    # Muat data
    train_data_dir = os.path.join(config.DATA_PATH, 'train')
    val_data_dir = os.path.join(config.DATA_PATH, 'validation')
    train_datagen = gutils.make_datagen(train_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE)
    val_datagen = gutils.make_datagen(val_data_dir, config.IMAGE_SIZE, config.BATCH_SIZE)
    
    # Training
    model = create_model(config.INPUT_SHAPE, config.NUM_CLASSES, config.MODEL_CONFIG)
    model.build_model()
    model.compile_model(optimizer=optimizer)
    model.train(train_datagen, val_datagen, epochs = config.EPOCHS, 
                batch_size=config.BATCH_SIZE,callbacks=callbacks)
    
    # Release memory
    signal.signal(signal.SIGINT, gutils.clean_memory)
    
if __name__ == "__main__":
    main()
