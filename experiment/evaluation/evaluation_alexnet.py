import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)

train_dir = os.path.join(working_dir, '..', 'train')
sys.path.append(train_dir)


from configs import data_location as dataloc
from configs.alexnet_cfg import config3 as config

from utils import general_utils as gutils
gutils.use_gpu(False)

import numpy as np
from keras.models import load_model
from utils import evaluation_utils as eutils


def main():  
    at_epoch = 50       
    result_folder = 'config3_1717601560.6066334'
    result_path = os.path.join(train_dir, 'training_result', 
                               config.MODEL_CONFIG['model_name'])
    result_path = os.path.join(result_path, result_folder)
    
    # 
    test_data_dir = os.path.join(dataloc.DATA_PATH, 'test')
    test_generator = gutils.make_datagen(test_data_dir, config.IMAGE_SIZE, 
                                       config.BATCH_SIZE)
    
    # 
    model_path = os.path.join(result_path, f"model_at_epoch{at_epoch}.keras")
    model = load_model(model_path)
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
    
    # 
    class_labels = list(test_generator.class_indices.keys())
    eutils.plot_confusion_matrix(model, test_generator, class_labels, text_rotation=90)
    
    #
    # history_path = os.path.join(result_path, f"history_at_epoch{at_epoch}.keras")
    # history = eutils.load_history(history_path)
    

    
if __name__ == "__main__":
    main()
    
    