# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:12:44 2024

@author: reinh
"""

import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)

train_dir = os.path.join(working_dir, '..', 'train')
sys.path.append(train_dir)


from configs import data_location as dataloc
from configs.mobilenetv2_cfg import config_imagenet1_augment2_7_4 as config

from utils import general_utils as gutils
# gutils.use_gpu(False)

import numpy as np
from keras.models import load_model
from utils import evaluation_utils as eutils
from sklearn.metrics import classification_report
from PIL import Image


def main():  
    at_epoch = 200
    config_num = config.CFG_NUM
    result_folder = f'config{config_num}'
    result_path = os.path.join(train_dir, 'training_result', 
                                config.MODEL_CONFIG['model_name'])
    result_path = os.path.join(result_path, result_folder)
    model_path = os.path.join(result_path, f"model_at_epoch{at_epoch}")
    history_path = os.path.join(result_path, f"history_at_epoch{at_epoch}.pkl")
    
    #
    # history = eutils.load_history(history_path)
    
    # eutils.plot_loss(history)
    # eutils.plot_lr(history)
    # 
    test_data_dir = os.path.join(dataloc.DATA_PATH, 'test')
    test_data_dir = os.path.join(dataloc.DATA_PATH, 'validation')
    test_data_dir = os.path.join(dataloc.ADDITIONAL_DATA__2_PATH, 'test')
    test_generator = gutils.make_datagen(test_data_dir, config.IMAGE_SIZE, 
                                        config.BATCH_SIZE, shuffle=False)
    
    # 
    model = load_model(model_path)
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
    
    # # 
    class_labels = list(test_generator.class_indices.keys())
    print(class_labels)
    predictions = model.predict(test_generator)
    true_classes = test_generator.classes
    
    eutils.plot_confusion_matrix(predictions, true_classes, class_labels, 
                                  rotation=90)
    
    predicted_classes = np.argmax(predictions, axis=1)
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # # predict single image
    # image_filename = 'add_bacterial_blight_177.jpeg'
    # image_path = os.path.join(dataloc.BASE_PATH, 'bb', image_filename)
    # img = keras.utils.load_img('drive/MyDrive/abcdef.jpg', target_size=image_size)
    # img_array = keras.utils.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    # img_predictions = model.predict(image)
    # predicted_class = np.argmax(img_predictions, axis=1)
    # print(class_labels[predicted_class[0]])


if __name__ == "__main__":
    main()
    
    