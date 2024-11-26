# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:48:05 2024

@author: reinh
"""

import sys
import os

# Menambahkan path src ke Python path agar modul-modul dalam source code dapat dipanggil
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', 'src')
configs_dir = os.path.join(working_dir, '..', 'config')
sys.path.append(src_dir)
sys.path.append(configs_dir)

import numpy as np
from configs.other_configs import data_info as dinfo
from keras.models import load_model
from utils import evaluation_utils as eutils
from utils import general_utils as gutils
from PIL import Image
from sklearn.metrics import classification_report
 
# Paths
EXP_NUM                 = 3
SAVE_PATH               = f'../training_result/squeezenet/{EXP_NUM}'
BEST_MODEL_FILENAME     = 'best_model.tf'
LATEST_MODEL_FILENAME   = 'last_model.tf'
LOGDIR                  = os.path.join(SAVE_PATH, 'logs')
MODEL_PATH              = os.path.join(SAVE_PATH, 
                                       BEST_MODEL_FILENAME)

# Params
batch_size  = 32

test_data_dir = os.path.join(dinfo.DATA_PATH, 'test')
# test_data_dir = os.path.join(dinfo.DATA_PATH, 'validation')
test_data_dir = os.path.join(dinfo.ADDITIONAL_DATA__2_PATH, 'test')
test_generator = gutils.make_datagen(test_data_dir, 
                                     dinfo.IMG_RES, 
                                     batch_size, 
                                     shuffle=False)

# 
model = load_model(MODEL_PATH)
loss, accuracy, recall = model.evaluate(test_generator)
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