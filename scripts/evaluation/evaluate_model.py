# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 00:08:34 2024

@author: reinh
"""

import sys, os

working_dir = os.path.abspath(os.path.dirname(__file__))
configs_dir = os.path.join(working_dir, '..', '..', 'config')
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)
sys.path.append(configs_dir)

from utils import general_utils as gutils, evaluation_utils as eutils
gutils.use_mixed_precision()

import numpy as np
from mymodels.mobilenetv2 import MyMobileNetV2
from sklearn.metrics import classification_report

# Load model
highest_acc_epoch = 0
highest_acc = 0

model_path = '../../training_result/mobilenetv2/final/model_at_epoch160'
model = MyMobileNetV2((0), 8)
model.load_model(model_path)

# Load data
data_dir = '../../Dataset/split_prepped_additional_data_2_2/test'
test_data = gutils.make_datagen(data_dir, (224,224), 8, shuffle=False, augment=False)

# Evaluasi
model.evaluate(test_data)

# Lihat Heatmap Confusion Matrix
class_labels = list(test_data.class_indices.keys())
true_classes = test_data.classes
predictions = model.predict(test_data)
eutils.plot_confusion_matrix(predictions, 
                             true_classes, 
                             class_labels, 
                             rotation=90, 
                             title='Hasil Klasifikasi Model pada Set Data Pengujian Tambahan')

predicted_classes = np.argmax(predictions, axis=1)
report = classification_report(true_classes, 
                               predicted_classes, 
                               target_names=class_labels,
                               output_dict=False)
print(report)

# for i in range (150,201):
#     model_path = f'../../training_result/mobilenetv2/final/model_at_epoch{i}'
#     model = MyMobileNetV2((0), 8)
#     model.load_model(model_path)
    
#     # Load data
#     data_dir = '../../Dataset/split_prepped_additional_data_2_2/validation'
#     test_data = gutils.make_datagen(data_dir, (224,224), 8, shuffle=False, augment=False)
    
#     # Evaluasi
#     model.evaluate(test_data)
    
#     # Lihat Heatmap Confusion Matrix
#     class_labels = list(test_data.class_indices.keys())
#     true_classes = test_data.classes
#     predictions = model.predict(test_data)
#     eutils.plot_confusion_matrix(predictions, 
#                                  true_classes, 
#                                  class_labels, 
#                                  rotation=90, 
#                                  title='Hasil Klasifikasi Model pada Set Data Validasi Tambahan')
    
#     predicted_classes = np.argmax(predictions, axis=1)
#     report = classification_report(true_classes, 
#                                    predicted_classes, 
#                                    target_names=class_labels,
#                                    output_dict=True)
#     if report['macro avg']['recall'] > highest_acc:
#         highest_acc = report['macro avg']['recall'] 
#         highest_acc_epoch = i
        
# print(i)
