# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:15:08 2024

@author: reinh
"""

import numpy as np
import pickle
import EvaluationUtils as eu
from os import path
from TrainingUtils import make_datagen
from keras.models import load_model


if __name__ == "__main__":
    use_gpu = True
    batch_size = 16
    image_size = (224, 224)  
    input_shape = tuple(list(image_size) + [3])
    num_classes = 8
    at_epoch = 30
    seed=42 
    
    # Load history
    result_path = './training_result/mobilenetv1-20240521-205117'
    history_path = path.join(result_path, f"history_at_epoch_{at_epoch}.pkl")
    with open(history_path,'rb') as file:
        history = pickle.load(file)
    eu.plot_loss(history)
    
    # Load model
    model_path = path.join(result_path, f"model_at_epoch_{at_epoch}.h5")
    model = load_model(model_path)
    
    
    # Load data
    output_path = "..\Dataset\split_prepped_data" 
    test_data_dir = path.join(output_path, 'test')    
    test_generator = make_datagen(test_data_dir, 
                                  image_size, 
                                  batch_size,
                                  seed=seed,
                                  shuffle=None) 

    
    # 
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
    
    # Buat confusion matrix
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1) # Mengambil kelas dengan probabilitas tertinggi
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    conf_matrix = eu.plot_confusion_matrix(predictions, true_classes, class_labels, rotation=90)
    
    
    for i in range(len(class_labels)):
        eu.plot_confusion_matrix_per_class(conf_matrix, class_labels, i, rotation=45)   