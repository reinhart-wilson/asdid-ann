# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:44:37 2024

@author: reinh
"""

import os
import sys

working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '../..', 'src')
configs_dir = os.path.join(working_dir, '../..', 'config')
sys.path.append(src_dir)
sys.path.append(configs_dir)

import numpy as np
import sys
import tensorflow as tf
import time
from configs.other_configs import data_info as dinfo
from mymodels.dummycnn import DummyCNN
from utils import evaluation_utils as eutils
from utils import general_utils as gutils
import tensorflow as tf
import numpy as np
from PIL import Image

def load_image(image_path, target_size=(224, 224)):
    """
    Loads and preprocesses the image.
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for the image (default is 224x224).
    Returns:
        np.array: Preprocessed image array.
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array

def load_tflite_model(model_path):
    """
    Loads the TFLite model.
    Args:
        model_path (str): Path to the TFLite model file.
    Returns:
        tf.lite.Interpreter: Loaded TFLite model interpreter.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, input_data):
    """
    Runs inference on the input data using the TFLite model.
    Args:
        interpreter (tf.lite.Interpreter): Loaded TFLite model interpreter.
        input_data (np.array): Preprocessed input image data.
    Returns:
        np.array: Output predictions from the model.
    """
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    # Set input tensor
    interpreter.set_tensor(input_index, input_data)

    # Run inference
    interpreter.invoke()

    # Retrieve output tensor
    predictions = interpreter.get_tensor(output_index)
    return predictions

def get_inference_time(interpreter, input_data):
    """
    Runs inference on the input data using the TFLite model and measures the time taken.
    Args:
        interpreter (tf.lite.Interpreter): Loaded TFLite model interpreter.
        input_data (np.array): Preprocessed input image data.
    Returns:
        float: Time taken for inference in seconds.
    """
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    # Set input tensor
    interpreter.set_tensor(input_index, input_data)

    # Measure inference time
    start_time = time.time()  # Start time right before invoking inference
    interpreter.invoke()      # Run inference
    end_time = time.time()    # End time right after invoking inference
    
    # Calculate time taken for inference
    return end_time - start_time

def get_top_prediction(predictions, class_names):
    """
    Retrieves the top predicted class and confidence score.
    Args:
        predictions (np.array): Model prediction probabilities.
        class_names (list): List of class names.
    Returns:
        str: Predicted class name and confidence score.
    """
    top_index = np.argmax(predictions)
    top_class = class_names[top_index]
    confidence = predictions[0][top_index]
    return top_class, confidence



#config
MODE = 1 #0 = all
DEVICE = '/CPU:0'
folders = ['alexnet']

exp1_path = '../../training_result/exp1'
exp_data_path = '../../other_experiment_data/inference_times'
IMAGE_FOLDER = '../../dataset/inference_time_data'
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

folders = folders if MODE == 1 else os.listdir(exp1_path)
for folder in folders:
    # load model, konversi ke tflite
    model_path = os.path.join(
        exp1_path, folder, 'last_model.tf'
    )
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.experimental_new_converter = True  # Optional, to use the new converter
    tflite_model = converter.convert()
    
    # Load model ke interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Infer semua gambar
    infer_times=[]
    for root, dirs, files in os.walk(IMAGE_FOLDER):
        for filename in files:
            if filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
                file_path = os.path.join(root, filename)
                data = load_image(file_path)
                        
                # Run inference
                infer_time = get_inference_time(interpreter, data)
                infer_times.append(infer_time*1000)
                
                # Simpan data
                infer_times_dir = os.path.join(exp_data_path, folder, )
                infer_times_file = os.path.join(exp_data_path, folder, 'infer_times.txt')
                os.makedirs(infer_times_dir, exist_ok=True)
                with open(infer_times_file, 'w') as f:
                    for infer_time in infer_times:
                        f.write(f"{infer_time}\n")
             
    # Ukuran model
    model_size_memory = sys.getsizeof(tflite_model)
    model_size_mb_memory = model_size_memory / (1024 * 1024)
    
    avg = np.mean(infer_times)
    std = np.std(infer_times, ddof=1)  # Use ddof=1 for sample standard deviation
    print(f"Results for {folder}:")
    print(f"mean: \({round(avg, 3)} \pm {round(std,3)}\)")
    print(f'median: \({round(np.median(infer_times),3)}\)')
    print(f'size(mb): \({round(model_size_mb_memory, 3)}\)','\n') 
    
    