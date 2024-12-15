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
from utils import evaluation_utils as eutils
from PIL import Image

#config
MODE = 0 #0 = all
DEVICE = '/CPU:0'
folders = ['alexnet']

exp1_path = '../../training_result/exp1'
exp_data_path = '../../other_experiment_data/inference_time'
IMAGE_FOLDER = '../../dataset/inference_time_data'
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
folders = ['nasnetmobile/1']
folders = folders if MODE == 1 else os.listdir(exp1_path)
for folder in folders:
    # load model, konversi ke tflite
    model_path = os.path.join(
        exp1_path, folder, 'model.tf'
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
                data = eutils.load_image(file_path)
                
                        
                # Run inference
                infer_time = eutils.get_inference_time(interpreter, data)
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
    
    