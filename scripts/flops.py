# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 21:34:55 2024

@author: reinh
"""

import os
import sys

working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', 'src')
configs_dir = os.path.join(working_dir, '..', 'config')
sys.path.append(src_dir)
sys.path.append(configs_dir)

import numpy as np
import tensorflow as tf
from configs.other_configs import data_info as dinfo
from mymodels.dummycnn import DummyCNN
from utils import evaluation_utils as eutils
from utils import general_utils as gutils

#config
MODE = 0 #0 = all
DEVICE = '/CPU:0'

exp1_path = '../training_result/exp1'
folders = ['efficientnetv2b0', 'mobilenetv1', 'mobilenetv2', 'mobilenetv3large', 'mobilenetv3small']
folders = ['squeezenet/1', 'nasnetmobile/1']
data_path = '../dataset/split_prepped_additional_data_2/test'
data_path = '../dataset/prepped_data'
# data = gutils.make_datagen(data_path, dinfo.IMG_RES, 1, shuffle=False)

folders = folders if MODE == 1 else os.listdir(exp1_path)
for folder in folders:
    # load model
    model_dir = os.path.join(
        exp1_path, folder, 'last_model.tf'
    )
    dummy_model = DummyCNN(dinfo.IMG_DIM, dinfo.NUM_CLASSES)
    dummy_model.load_model(model_dir)
    
    print(f"Results for {folder}:")
    #flop
    model_inputs = tf.constant(np.random.randn(1,224,224,3))
    flop = eutils.get_flops(dummy_model.get_tf_model(), [model_inputs])
    print(f'{folder} FLOPs : ', flop/(10^6))
    print(f'{folder} Params : ', dummy_model.get_tf_model().count_params())
    # inference time

    # print('CPU')
    # with tf.device('/CPU:0'):
    #     # for i in range(50):
    #     #     dummy_model.predict(data)
    #     dummy_model.predict(data)
    print()
        
        
    
