# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:35:34 2024

@author: reinh
"""
import sys
import os
working_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(working_dir, '..', 'src')
sys.path.append(src_dir)


from utils import evaluation_utils as eutils
from mymodels.dummycnn import DummyCNN
import numpy as np
import tensorflow as tf

# lokasi model
arc_name = 'lenet'
training_result_folder = os.path.join(
    '../training_result/exp1', 
    arc_name)
attempt_no = str(1)

# muat model
input_shape = (224, 224, 3)
cnn_model = DummyCNN(input_shape, 8)
cnn_model.load_model( 
    os.path.join(
        training_result_folder,
        attempt_no,
        'best_model.tf'
        )
    )

# hitung flops
dummy_input = tf.constant(np.random.randn(1,224,224,3))
returned_model = cnn_model.get_tf_model()
print('Flops = ', eutils.get_flops(cnn_model.get_tf_model(), [dummy_input]))

# Hitung waktu inferensi
img_folder = '../dataset/split_prepped_data/test/bacterial_blight'
