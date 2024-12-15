# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:17:54 2024

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


from tensorflow import lite as tflite
from configs.mobilenetv2_cfg import config_imagenet1_augment2_7_2 as config
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from mymodels.dummycnn import DummyCNN
import tensorflow as tf
# Informasi model yang sudah dilatih
saved_model_path = os.path.join(
    'C:/Users/reinh/Documents/GitHub/asdid-ann/training_result/mobilenetv2/augment/True+adddata+1e-4+imagenet+epoch100', 
    "best_model_epoch.tf")

# Ubah semua operasi layer ke float32. Wajib dilakukan jika mixed precision digunakan saat training.
model = load_model(saved_model_path)
for layer in model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
    if hasattr(layer, "dtype"):
        layer._dtype = tf.float32
save_path = os.path.join(
    'C:/Users/reinh/Documents/GitHub/asdid-ann/training_result/mobilenetv2/augment/True+adddata+1e-4+imagenet+epoch100', 
    "best_model_epoch_frozen.tf")
model.save(os.path.join(
    'C:/Users/reinh/Documents/GitHub/asdid-ann/training_result/mobilenetv2/augment/True+adddata+1e-4+imagenet+epoch100', 
    "best_model_epoch_frozen.tf"))

# Ubah model ke model TFLite
converter = tflite.TFLiteConverter.from_saved_model(save_path)
converter.target_spec.supported_ops = [
  tflite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tflite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
tflite_model = converter.convert()

# Simpan model tflite
tflite_filename = os.path.join(
    'C:/Users/reinh/Documents/GitHub/asdid-ann/training_result/mobilenetv2/augment/True+adddata+1e-4+imagenet+epoch100',  
    # f"model_at_epoch_{at_epoch}.tflite")
    "SoybeanMobileNet.tflite")
with open(tflite_filename,'wb') as f:
    f.write(tflite_model)

# Metadata informasi model
model_metadata = _metadata_fb.ModelMetadataT()
model_metadata.name = "MobileNetV2 Soybean Leaf Disease Classifier"
model_metadata.description = ("Model ini dapat mengidentifikasi 8 penyakit yang"
                              "terdapat pada foto daun kedelai.")
model_metadata.version = "2.0"
model_metadata.author = "reinh"

# Metadata informasi input
input_metadata = _metadata_fb.TensorMetadataT()
input_metadata.name = "image"
input_metadata.description = (
    "Gambar masukan yang akan diklasifikasi. Gambar harus berukuran 224 * 224"
    "dan memiliki kanal RGB per piksel. Nilai tiap kanal harus dinormalisasi "
    "dalam rentang [0.0, 1.0] sebelum digunakan sebagai input model.")
input_metadata.content = _metadata_fb.ContentT()
input_metadata.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_metadata.content.contentProperties.colorSpace = (
    _metadata_fb.ColorSpaceType.RGB)
input_metadata.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.ImageProperties)
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (
    _metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [0.0] # (Nilai masukan - mean) / std
input_normalization.options.std = [255.0]
input_metadata.processUnits = [input_normalization]
input_stats = _metadata_fb.StatsT()
input_stats.max = [1.0]
input_stats.min = [0.0]
input_metadata.stats = input_stats

# Metadata informasi output
output_metadata = _metadata_fb.TensorMetadataT()
output_metadata.name = "probability"
output_metadata.description = "Tingkat keyakinan dari 8 label penyakit."
output_metadata.content = _metadata_fb.ContentT()
output_metadata.content.content_properties = _metadata_fb.FeaturePropertiesT()
output_metadata.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
output_stats = _metadata_fb.StatsT()
output_stats.max = [1.0]
output_stats.min = [0.0]
output_metadata.stats = output_stats
label_file = _metadata_fb.AssociatedFileT()
label_file.name = os.path.basename("label.txt")
label_file.description = "File berisi 8 penyakit yang dapat dikenali model."
label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
output_metadata.associatedFiles = [label_file]

# combines the model information with the input and output information:
subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_metadata]
subgraph.outputTensorMetadata = [output_metadata]
model_metadata.subgraphMetadata = [subgraph]

b = flatbuffers.Builder(0)
b.Finish(
    model_metadata.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()

# Menambahkan metadata ke model tflite
model_file = (tflite_filename)
populator = _metadata.MetadataPopulator.with_model_file(model_file)
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files(["label.txt"])
populator.populate()