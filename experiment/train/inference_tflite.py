# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:44:37 2024

@author: reinh
"""

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

if __name__ == "__main__":
    # Path to the image and model
    test_data_folder = 'C:/Users/reinh/Documents/GitHub/asdid-ann/dataset/additional_data/'
    class_folder = 'healthy/'
    image_filename = 'DSC_01611.jpg'
    image_path = test_data_folder + class_folder + image_filename
    model_path = 'C:/Users/reinh/Documents/GitHub/asdid-ann/experiment/train/training_result/mobilenetv2/configimagenet1_augment2.7_2/model_at_epoch_150.tflite'  # Replace with your TFLite model path

    # Load and preprocess the image
    input_data = load_image(image_path)

    # Load the TFLite model
    interpreter = load_tflite_model(model_path)

    # Run inference
    predictions = run_inference(interpreter, input_data)

    # Display the results
    print("Model Predictions:", predictions)
    
    # Get the top predicted class and confidence score
    class_names = ['bacterial_blight', 'cercospora_leaf_blight', 'downey_mildew', 'frogeye', 'healthy', 'potassium_deficiency', 'soybean_rust', 'target_spot']
    top_class, confidence = get_top_prediction(predictions, class_names)
    
    # Display the results
    print(f"Predicted Class: {top_class}, Confidence: {confidence:.2f}")
