import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import  ConfusionMatrixDisplay, confusion_matrix

def load_history(file_path):
    with open(file_path,'rb') as file:
        history = pickle.load(file)
    return history

def predict():
    pass

# def plot_confusion_matrix(model, test_generator, class_labels, text_rotation=45):
#     """
#     Fungsi untuk memplot confusion matrix dari model TensorFlow dengan opsi mengatur kemiringan teks sumbu x.
    
#     Parameters:
#     model : TensorFlow model
#         Model TensorFlow yang telah dilatih.
#     test_generator : DirectoryIterator
#         Generator untuk data uji.
#     class_labels : list
#         Daftar label kelas.
#     text_rotation : int, optional (default=45)
#         Derajat kemiringan teks pada sumbu x.
#     """
#     # Membuat prediksi
#     predictions = model.predict(test_generator)
#     predicted_classes = np.argmax(predictions, axis=1)  # Mengambil kelas dengan probabilitas tertinggi
#     true_classes = test_generator.classes

#     # Membuat confusion matrix
#     conf_matrix = confusion_matrix(true_classes, predicted_classes)

#     # Plot confusion matrix untuk setiap kelas
#     for i in range(len(class_labels)):
#         TP = conf_matrix[i, i]
#         FP = conf_matrix[:, i].sum() - TP
#         FN = conf_matrix[i, :].sum() - TP
#         TN = conf_matrix.sum() - (FP + FN + TP)

#         matrix = np.array([[TP, FP], [FN, TN]])
        
#         disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Positive", "Negative"])
#         disp.plot(cmap=plt.cm.Blues, values_format='d')
#         plt.title(f'Confusion Matrix for {class_labels[i]}')
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')
#         plt.xticks([0, 1], [f'{class_labels[i]}', f'Not {class_labels[i]}'], rotation=text_rotation)
#         plt.yticks([0, 1], [f'{class_labels[i]}', f'Not {class_labels[i]}'])
#         plt.show()

def plot_confusion_matrix_per_class(conf_matrix, class_labels, class_index, rotation=0):
    TP = conf_matrix[class_index, class_index]
    FP = conf_matrix[:, class_index].sum() - TP
    FN = conf_matrix[class_index, :].sum() - TP
    TN = conf_matrix.sum() - (FP + FN + TP)

    matrix = np.array([[TP, FP], [FN, TN]])
    
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Positive", "Negative"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix for {class_labels[class_index]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], [f'{class_labels[class_index]}', f'Not {class_labels[class_index]}'])
    plt.yticks([0, 1], [f'{class_labels[class_index]}', f'Not {class_labels[class_index]}'])
    plt.show()
    
def plot_confusion_matrix(predictions, true_classes, class_labels, rotation=0):
    # Buat confusion matrix
    predicted_classes = np.argmax(predictions, axis=1) # Mengambil kelas dengan probabilitas tertinggi
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xticks(rotation=rotation)
    plt.show()
    return conf_matrix


def plot_loss(history):
    # Plot training loss vs. validation loss
    plt.figure(figsize=(12, 4))
    
    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history['loss']) + 1), history['loss'], label='Training Loss')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.legend()
    
    # Menampilkan bilangan bulat pada sumbu x dengan interval yang sesuai
    epoch_range = range(1, len(history['loss']) + 1)
    plt.xticks(epoch_range[::max(len(epoch_range)//10, 1)], rotation=0)  # Interval setiap 10 epoch
    plt.grid()
    
    # Plot Training Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['accuracy']) + 1), history['accuracy'], label='Training Accuracy')
    plt.plot(range(1, len(history['val_accuracy']) + 1), history['val_accuracy'], label='lr')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.legend()
    
    # Menampilkan bilangan bulat pada sumbu x dengan interval yang sesuai
    plt.xticks(epoch_range[::max(len(epoch_range)//10, 1)], rotation=0)  # Interval setiap 10 epoch
    plt.grid()
    
    # Tampilkan plot
    plt.tight_layout()  # Agar subplot tidak tumpang tindih
    plt.show()
    
def plot_lr(history):
    # Plot Training Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['lr']) + 1), history['lr'], label='lr')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Overtime')
    plt.legend()
    plt.show()
    
def print_parameters(use_gpu, batch_size, image_size, epochs,  
                     seed, alpha):
    print(f"{'Parameter':<15} {'Value'}")
    print("-" * 30)
    print(f"{'use_gpu':<15} {use_gpu}")
    print(f"{'batch_size':<15} {batch_size}")
    print(f"{'image_size':<15} {image_size}")
    print(f"{'epochs':<15} {epochs}")
    print(f"{'seed':<15} {seed}")
    print(f"{'alpha':<15} {alpha:.4f}")
        