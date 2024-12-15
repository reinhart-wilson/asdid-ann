import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import pickle
import tensorflow as tf
import time
from PIL import Image
from matplotlib.ticker import MultipleLocator
from tensorflow.python.summary.summary_iterator import summary_iterator
from sklearn.metrics import  ConfusionMatrixDisplay, confusion_matrix

def load_image(image_path, target_size=(224, 224)):
    """
    Memuat dan melakukan praproses pada citra

    Parameters
    ----------
    image_path : str
        Path ke file citra
    target_size : tuple, optional
        Ukuran target. Nilai default adalah (224, 224).

    Returns
    -------
    image_array : np.array
        Data yang sudah diproses, dalam bentuk array.

    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array


def get_inference_time(interpreter, input_data):
    """
    Menjalankan inferensi data masukan menggunakan model tensorflow lite untuk
    mengukur waktu yang diperlukan
    
    Parameters
    ----------
    interpreter: tf.lite.Interpreter
        TFLite model interpreter.
    input_data: np.array
        Data citra input yang sudah dipraproses
        
    Returns
    ----------
    float
        Waktu inferensi dalam milidetik
    """
    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_index, input_data)

    start_time = time.time()  
    interpreter.invoke()      
    end_time = time.time()   
    
    return end_time - start_time

def load_history(file_path):
    """
    Memuat riwayat pelatihan (history) model dari file yang disimpan menggunakan 
    pickle. 

    Parameters
    ----------
    file_path : str
        Path lengkap ke file yang berisi objek riwayat model yang telah disimpan 
        dengan format pickle.

    Returns
    -------
    history : dict
        Objek yang berisi riwayat pelatihan model, biasanya berupa dictionary dengan metrik-metrik
        yang dicatat selama pelatihan, seperti `loss`, `accuracy`, dll.

    """
    with open(file_path,'rb') as file:
        history = pickle.load(file)
    return history

def show_tensorboard_plots(tensorboard_data_path, csv_files, labels, 
                           epochs_monitored, show_legend=False, line_colors=None):
    """
    Menampilkan grafik metrik (Akurasi, Loss, Recall) dari data TensorBoard yang 
    disimpan dalam file CSV. Fungsi ini membaca data dari file CSV yang berisi 
    metrik pelatihan dan menghasilkan grafik untuk beberapa metrik utama seperti 
    akurasi, loss, dan recall. Grafik ini akan menampilkan perubahan metrik tersebut 
    seiring berjalannya epoch selama pelatihan. File CSV didapatkan dengan cara 
    mengunduh dari TensorBoard.

    Parameters
    ----------
    tensorboard_data_path : str
        Path ke direktori yang berisi subdirektori untuk setiap metrik (Akurasi, 
        Loss, Recall) yang berisi file CSV.
    
    csv_files : list of str
        Daftar nama file CSV yang berisi data metrik dari TensorBoard untuk setiap 
        eksperimen.
    
    labels : list of str
        Daftar label yang digunakan untuk menandai setiap plot (misalnya, nama 
        eksperimen atau model).
    
    epochs_monitored : int
        Jumlah epoch yang akan dipantau dan diplotkan pada grafik.
    
    show_legend : bool, optional
        Jika True, akan menampilkan legenda pada grafik. Default adalah False.
    
    line_colors : list of str, optional
        Daftar warna yang digunakan untuk setiap garis pada grafik. Jika None, 
        warna default akan digunakan.

    Returns
    -------
    None
        Fungsi ini hanya menampilkan grafik tanpa mengembalikan nilai.
    """
    metrics = ['Akurasi', 'Loss', 'Recall']
    
    for metric in metrics:
        data_path = os.path.join(tensorboard_data_path, metric)
        
        for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
            data = pd.read_csv(os.path.join(data_path, csv_file))
            data = data[data['Step'] < epochs_monitored]
            
            steps = data['Step'] + 1  # Epoch - 1
            if metric == 'Akurasi':
                values = data['Value'] * 100  # Metric to plot
            else:
                values = data['Value']
            
            if line_colors and i < len(line_colors):
                plt.plot(steps, values, label=label, color=line_colors[i])
            else:
                plt.plot(steps, values, label=label)
    
        plt.xlabel("Epoch")
        ylim = math.ceil(max(values))
        if metric == 'Akurasi':
            metric = metric+' (%)'
            y_interval = math.ceil(ylim/10)
        if metric == 'Loss':
            y_interval = math.ceil(ylim/10)
        else:
            y_interval = (ylim/10)
        plt.ylabel(metric)
        plt.title("")
        plt.gca().yaxis.set_major_locator(MultipleLocator(19)) 
        plt.ylim(0, 100)
        if show_legend:
            plt.legend()  # Add a legend
        else:
            plt.grid(True)  # Optional: Add a grid
        plt.tight_layout()
    
        plt.show()


def extract_metrics_from_logs(log_dir, loss_key='epoch_loss', 
                              acc_key='epoch_accuracy', 
                              recall_key='epoch_recall'):
    """
    Fungsi ini memindai seluruh file log yang ada di dalam direktori `log_dir` 
    dan mengekstrak nilai metrik tertentu, seperti loss, akurasi, dan recall, 
    dari setiap file log. Nilai metrik yang diekstrak adalah nilai terbaik. 
    
    Parameters
    ----------
    log_dir : str
        Path ke direktori yang berisi file log yang dihasilkan oleh TensorBoard. 
        
    loss_key : str, optional
        Nama tag yang digunakan untuk menandai nilai loss dalam log. Default 
        adalah 'epoch_loss'.

    acc_key : str, optional
        Nama tag yang digunakan untuk menandai nilai akurasi dalam log. Default 
        adalah 'epoch_accuracy'.

    recall_key : str, optional
        Nama tag yang digunakan untuk menandai nilai recall dalam log. Default 
        adalah 'epoch_recall'.

    Returns
    -------
    dict
        Dictionary yang berisi nilai terbaik untuk loss, akurasi, dan recall yang 
        ditemukan dalam log.Keys yang ada dalam dictionary adalah:
        - 'lowest_loss': Loss terendah yang ditemukan.
        - 'highest_accuracy': Akurasi tertinggi yang ditemukan.
        - 'highest_recall': Recall tertinggi yang ditemukan.
    """

    lowest_loss = float('inf')
    highest_accuracy = float('-inf')
    highest_recall = float('-inf')
    
    # Iterate over all event files in the log directory
    for file_name in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file_name)
        
        # Parse the event file
        for event in summary_iterator(file_path):
            
            for value in event.summary.value:
                
                if value.tag == loss_key:
                    lowest_loss = min(lowest_loss, float(tf.make_ndarray(value.tensor)))
                elif value.tag == acc_key:
                    highest_accuracy = max(highest_accuracy, float(tf.make_ndarray(value.tensor)))
                elif value.tag == recall_key or value.tag == recall_key+'_1' or value.tag == recall_key+'_2':
                    highest_recall = max(highest_recall, float(tf.make_ndarray(value.tensor)))
                    
    
    return {
        'lowest_loss': lowest_loss,
        'highest_accuracy': highest_accuracy,
        'highest_recall': highest_recall
    }

def get_flops(model, model_inputs) -> float:
    """
    Menghitung jumlah FLOPs (Floating Point Operations) dari model TensorFlow
    atau Keras untuk inferensi pada satu sample input.

    Parameters
    ----------
    model : tf.keras.Model atau tf.keras.Sequential
        Model Keras atau TensorFlow yang ingin dihitung FLOPs-nya. Model harus 
        berupa instance dari `tf.keras.Model` atau `tf.keras.Sequential`.
    model_inputs : list of tf.Tensor
        Contoh input model dalam bentuk tensor TensorFlow. Tensor ini digunakan 
        untuk menentukan spesifikasi input saat membekukan grafik model.

    Raises
    ------
    ValueError
        Jika `model` bukan instance dari `tf.keras.Model` atau 
        `tf.keras.Sequential`.
    
    Returns
    -------
    float
        Jumlah total FLOPs yang dibutuhkan model untuk melakukan inferensi pada 
        satu input, dikembalikan dalam unit FLOP.

    """

    # Validasi input
    if not isinstance(
        model, (tf.keras.models.Sequential, tf.keras.models.Model)
    ):
        raise ValueError(
            "Calculating FLOPS is only supported for "
            "`tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    # Atur agar FLOP yang dihitung hanya untuk 1 operasi.
    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
        for inp in model_inputs
    ]

    # Bekukan grafik agar dapat dihitung FLOP-nya oleh profiler.
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Kalkulasi FLOPs dengan tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )
    tf.compat.v1.reset_default_graph()
    
    return (flops.total_float_ops)/2

def plot_confusion_matrix(predictions, true_classes, class_labels, rotation=0,
                          title='Confusion Matrix', xlabel='Label Terprediksi',
                          ylabel='Label Sebenarnya'):
    """
    Membuat dan menampilkan matriks kebingunguan (confusion matrix) berdasarkan 
    prediksi model dan kelas sebenarnya. Plot digambar dengan `matplotlib`.
    
    Parameters
    ----------
    predictions : array-like, shape (n_samples, n_classes)
        Matriks prediksi dari model, dimana setiap elemen adalah probabilitas 
        atau skor untuk setiap kelas. Fungsi ini akan memilih kelas dengan 
        probabilitas tertinggi sebagai prediksi akhir.

    true_classes : array-like, shape (n_samples,)
        Array berisi kelas sebenarnya (label yang benar) dari data.

    class_labels : list of str
        Daftar label kelas yang digunakan untuk memberi nama pada sumbu x dan y 
        dari matriks kebingunguan.

    rotation : int, opsional
        Sudut rotasi label pada sumbu x (label kelas terprediksi). Default adalah 
        0 derajat.

    title : str, opsional
        Judul grafik. Default adalah 'Confusion Matrix'.

    xlabel : str, opsional
        Label untuk sumbu x (kelas yang diprediksi). Default adalah 'Label Terprediksi'.

    ylabel : str, opsional
        Label untuk sumbu y (kelas sebenarnya). Default adalah 'Label Sebenarnya'.

    Returns
    -------
    conf_matrix : ndarray, shape (n_classes, n_classes)
        Matriks kebingunguan yang dihitung berdasarkan `predictions` dan `true_classes`.
        Matriks ini menggambarkan seberapa banyak prediksi yang benar dan salah 
        pada setiap kelas.
    """
    # Buat confusion matrix
    predicted_classes = np.argmax(predictions, axis=1) # Mengambil kelas dengan probabilitas tertinggi
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.show()
    return conf_matrix


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
    

def plot_loss(history, title=None):
    # Plot training loss vs. validation loss
    plt.figure(figsize=(12, 4))
    
    if title:
        plt.suptitle(title, fontsize=16)
    
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
