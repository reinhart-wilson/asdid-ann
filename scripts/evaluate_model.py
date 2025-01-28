import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Menambahkan path untuk import modul
working_dir = os.path.abspath(os.path.dirname(__file__))
configs_dir = os.path.join(working_dir, '..', 'config')
src_dir = os.path.join(working_dir, '..', 'src')
sys.path.append(src_dir)
sys.path.append(configs_dir)

from utils import general_utils as gutils, evaluation_utils as eutils
gutils.use_mixed_precision()

from mymodels.mobilenetv2 import MyMobileNetV2

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# Load model
highest_acc_epoch = 0
highest_acc = 0
model_path = '../training_result/mobilenetv2/transfer_learning/imagenet/best_model_epoch.tf'
model = MyMobileNetV2((0), 8)
model.load_model(model_path)

# Load data
data_dir = '../Dataset/split_prepped_data _with_added_data_all_class/test'
# data_dir = '../Dataset/prepped_additional_data_lagi_split/test'
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
                             title='Hasil Klasifikasi Model pada Set Data Pengujian')

# Prediksi Kelas
predicted_classes = np.argmax(predictions, axis=1)

# Laporan Klasifikasi
report = classification_report(true_classes, 
                               predicted_classes, 
                               target_names=class_labels,
                               output_dict=False)
print(report)

# Identifikasi indeks gambar yang salah diprediksi
incorrect_indices = np.where(predicted_classes != true_classes)[0]

# Ambil gambar yang salah diprediksi
images = test_data.filepaths  # Filepath dari dataset
incorrect_images = [images[i] for i in incorrect_indices]

# Tampilkan gambar yang salah diprediksi
plt.figure(figsize=(12, 12))

for i, idx in enumerate(incorrect_indices[:16]):  # Menampilkan hingga 16 gambar
    plt.subplot(4, 4, i + 1)
    # Membaca gambar dari filepath
    img = plt.imread(images[idx])
    plt.imshow(img)
    plt.axis('off')
    
    # Menampilkan label sebenarnya dan prediksi
    true_label = class_labels[true_classes[idx]]
    pred_label = class_labels[predicted_classes[idx]]
    plt.title(f"True: {true_label}\nPred: {pred_label}", color="red")

plt.tight_layout()
plt.show()

