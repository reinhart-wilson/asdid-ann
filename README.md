# ASDID ANN

ASDID ANN merupakan proyek personal yang bertujuan untuk mengembangkan model *convolutional neural network* (CNN) untuk mengklasifikasi set data [Auburn Soybean Disease Image Dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.41ns1rnj3) dan dapat diintegrasikan pada aplikasi *smartphone* Android tanpa memerlukan koneksi internet. 

Model CNN yang dikembangkan memanfaatkan library [TensorFlow untuk Python](https://github.com/tensorflow/tensorflow).

## Ketergantungan
Proyek ini dikembangkan dalam lingkungan dengan pustaka-pustaka berikut:
- Python 3.9
- TensorFlow dan Tensorboard 2.10 (versi terakhir dengan dukungan GPU di Windows)
- Numpy 1.26.4
- scikit-learn 1.5.1

## Penggunaan
Proyek ini terdiri atas dua *folder*:
  1. `src`: Berisi source code yang digunakan dalam eksperimen.
  2. `scripts`: Berisi skrip-skrip yang digunakan untuk menjalankan program. 

Semua data yang diperlukan disimpan di `[root]/training_result`. Ini termasuk set data citra untuk pelatihan dan pengujian serta set data yang diperoleh dari TensorBoard untuk visualisasi.

Set data dapat diunduh dari [Auburn Soybean Disease Image Dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.41ns1rnj3). Perlu diperhatikan bahwa nama folder harus disesuaikan dengan skrip-skrip yang diperlukan.

Hasil pelatihan akan disimpan di `[root]/training_result`. Folder akan dibuat secara otomatis bila belum pernah dibuat sebelumnya.
