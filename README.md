# ASDID ANN

ASDID ANN merupakan proyek personal yang bertujuan untuk mengembangkan model *convolutional neural network* (CNN) untuk mengklasifikasi set data [Auburn Soybean Disease Image Dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.41ns1rnj3) dan dapat diintegrasikan pada aplikasi *smartphone* Android tanpa memerlukan koneksi internet. 

Model CNN yang dikembangkan memanfaatkan library [TensorFlow untuk Python](https://github.com/tensorflow/tensorflow).

## Penggunaan
Proyek ini terdiri atas dua *folder*:
<ol>
  <li>`src`: Berisi source code yang digunakan dalam eksperimen.</li>
  <li>`scripts`: Berisi skrip-skrip yang digunakan untuk menjalankan program</li>
</ol>

Set data yang digunakan diletakkan di folder `dataset` yang diletakkan di `root` dari struktur proyek. Hasil pelatihan akan disimpan di `root/training_result`. 
