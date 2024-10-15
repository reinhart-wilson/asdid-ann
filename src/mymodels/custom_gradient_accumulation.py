# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:37:30 2024

@author: Mr. For Example
"""
import tensorflow as tf
 
class CustomGradientAccumulation(tf.keras.Model):
    def __init__(self, n_gradients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Menyimpan nilai n_gradients sebagai tensor konstanta. Nantinya akan 
        # digunakan sebagai acuan berapa banyak langkah (step) gradien yang 
        # perlu terakumulasi sebelum optimizer diterapkan.
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        
        # Variabel untuk menghitung langkah gradien yang sudah terakumulasi 
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        
        # List untuk menampung gradien yang akan diakumulasi. Setiap elemen 
        # dalam list ini adalah variabel yang bentuknya sama dengan variabel 
        # yang dapat dilatih (trainable_variables) dalam model. Digunakan untuk 
        # menyimpan gradien setiap batch.
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) 
                                      for v in self.trainable_variables]

    def train_step(self, data):
        # Setiap kali fungsi ini dipanggil, jumlah langkah akumulasi akan bertambah satu.
        self.n_acum_step.assign_add(1)

        # Memisahkan data input menjadi x (fitur) dan y (label).
        x, y, sample_weight = data
        
        # Menggunakan GradientTape untuk merekam operasi yang dilakukan selama perhitungan forward pass.
        with tf.GradientTape() as tape:
            # Melakukan prediksi pada input x.
            y_pred = self(x, training=True)
            
            # Menghitung loss menggunakan fungsi loss yang telah dikompilasi sebelumnya.
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Menghitung gradien loss terhadap variabel yang dapat dilatih.
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Menambahkan gradien yang baru dihitung ke dalam akumulasi gradien yang telah disimpan.
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
 
        # Jika gradien sudah terakumulasi sebanyak n_gradients, maka terapkan optimizer.
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), 
                self.apply_accu_gradients, 
                lambda: None)

        # Memperbarui metrik yang telah dikompilasi (misalnya akurasi, loss) dengan prediksi terbaru.
        self.compiled_metrics.update_state(y, y_pred)
        
        # Mengembalikan hasil dari metrik yang diperbarui dalam bentuk dictionary (misalnya loss dan akurasi).
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # Menerapkan gradien yang telah terakumulasi ke variabel yang dapat dilatih
        # menggunakan optimizer yang telah dikompilasi sebelumnya.
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # Reset jumlah langkah akumulasi ke 0.
        self.n_acum_step.assign(0)
        
        # Mengatur ulang akumulasi gradien ke nol untuk setiap variabel.
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))
