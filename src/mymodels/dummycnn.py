# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:36:51 2024

@author: reinh
"""

from .abstract_cnn import AbstractCNN

# Kelas dummy untuk pengujian
class DummyCNN(AbstractCNN):
    def build_model(self):
        pass  # Tidak diperlukan karena model langsung dimuat dari file