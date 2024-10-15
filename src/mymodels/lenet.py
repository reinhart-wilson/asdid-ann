from keras import models, layers
from .abstract_cnn import AbstractCNN

class LeNet (AbstractCNN):    

    def build_model(self):
        model = models.Sequential()
    
        # Lapisan ke-1: Convolutional + MaxPooling
        model.add(layers.Conv2D(6, kernel_size=(5, 5), 
                                activation='relu', 
                                input_shape=self.input_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
        # Lapisan ke-2: Convolutional + MaxPooling
        model.add(layers.Conv2D(16, kernel_size=(5, 5), 
                                activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
        model.add(layers.Flatten())
    
        # Lapisan ke-3: Fully Connected 
        model.add(layers.Dense(120, activation='relu'))
    
        # Lapisan ke-4: Fully Connected
        model.add(layers.Dense(84, activation='relu'))
    
        # Lapisan output
        model.add(layers.Dense(self.num_classes, activation='softmax'))
    
        self.model = model   