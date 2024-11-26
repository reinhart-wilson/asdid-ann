from keras import layers
from keras import models
from .abstract_cnn import AbstractCNN
from .base_models import SqueezeNet

class MySqueezeNet(AbstractCNN):   
    
    def build_model(self, 
                    include_classification_head=True, 
                    include_top = False):   
         
        base_model = SqueezeNet(
            include_top=include_top, 
            weights=None,
            input_shape=self.input_shape,
            pooling='avg',
            classes=self.num_classes
        )
        
        # Base model output
        x = base_model.output
        
        # Tambahkan head klasifikasi jika diinginkan
        if include_top is not True:
            if include_classification_head:
                x = layers.Dense(self.num_classes, activation='softmax')(x)
                self.classification_head = None
            else:
                self.classification_head = layers.Dense(self.num_classes, 
                                                        activation='softmax')
        else:
            self.classification_head = None
        
        # Buat model awal
        self.model = models.Model(inputs=base_model.input, outputs=x)
        
    def get_classification_head(self):
        return self.classification_head
        


        
