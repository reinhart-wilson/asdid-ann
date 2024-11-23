from keras import layers
from keras import models
from .abstract_cnn import AbstractCNN
from .base_models import SqueezeNet

class OriginalSqueezeNet(AbstractCNN):   
    
    def build_model(self):            
        base_model = SqueezeNet(
            include_top=False, 
            weights=None,
            input_shape=self.input_shape,
            pooling='avg',
            classes=self.num_classes
            )
        
        # Classification head
        x = base_model.output
        predictions = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=base_model.input, 
                                  outputs=predictions)
        
        
