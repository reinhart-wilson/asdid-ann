from .alexnet import AlexNet
from .lenet import LeNet
from .mobilenetv1 import MyMobileNetV1, OriginalMobileNet
from .mobilenetv2 import MyMobileNetV2, OriginalMobileNetV2
from .mobilenetv3 import MyMobileNetV3
from .squeezenet import MySqueezeNet
from .nasnetmobile import OriginalNasNetMobile
from .efficientnet import EfficientNet
from .efficientnetv2 import EfficientNetV2
from .vgg16 import VGG16

def create_model(input_shape, num_classes, model_config_dict):
    model_name = model_config_dict['model_name']
    if model_name == 'mobilenetv1':
        return MyMobileNetV1(input_shape, 
                           num_classes, 
                           alpha = model_config_dict['alpha'],
                           dense_neuron_num= model_config_dict['dense'])
    elif model_name == 'originalmobilenetv1':
        return OriginalMobileNet(input_shape, 
                           num_classes, 
                           alpha = model_config_dict['alpha'],
                           dense_neuron_num= model_config_dict['dense'])
    elif model_name == 'originalmobilenetv2':
        return OriginalMobileNetV2(input_shape, 
                           num_classes, 
                           alpha = model_config_dict['alpha'],
                           dense_neuron_num= model_config_dict['dense'])
    elif model_name == 'originalnasnetmobile':
        return MySqueezeNet(input_shape, num_classes)
    elif model_name == 'originalsqueezenet':
        return OriginalNasNetMobile(input_shape, num_classes)
    elif model_name == 'mobilenetv2':
        weights = model_config_dict.get('weights', None)  # Default is None
        dropout = model_config_dict.get('dropout', 0.0)  # Default dropout rate is 0.0
        batch_norm = model_config_dict.get('batch_norm', False)  # Default is no batch norm
        weight_decay = model_config_dict.get('weight_decay', 0)
        if weights not in [None, 'imagenet']:
            raise ValueError("Weights not recognized.")
        n_gradients = model_config_dict.get('n_gradients', 0)
        
        return MyMobileNetV2(input_shape, 
                           num_classes, 
                           alpha = model_config_dict['alpha'],
                           dense_neuron_num= model_config_dict['dense'],
                           weights = weights,
                           dropout=dropout,
                           weight_decay=weight_decay,
                           n_gradients=n_gradients)
    elif model_name == 'mobilenetv3':
        weights = model_config_dict.get('weights', None)  # Default is None
        dropout = model_config_dict.get('dropout', 0.0)  # Default dropout rate is 0.0
        weight_decay = model_config_dict.get('weight_decay', 0)
        if weights not in [None, 'imagenet']:
            raise ValueError("Weights not recognized.")
        
        return MyMobileNetV3(input_shape, 
                           num_classes, 
                           alpha = model_config_dict['alpha'],
                           dense_neuron_num= model_config_dict['dense'],
                           model_variant = model_config_dict['variant'],
                           weights = weights,
                           dropout=dropout,
                           weight_decay=weight_decay)
    elif model_name == 'alexnet':
        return AlexNet(input_shape, num_classes)
    elif model_name == 'lenet':
        return LeNet(input_shape, num_classes)
    elif model_name == 'efficientnet':
        return EfficientNet(input_shape, num_classes)
    
    elif model_name == 'efficientnetv2b0':
        return EfficientNetV2(input_shape, 
                           num_classes
                           # dense_neuron_num= model_config_dict['dense']
                           )
    elif model_name == 'vgg16':
        return VGG16(input_shape, num_classes)
    else:
        raise ValueError(f"Model {model_name} is not recognized.")