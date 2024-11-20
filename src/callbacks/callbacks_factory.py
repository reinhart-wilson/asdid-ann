from .history_saver import HistorySaver
from .learning_rate_logger import LearningRateLogger
from keras.callbacks import ModelCheckpoint

def create_callback(cb_name, cfg_dict):
    if cb_name == 'history_saver':
        return HistorySaver(cfg_dict['interval'], cfg_dict['save_path'], 
                            save_lr=cfg_dict.get('save_lr',False), 
                            initial_epoch=(cfg_dict['at_epoch'] if 'at_epoch' in cfg_dict else 0))
    elif cb_name == 'learning_rate_logger':
        return LearningRateLogger()
    elif cb_name == 'model_checkpoint':
        return ModelCheckpoint(cfg_dict['save_path'], 
                               save_weights_only=False, 
                               save_freq='epoch', 
                               period = cfg_dict['interval'])
    elif cb_name == 'save_best':
        best_model_checkpoint = ModelCheckpoint(
            filepath=cfg_dict['save_path'] + '/best_model.h5',
            monitor='val_loss',  # Change to your preferred metric
            save_best_only=True,
            mode='min',  # Use 'min' if monitoring loss
            save_weights_only=False
        )
        return best_model_checkpoint
    else:
        raise ValueError(f"Callback {cb_name} is not recognized.")