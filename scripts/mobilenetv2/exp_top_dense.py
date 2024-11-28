import sys, os

working_dir = os.path.abspath(os.path.dirname(__file__))
configs_dir = os.path.join(working_dir, '..', '..', 'config')
src_dir = os.path.join(working_dir, '..', '..', 'src')
sys.path.append(src_dir)
sys.path.append(configs_dir)

# Set mixed precision
from utils import general_utils as gutils
gutils.use_mixed_precision()

from callbacks.save_latest_model import SaveLatestModel
from configs.other_configs import data_info as dinfo
from mymodels.mobilenetv2 import MyMobileNetV2
from tensorflow.keras import optimizers, metrics, callbacks, layers, regularizers 
from utils import training_utils as tutils

# Params
mode        = 0 # 0 cont 1 train
last_epoch  = 0
epoch       = 200
lr          = 1e-4
batch_size  = 12
dense       = 512*2 # Currently tuning
dropout     = 0
weights     = None
wdecay      = 0
alpha       = 1.0 # Bukan learning rate
lr_config   = {
    'init_value' : 1e-4,
    'scheduler_config' : {
        'name' : 'cosine_decay',
        'lr_alpha' : 1e-2,
        'epochs_to_decay' : epoch
    }
}

# Paths
PARAM_VAL               = dense
TUNED_PARAM             = 'top_layers_dense'
SAVE_PATH               = f'../../training_result/mobilenetv2/{TUNED_PARAM}/{PARAM_VAL}'
BEST_MODEL_FILENAME     = 'best_model_epoch.tf'
LAST_MODEL_FILENAME     = 'model_at_{epoch}.tf'
LATEST_MODEL_FILENAME   = 'latest_model.tf'
LOGDIR                  = os.path.join(SAVE_PATH, "logs")

# Load data
train_data_dir = os.path.join(dinfo.DATA_PATH, 'train')
val_data_dir = os.path.join(dinfo.DATA_PATH, 'validation')
train_datagen = gutils.make_datagen(train_data_dir, 
                                    dinfo.IMG_RES, 
                                    batch_size,
                                    augment=False)
val_datagen = gutils.make_datagen(val_data_dir, 
                                  dinfo.IMG_RES, 
                                  batch_size,
                                  shuffle=False)

# Callbacks
model_callbacks = [
    callbacks.TensorBoard(log_dir=LOGDIR),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(SAVE_PATH, BEST_MODEL_FILENAME),
        monitor='val_loss',
        save_best_only=True,
        mode='min', 
        save_weights_only=False
    ),
    # SaveLatestModel(os.path.join(SAVE_PATH, LATEST_MODEL_FILENAME))
]

# Definisi model
model = MyMobileNetV2(dinfo.IMG_DIM, 
                      dinfo.NUM_CLASSES, 
                      dense_neuron_num=dense,
                      dropout=dropout,
                      weights = weights,
                      weight_decay=wdecay,
                      alpha=alpha
                      )

if mode == 1:
    model.build_model()
    
    
    # Learning Rate
    if 'scheduler_config' in lr_config:
        lr_scheduler_config = lr_config['scheduler_config']
        if lr_scheduler_config['name'] == 'cosine_decay':
            decay_steps = tutils.compute_decay_steps(
                train_datagen.samples,                                                  
                batch_size,                                                  
                lr_scheduler_config['epochs_to_decay'])
            lr = optimizers.schedules.CosineDecay(
                initial_learning_rate=lr_config['init_value'],
                alpha = lr_scheduler_config['lr_alpha'],
                decay_steps = decay_steps
            )
    else:
        lr = lr_config['init_value']
    
    # Compile
    optimizer = optimizers.Adam(learning_rate=lr)
    model_metrics = [
        'accuracy',
        metrics.Recall(name='recall')    
        ]
    model.compile_model(optimizer=optimizer, metrics=model_metrics)
    
    # Train
    model.train(train_datagen,
                val_datagen, 
                epochs=epoch,
                batch_size=batch_size,
                callbacks=model_callbacks)
    model.save_model(os.path.join(SAVE_PATH, LAST_MODEL_FILENAME.format(epoch=epoch)))
if mode == 0:
    model.load_model(os.path.join(SAVE_PATH, LATEST_MODEL_FILENAME))
    model.train(train_datagen,
                val_datagen, 
                epochs=epoch,
                batch_size=batch_size,
                callbacks=model_callbacks,
                initial_epoch=last_epoch)
    