#this file define the callback used in the training process
import tensorflow as tf
import schedular
import config

CALLBACKS = []

if (config.DECAY == True):
    CALLBACKS.append(tf.keras.callbacks.LearningRateScheduler(schedular.lr_exp_decay))

if (config.EARLY_STOP == True):
    CALLBACKS.append(tf.keras.callbacks.EarlyStopping(monitor=config.MONITOR, patience=config.PATIENCE))

if (config.CHECKPOINT == True):
    CALLBACKS.append(tf.keras.callbacks.ModelCheckpoint(filepath=config.CHECKPOINT_PATH, 
        monitor='val_loss', 
        save_best_only=True, 
        save_weights_only=True, 
        verbose=0))