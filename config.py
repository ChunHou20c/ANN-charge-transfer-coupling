"""
This file contains all the configuration for the training of model
"""

DATA_DIRECTORY = "./data"
FEATURE = "CM-inter-duplicate/feature.npy"
RESULT = "CM-inter-duplicate/label.npy"
LEARNING_RATE = 0.006
METRICS = 'MeanAbsolutePercentageError'
LOSS = 'Huber'
TEST_SIZE = 0.2
USE_TEST_SET = True
SCALE = False

ENABLE_OVERSAMPLING = True

BATCH_SIZE = 20
EPOCHES = 10
SAVE_PATH = 'result'
DECAY = True
DECAY_RATE = 0.02

EARLY_STOP = True
#configuration related to early stop
#option = "loss" or "metrics"
MONITOR = "loss" 
PATIENCE = 3 

CHECKPOINT = True
CHECKPOINT_PATH = "checkpoint/cp.ckpt"

import tensorflow as tf
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
