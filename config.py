# this configuration file define the parameters used in the training
FEATURE = "inter_molecular_element.npy"
RESULT = "results.npy"
LEARNING_RATE = 0.003
LOSS = 'MeanAbsolutePercentageError'
METRICS = 'MeanSquaredError'
TEST_SIZE = 0.1
SCALE = False
BATCH_SIZE = 20
EPOCHES = 250
SAVE_PATH = 'result'
DECAY_RATE = 0.03


#if SCALE = true
if SCALE == True:
    global FEATURE_RANGE
    FEATURE_RANGE = (0,1)

import tensorflow as tf
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
