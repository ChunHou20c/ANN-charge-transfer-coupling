# this configuration file define the parameters used in the training
FEATURE = "CM_matrix_reshaped.npy"
RESULT = "results.npy"
LEARNING_RATE = 0.004
LOSS = 'MeanSquaredError'
METRICS = 'MeanAbsolutePercentageError'
TEST_SIZE = 0.2
SCALE = False
BATCH_SIZE = 20
EPOCHES = 1000
SAVE_PATH = 'result'
DECAY_RATE = 0.020


#if SCALE = true
if SCALE == True:
    global FEATURE_RANGE
    FEATURE_RANGE = (0,1)

import tensorflow as tf
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
