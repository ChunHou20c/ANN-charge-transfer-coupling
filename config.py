# this configuration file define the parameters used in the training
DATA_DIRECTORY = "/home/ChunHou/Project/ANN_charge-transfer-coupling/data/DBT1"
FEATURE = "CM_matrix.npy"
RESULT = "results.npy"
LEARNING_RATE = 0.006
METRICS = 'MeanAbsolutePercentageError'
LOSS = 'Huber'
TEST_SIZE = 0.2
USE_TEST_SET = True
SCALE = False
BATCH_SIZE = 20
EPOCHES = 10
SAVE_PATH = 'result'
DECAY = True
DECAY_RATE = 0.02

EARLY_STOP = False
#configuration related to early stop
#option = "loss" or "metrics"
MONITOR = "loss" 
PATIENCE = 3 

CHECKPOINT = True
CHECKPOINT_PATH = "checkpoint/cp.ckpt"

if SCALE == True:
    global FEATURE_RANGE
    FEATURE_RANGE = (0,1)

import tensorflow as tf
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)