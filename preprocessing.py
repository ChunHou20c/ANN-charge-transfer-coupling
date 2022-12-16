#all the data preprocessing is done here
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler

import config
import oversampler

_feature = np.load(config.DATA_DIRECTORY+'/'+config.FEATURE)
_result = np.load(config.DATA_DIRECTORY+'/'+config.RESULT)

#if config.SCALE == True:
#    scaler = MinMaxScaler(feature_range=config.FEATURE_RANGE)
#    scaler.fit(_feature)
#    _feature = scaler.transform(_feature)

x_train, x_val, y_train, y_val = train_test_split(_feature, _result, test_size = config.TEST_SIZE, random_state = 0)

if (config.USE_TEST_SET == True):
    x_test, y_test = x_val[-5:], y_val[-5:]
    x_val, y_val = x_val[:-5], y_val[:-5]

if (config.ENABLE_OVERSAMPLING == True):
    #this part will apply overampling to the training set only

    x_train, y_train = oversampler.oversampler(x_train, y_train)
