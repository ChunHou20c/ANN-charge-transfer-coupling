#all the data preprocessing is done here

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import config 
_feature = np.load("../"+config.FEATURE)
_result = np.load("../"+config.RESULT)

if config.SCALE == True:
    scaler = MinMaxScaler(feature_range=config.FEATURE_RANGE)
    scaler.fit(_feature)
    _feature = scaler.transform(_feature)

x_train, x_val, y_train, y_val = train_test_split(_feature, _result, test_size = config.TEST_SIZE, random_state = 0)
x_test, y_test = x_val[-5:], y_val[-5:]
x_val, y_val = x_val[:-5], y_val[:-5]
