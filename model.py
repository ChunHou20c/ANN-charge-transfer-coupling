from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(56,56)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(36,activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(18,activation='relu'),
    keras.layers.AlphaDropout(rate = 0.0005),
    keras.layers.Dense(9,activation='relu'),
    keras.layers.Dense(3,activation='relu'),
    keras.layers.Dense(1,activation='linear')
])
