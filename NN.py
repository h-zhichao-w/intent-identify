# For route prediction
from keras import layers
from keras import models
from tensorflow import keras

# Best Individual: {'n_lstm': 128, 'n_dense1': 64, 'n_dense2': 256, 'learning_rate': 0.001}

inputs = layers.Input(shape=(5, 82))

lstm1 = layers.LSTM(128, return_sequences=True)(inputs)
lstm2 = layers.LSTM(128, return_sequences=True)(lstm1)

attention = layers.Attention()([lstm1, lstm2])

dense1 = layers.Dense(64, activation='relu')(attention)

flatten = layers.Flatten()(dense1)

dense2 = layers.Dense(256, activation='relu')(flatten)

# dropout = layers.Dropout(0.2)(dense)

outputs = layers.Dense(40, activation='linear')(dense2)

model = models.Model(inputs=inputs, outputs=outputs)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

model.summary()

model.save('dataset_new/route_model.h5')