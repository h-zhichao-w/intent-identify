"""
__title__    = NN_intent.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2023/4/11 16:29
__Software__ = Pycharm
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓    ┏┓
            ┏┛┻━━━┛ ┻┓
            ┃         ┃
            ┃  ┳┛  ┗┳  ┃
            ┃     ┻    ┃
            ┗━┓      ┏━┛
              ┃      ┗━━━┓
              ┃  神兽保佑  ┣┓
              ┃　永无BUG！ ┏┛
                ┗┓┓┏━┳┓┏┛
                 ┃┫┫  ┃┫┫
                 ┗┻┛  ┗┻┛
"""
# For route prediction
from keras import layers
from keras import models
from tensorflow import keras

# Best Individual: {'n_lstm': 256, 'n_dense1': 256, 'n_dense2': 64, 'learning_rate': 0.001}

inputs = layers.Input(shape=(1, 40))

lstm = layers.LSTM(256, return_sequences=True)(inputs)

dense1 = layers.Dense(64, activation='relu')(lstm)

dense2 = layers.Dense(256, activation='relu')(dense1)

outputs = layers.Dense(3, activation='softmax')(dense2)

model = models.Model(inputs=inputs, outputs=outputs)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.save('dataset_new/model_intent.h5')