"""
__title__    = Training_intent.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2023/4/11 16:34
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
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import models
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from New_AR_func import normalization

path = 'dataset_new/Sum'
files = os.listdir(path)
l = []
for file in files:
    if not os.path.isdir(file):
        l.append(file)

GroundTrue = []
Intention = []
DF = []
for file in l:
    data_df = pd.read_csv(path + '/' + file)
    # print(data_df)
    DF.append(data_df.iloc[:, 1:])

data_mean = 106.03646163873081
data_std = 151.59025900983642

for df in DF:
    data_np = df.iloc[:, :].to_numpy()
    data_np[:, :-1] = normalization(data_np[:, :-1], data_mean, data_std)
    for i in range(data_np.shape[0] - 15):
        GroundTrue.append(
            data_np[i + 15, [(j * 4, j * 4 + 1) for j in range(20)]]
        )
        Intention.append(int(data_np[i + 15, -1]))

GroundTrue = np.array(GroundTrue)
Intention = np.array(Intention)
GroundTrue = GroundTrue.reshape([GroundTrue.shape[0], 1, 40])
print(GroundTrue.shape)
print(Intention.shape)

# 进行独热编码
OHE = np.eye(np.max(Intention) + 1)[Intention]
OHE = OHE.reshape([OHE.shape[0], 1, 3])
print(OHE.shape)

X_train, X_test, y_train, y_test = train_test_split(
    GroundTrue, OHE, test_size=.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=.2, random_state=42)

print(X_train.shape)
print(y_train.shape)

model = load_model('dataset_new/model_intent.h5')

model.summary()

checkpoint_path = 'dataset_new/best_model_intent.h5'
checkpoint = ModelCheckpoint(
    checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

history = model.fit(
    X_train, y_train, epochs=100, batch_size=5, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stop])


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('route_model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('route_model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()
