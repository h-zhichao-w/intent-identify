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

Data5Sec = []
GroundTrue = []
DF = []
for file in l:
    data_df = pd.read_csv(path + '/' + file)
    data_df = data_df.rename(columns={'Unnamed: 0': 'Cos'})
    data_df.insert(1, 'Sin', data_df['Cos'])
    data_df['Cos'] = np.cos(data_df['Cos'] / 100 * np.pi)
    data_df['Sin'] = np.sin(data_df['Sin'] / 100 * np.pi)
    # print(data_df.head())
    DF.append(data_df)

data_mean = 106.03646163873081
data_std = 151.59025900983642

for df in DF:
    data_np = df.iloc[:, :-1].to_numpy()
    data_np[:, 2:] = normalization(data_np[:, 2:], mean=data_mean, std=data_std)
    for i in range(data_np.shape[0] - 15):
        Data5Sec.append(data_np[i: i + 5, : ])
        GroundTrue.append(
            data_np[i + 15, [(2 + j * 4, 2 + j * 4 + 1) for j in range(20)]])
Data5Sec = np.array(Data5Sec)
GroundTrue = np.array(GroundTrue)
GroundTrue = GroundTrue.reshape([GroundTrue.shape[0], 40])

X_train, X_test, y_train, y_test = train_test_split(
    Data5Sec, GroundTrue, test_size=.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=.2, random_state=42)

model = load_model('dataset_new/model.h5')

model.summary()

checkpoint_path = 'dataset_new/best_model.h5'
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
