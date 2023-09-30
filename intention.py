"""
__title__    = intention.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2023/4/1 14:08
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
import random
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Attention, Flatten
from New_AR_func import normalization
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

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

# Define the search space
space = {
    'n_lstm': [64, 128, 256],
    'n_dense1': [64, 128, 256],
    'n_dense2': [64, 128, 256],
    'learning_rate': [0.001, 0.01, 0.1]
}


# Define the fitness function
def fitness_function(individual):
    # Build the route_model
    inputs = Input(shape=(1, 40))
    lstm = LSTM(units=individual['n_lstm'], return_sequences=True)(inputs)
    dense1 = Dense(units=individual['n_dense1'], activation='relu')(lstm)
    dense2 = Dense(units=individual['n_dense2'], activation='relu')(dense1)
    outputs = Dense(units=3, activation='softmax')(dense2)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Train the route_model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the fitness
    val_loss = model.evaluate(X_test, y_test, verbose=0)
    return val_loss  # Minimize the loss


# Initialize the population
population_size = 10
population = []
for i in range(population_size):
    individual = {}
    individual['n_lstm'] = random.choice(space['n_lstm'])
    individual['n_dense1'] = random.choice(space['n_dense1'])
    individual['n_dense2'] = random.choice(space['n_dense2'])
    individual['learning_rate'] = random.choice(space['learning_rate'])
    population.append(individual)

# GA loop
num_generations = 5
for generation in range(num_generations):
    # Evaluate the fitness of each individual
    fitness_scores = []
    print('Generation', generation + 1, ':')
    for individual in population:
        fitness_scores.append(fitness_function(individual))
    print('Generation', generation + 1, '- Best Fitness:', max(fitness_scores))

    # Select parents
    parents = []
    for i in range(2):
        parent_index = np.random.choice(range(population_size), size=5, replace=False,
                                        p=np.array(fitness_scores) / sum(fitness_scores))
        parent = population[max(parent_index, key=lambda x: fitness_scores[x])]
        parents.append(parent)

    # Crossover and mutation
    offspring = []
    for i in range(population_size):
        child = {}
        for key in space.keys():
            if np.random.rand() < 0.5:
                child[key] = parents[0][key]
            else:
                child[key] = parents[1][key]
            if np.random.rand() < 0.1:
                child[key] = random.choice(space[key])
        offspring.append(child)

    population = offspring

# Evaluate the final route_model
best_individual = max(population, key=lambda x: fitness_function(x))
inputs = Input(shape=(1, 40))
lstm = LSTM(units=best_individual['n_lstm'], return_sequences=True)(inputs)
dense1 = Dense(units=best_individual['n_dense1'], activation='relu')(lstm)
dense2 = Dense(units=best_individual['n_dense2'], activation='relu')(dense1)
outputs = Dense(units=3, activation='softmax')(dense2)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print('Final Test Loss:', test_loss)
print('Best Individual:', best_individual)