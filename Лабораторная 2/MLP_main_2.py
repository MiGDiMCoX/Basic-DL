# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from neural_2 import MLP
from sklearn.preprocessing import LabelBinarizer

df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]

#выбираем 5 столбец из 100 строк
y = df.iloc[0:100, 4].values

#преобразуем строковые значения в бинарные массивы 
#[[1, 0, 0],[0, 1, 0],[0, 0, 1]]
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

# возьмем четыре признака
X = df.iloc[0:100, [0, 1, 2, 3]].values

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 3 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи

iterations = 50
learning_rate = 0.1

net = MLP(inputSize, outputSize, learning_rate, hiddenSizes)

# обучаем сеть (фактически сеть это вектор весов weights)
for i in range(iterations):
    net.train(X, y)

    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(y - net.predict(X)))))

# считаем ошибку на обучающей выборке
pr = net.predict(X)
print(sum(abs(y-(pr>0.5))))