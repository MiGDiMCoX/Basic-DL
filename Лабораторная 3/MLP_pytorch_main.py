import pandas as pd
import numpy as np
from neural import MLPptorch
from neural import MLPptorch_ReLU
import torch
import torch.nn as nn

# функция обучения
def train(net, optimizer, x, y, num_iter):
    for i in range(0,num_iter):
        optimizer.zero_grad() # Обнуление градиентов
        pred = net.forward(x)
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()
        if i%(num_iter/10)==0:
           print('Ошибка на ' + str(i) + ' итерации: ', loss.item())
    return loss.item()


df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]

X = df.iloc[0:100, 0:3].values
y = df.iloc[0:100, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor':3}).values.reshape(-1,1)
Y = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y[:,i-1] = np.where(y == i, 1, 0).reshape(1,-1)

X_test = df.iloc[100:150, 0:3].values
y = df.iloc[100:150, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor':3}).values.reshape(-1,1)
Y_test = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y_test[:,i-1] = np.where(y == i, 1, 0).reshape(1,-1)


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = [50] # задаем число нейронов скрытого слоя 
outputSize = Y.shape[1] if len(Y.shape) else 1 # количество выходных сигналов равно количеству классов задачи

#lossFn = nn.MSELoss()
lossFn = nn.BCELoss()

net_relu = MLPptorch_ReLU(inputSize,hiddenSizes,outputSize)
optimizer_relu = torch.optim.SGD(net_relu.parameters(), lr=0.01)
loss_relu = train(net_relu, optimizer_relu, torch.from_numpy(X.astype(np.float32)), 
              torch.from_numpy(Y.astype(np.float32)), 50)

pred = net_relu.forward(torch.from_numpy(X.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y))
print("ReLU =", err)   

pred = net_relu.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y_test))
print("ReLU =", err)

hiddenSizes = 50

net_sigmoid = MLPptorch(inputSize,hiddenSizes,outputSize)
optimizer_sigmoid = torch.optim.SGD(net_sigmoid.parameters(), lr=0.05)
loss_sigmoid = train(net_sigmoid, optimizer_sigmoid, torch.from_numpy(X.astype(np.float32)), 
              torch.from_numpy(Y.astype(np.float32)), 50)

pred = net_sigmoid.forward(torch.from_numpy(X.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y))
print("Sigmoid =", err)   

pred = net_sigmoid.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y_test))
print("Sigmoid =", err)