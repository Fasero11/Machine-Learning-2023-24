# Aprendizaje Automático 2023 - 2024

# Práctica 4 - Ejercicio 1 - versión Julia

# Julia López Augusto
# Gonzalo Vega Pérez


import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os


current_dir = os.getcwd()
dataframe = pd.read_csv(str(current_dir) + "/P4/Data_P4/housing.csv")
print(dataframe.shape)
# muestra los primeros 5 elementos
dataframe.head()


dataset_train, dataset_test = train_test_split(dataframe.values, test_size= 0.2, random_state =42)

X_train = dataset_train[:, 0:8]
y_train = dataset_train[:,8]
X_test = dataset_test[:, 0:8] # de 0 a 7
y_test = dataset_test[:, 8]


class MyDatasett(Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X)
    self.y = torch.tensor(y)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx,:], self.y[idx]

#Datasets

train_dataset = MyDatasett(X_train, y_train)
test_dataset = MyDatasett(X_test, y_test)

#Dataloaders: nos da el tamaño de batch que queremos: según cada bloque vaya saliendo
#hacemos varios
#shuffle es para la aleatoriedad e inicialmente lo ponemos a false para que la depuración sea fácil
#cuando ya esté depurado ponemos shuffle a True
batch_size = 5
train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=False)
train_dataloader2 = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden1 = nn.Linear(8,4)
    self.relu1 = nn.ReLU()
    #es la salida: en regresión  es lineal, en clasificación no: binaria -> sigmoide y no binaria -> softmax
    self.output = nn.Linear(4,1)

  def forward(self,x):
    x = self.relu1(self.hidden1(x))
    x = self.output(x)
    return x

# Crea un modelo
net = Net()


# función de pérdida
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# Training time!
net.train()
num_epochs = 100
for epoch in range(num_epochs):
  for X, y in train_dataloader:

    #la red neuronal normalmente se usa float 32
    # estamos usando float 64 y hay que convertirlo
    X = X.type(torch.float32)
    y = y.type(torch.float32)
    y_hat = net(X)
    #comparo lo que me sale con lo que me tiene que salir
    # Tienen dimensión 1
    # a[1]   : segundo elem de array de 1 dim
    # a[0,1] : segundo elem de array de 1 dim
    loss = criterion(y_hat[:,0], y)

    # reinicia el optimizer
    optimizer.zero_grad()
    # hacia atrás
    loss.backward()
    # actualiza los pesos : ADAM
    optimizer.step()
  print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

#tenemos que ver que vaya bajando el error ya que si no es así algo está sucediendo mal


#ya la red neuronal está entrenada y la teneos que evaluar

net.eval()

#Train

X_train_torch, y_train_torch = next(iter(train_dataloader2))
X_train_torch = X_train_torch.type(torch.float32)
y_train_torch = y_train_torch.type(torch.float32)
y_hat_train_torch = net(X_train_torch)
loss_train = criterion(y_hat_train_torch[:,0], y_train_torch)
print('Train MSE: '+ str(loss_train.item()))

# Test
X_test_torch, y_test_torch = next(iter(test_dataloader))
X_test_torch = X_test_torch.type(torch.float32)
y_test_torch = y_test_torch.type(torch.float32)
y_hat_test_torch = net(X_test_torch)
loss_test = criterion(y_hat_test_torch[:,0], y_test_torch)
print('Test MSE: '+ str(loss_test.item()))

#  cuando hay overfitting el de test es mucho mayor que el del entrnamiento