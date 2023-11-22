# Aprendizaje Automático 2023 - 2024

# Práctica 4 - Ejercicio 3

# Julia López Augusto
# Gonzalo Vega Pérez

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import OneHotEncoder

# 1. Preparar datos
# 2. Preparar Red
# 3. Preparar criterios (función de pérdida)
# 4. Preparar optimizador
# 5. Entrenar
# 6. Validar

class myDatasett(Dataset):
  def __init__(self, x, y):
    self.x = torch.tensor(x)
    self.y = torch.tensor(y)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx,:], self.y[idx]

class Net(torch.nn.Module):
  # Red neuronal con función de activación ReLU en la capa oculta.
  # Y función de activación softmax en la capa de salida.
  def __init__(self):
    super().__init__()

    self.hidden1 = nn.Linear(4, 30)
    self.relu1 = nn.ReLU()
    self.hidden2 = nn.Linear(30, 20)
    self.relu2 = nn.ReLU()
    self.hidden3 = nn.Linear(20, 10)
    self.relu3 = nn.ReLU()
    self.output = nn.Linear(10,3) # Tres salidas porque hay tres tipos de iris

  # Definir como se utilizan los módulos
  def forward(self, x):
    x = self.relu1(self.hidden1(x)) # Pasa por la capa oculta, luego por la no linearidad (RelU)
    x = self.relu2(self.hidden2(x)) # Pasa por la segunda capa oculta y su función de activación
    x = self.relu3(self.hidden3(x)) # Pasa por la tercera capa oculta y su función de activación
    x = self.output(x) # Por último pasa por la capa de salida
    return x


# 1. Preparar datos

dataframe = pd.read_csv("/content/drive/MyDrive/iris.csv")
print(dataframe.shape)
dataframe.head()

# Separar 80% de los datos para entrenamiento y 20% para validación
dataset_train, dataset_test = train_test_split(dataframe.values, test_size=0.2, random_state=42)

print("dataset_train shape: " + str(dataset_train.shape))
print("dataset_test shape: " + str(dataset_test.shape))

x_train = dataset_train[:,0:4]  # Entradas, primeras cuatro columnas
y_train = dataset_train[:,4:]    # Salida (última columna). Hay que poner los ":" después del cuatro para que la dimensionalidad sea correcta para el OHE
x_test = dataset_test[:,0:4]
y_test = dataset_test[:,4:]


ohe_ytrain = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y_train)
ohe_ytest = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y_test)

y_train_ohe = ohe_ytrain.transform(y_train)
y_test_ohe = ohe_ytrain.transform(y_test)

# Convertir el tipo de datos de los arrays de object a un tipo numérico como float-32.
# No se puede crear un tensor de torch a partir de datos no numéricos
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Crear los datasets
train_dataset = myDatasett(x_train, y_train_ohe)
test_dataset = myDatasett(x_test, y_test_ohe)

# Crear los Dataloaders (objeto que nos da batches del dataset)
# shuffle normalmente es True para que en cada epoch los elementos que van en cojunto dentro del bach no sean siempre los mismos
batch_size = 5 # Número de elementos del dataset que contendrá cada batch
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = False)
train_dataloader2 = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle = False) # batch de tamaño del conjunto de datos entero para validar (Cómo todos los datos están en el batch, habrá solo un epoch)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

# Crear Red Neuronal
net = Net()

print(net)


# CRITERIOS DE ENTRENAMIENTO

# Función de pérdida
criterion = nn.CrossEntropyLoss() # Función de pérdida CrossEntropyLoss. Tiene en cuenta la función de activación Softmax. No hace falta añadirla a la capa de salida de la red.

# Optimizador
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001) # lr  = learning rate (tasa de aprendizaje)

# ENTRENAMIENTO

net.train()
num_epochs = 200

for epoch in range(num_epochs):
  for x, y in train_dataloader:
    #.#.#.#.#.# forward #.#.#.#.#.#

    x = x.type(torch.float32) # . Le pido el bach al dataloader. x = datos de entrada. La red se crea en float32 porque hay menos consumo de memoria, pero los elementos vienen en float64. Por eso hay que convertirlos
    y = y.type(torch.float32) # Le pido el bach de datos de salida al dataloader
    y_hat = net(x) # le paso el bach de entrada a la red neuronal y me da la predicción.

    loss = criterion(y_hat, y) # comparo lo que me sale con lo que me debería salir. Calculo la pérdida

    #.#.#.#.#.# backward #.#.#.#.#.#

    # El optimizer hace el back propagation, actualiza los pesos en base a las pérdidas que has tenido
    optimizer.zero_grad() # Reiniciar el optimizador, poner todo a cero para que no de error. Cada vez que se utiliza el optimizer hay que reiniciarlo.
    loss.backward() # Calcular pérdidas
    optimizer.step() # Optimizar pesos

  # log del epoch

  # Si se está entrenando bien, cada epoch las pérdidas van bajando.
  print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))



# Evaluación (DESPUÉS DEL ENTRENAMIENTO)

net.eval()

# Hacemos lo mismo que en el for, pero no hace falta hacer el for porque solo tenemos un batch. El batch es del tamaño del conjunto de datos enetero. Cogemos todos los elementos a la vez

# Datos de entrenamiento
x_train_torch, y_train_torch = next(iter(train_dataloader2))
x_train_torch = x_train_torch.type(torch.float32)
y_train_torch = y_train_torch.type(torch.float32)

y_hat_train_torch = net(x_train_torch)
cross_entropy = criterion(y_hat_train_torch, y_train_torch).item()
accuracy = (torch.argmax(y_hat_train_torch, 1) == torch.argmax(y_train_torch, 1)).float().mean()

print('Train CrossEntropyLoss: ' + str(cross_entropy) + ' accuracy: ' + str(accuracy*100) + '%')

# Datos de test
x_test_torch, y_test_torch = next(iter(test_dataloader))
x_test_torch = x_test_torch.type(torch.float32)
y_test_torch = y_test_torch.type(torch.float32)

y_hat_test_torch = net(x_test_torch)
cross_entropy = criterion(y_hat_test_torch, y_test_torch).item()
accuracy = (torch.argmax(y_hat_test_torch, 1) == torch.argmax(y_test_torch, 1)).float().mean()
print('Test CrossEntropyLoss: ' + str(cross_entropy) + ' accuracy: ' + str(accuracy*100) + '%')

# Si el modelo consigue resultados mucho mejores con los datos de entrenamiento que con los datos de test, es que hay overfitting.