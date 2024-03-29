# Aprendizaje Automático 2023 - 2024

# Práctica 4 - Ejercicio 1 - versión Gonzalo

# Julia López Augusto
# Gonzalo Vega Pérez

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os

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
  # Definir módulos que tenemos
  def __init__(self):
    super().__init__()

    self.hidden1 = nn.Linear(8, 4)  # 8 entradas, 4 salidas. Capa oculta. 8 entradas porque tenemos 8 catacterísticas. # Los pesos están en float32
    self.relu1 = nn.ReLU()          # RelU (no linearidad). Función de activación

    # Capa oculta: Primero linearidad (hidden1), luego no linearidad (relu1) y luego pasa a la siguiente capa.
    # La salida en regresión siempre va a ser lineal
    # Clasificación binaria: Sigmoide
    # Clasificación no binaria: SoftMax

    self.output = nn.Linear(4,1)    # 4 entradas, 1 salida. Capa de Salida. 1 salida porque queremos 1 resultado (precio final de la casa). No aplicamos linearidad en regresión porque restringe el tamaño de salida.

  # Definir como se utilizan los módulos
  def forward(self, x):
    x = self.relu1(self.hidden1(x)) # Pasa por la capa oculta, luego por la no linearidad (RelU)
    x = self.output(x) # Por último pasa por la capa de salida
    return x


# 1. Preparar datos

#dataframe = pd.read_csv("/content/drive/MyDrive/housing.csv")
current_dir = os.getcwd()
dataframe = pd.read_csv(str(current_dir) + "/P4/Data_P4/housing.csv")

print(dataframe.shape)
dataframe.head()

# Separar 80% de los datos para entrenamiento y 20% para validación
dataset_train, dataset_test = train_test_split(dataframe.values, test_size=0.2, random_state=42)

print("dataset_train shape: " + str(dataset_train.shape))
print("dataset_test shape: " + str(dataset_test.shape))

x_train = dataset_train[:,0:8]  # Entradas, primeras ocho columnas
y_train = dataset_train[:,8]    # Salida (última columna)
x_test = dataset_test[:,0:8]
y_test = dataset_test[:,8]

# Crear los datasets
train_dataset = myDatasett(x_train, y_train)
test_dataset = myDatasett(x_test, y_test)




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
criterion = nn.MSELoss() # mean score error. Si le hago la raiz cuadrado obtengo el número natural que se entiende lo que es el error.

# Se puede usar el root mean squared error error para que te de el error en números que se entiendan
# Por ejemplo si la salida son euros, el root mean squared error te da el número de euros de diferencia entre la predicción y la salida real.
# Bueno para poder visualizar si el modelo funciona o no.

# No se usa el root mean squared error en el entrenamiento porque añade mucha complejidad, pero se puede usar en la validación.

# Optimizador
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001) # lr  = learning rate (tasa de aprendizaje)

# ENTRENAMIENTO

net.train()
num_epochs = 100

for epoch in range(num_epochs):
  for x, y in train_dataloader:
    #.#.#.#.#.# forward #.#.#.#.#.#

    x = x.type(torch.float32) # . Le pido el bach al dataloader. x = datos de entrada. La red se crea en float32 porque hay menos consumo de memoria, pero los elementos vienen en float64. Por eso hay que convertirlos
    y = y.type(torch.float32) # Le pido el bach de datos de salida al dataloader
    y_hat = net(x) # le paso el bach de entrada a la red neuronal y me da la predicción.

    # En numpy se puede definir un array de dimensión 2 pero con sola una fila
    # Es igual que un array de una dimensión pero se accede con dos coordenadas
    # a[1] -> Segundo elemento del array de 1 dimensión
    # a[0, 1] -> Segundo elemento del array de 2 dimensiones
    # a tiene una sola fila en ambos casos

    loss = criterion(y_hat[:,0], y) # comparo lo que me sale con lo que me debería salir. Calculo la pérdida

    #.#.#.#.#.# backward #.#.#.#.#.#

    # El optimizer hace el back propagation, actualiza los pesos en base a las pérdidas que has tenido
    optimizer.zero_grad() # Reiniciar el optimizador, poner todo a cero para que no de error. Cada vez que se utiliza el optimizer hay que reiniciarlo.
    loss.backward() # Calcular pérdidas
    optimizer.step() # Optimizar pesos

  # log del epoch

  # Si se está entrenando bien, cada epoch las pérdidas van bajando.
  print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))


  # Evaluation (DESPUÉS DEL ENTRENAMIENTO)

net.eval()

# Train

# Hacemos lo mismo que en el for, pero no hace falta hacer el for porque solo tenemos un batch. El batch es del tamaño del conjunto de datos enetero. Cogemos todos los elementos a la vez
x_train_torch, y_train_torch = next(iter(train_dataloader2))
x_train_torch = x_train_torch.type(torch.float32)
y_train_torch = y_train_torch.type(torch.float32)
y_hat_train_torch = net(x_train_torch)
loss_train = criterion(y_hat_train_torch[:,0], y_train_torch)
print('Train MSE: ' + str(loss_train.item()))

# Test
x_test_torch, y_test_torch = next(iter(test_dataloader))
x_test_torch = x_test_torch.type(torch.float32)
y_test_torch = y_test_torch.type(torch.float32)
y_hat_test_torch = net(x_test_torch)
loss_test = criterion(y_hat_test_torch[:,0], y_test_torch)
print('Test MSE: ' + str(loss_test.item()))

# EXTRA (para el ejercicio 2)

# La función de pérdida compara lo que está saliendo con lo que tendría que salir. Si el resultado es grande es que se está equivocando. Si es pequeño es que está bastante acertado
# Accuracy score compara el primer parámetro con el segundo.

# Cosa rara de PYTORCH
#
# Cuando es multiclase, se utiliza la función de perdida CrossEntropyLoss().
# En este caso la no linearidad debería ser una Softmax que se indicaría en la capa de salida del objeto RedNeuronal.
# Pero si se pone en la red neuronal tal cual, da problemas de inestabilidad en PyTorch.
# Entonces se quita de ahí y la no linearidad se pone en la función de pérdida.