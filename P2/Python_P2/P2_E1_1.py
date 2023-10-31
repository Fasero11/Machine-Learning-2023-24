# Aprendizaje Automático 2023 - 2024
# Práctica 2 - Ejercicio 1

# Julia López Augusto
# Gonzalo Vega Pérez

# Importa las bibliotecas necesarias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':

  # Establecer una semilla para la aleatoriedad en scikit-learn
  random.seed(42)
  np.random.seed(42)
  
  data = {
    'PREVISIÓN': ['NUBLADO', 'LLUVIOSO', 'NUBLADO', 'LLUVIOSO', 'SOLEADO', 'LLUVIOSO', 'SOLEADO', 'SOLEADO', 'NUBLADO', 'SOLEADO'],
    'TEMPERATURA': ['MEDIA', 'ALTA', 'BAJA', 'MEDIA', 'ALTA', 'BAJA', 'BAJA', 'ALTA', 'BAJA', 'MEDIA'],
    'MAREA': ['BAJA', 'MEDIA', 'MEDIA', 'ALTA', 'ALTA', 'BAJA', 'ALTA', 'ALTA', 'MEDIA', 'BAJA'],
    'VIENTO': ['MEDIO', 'DEBIL', 'FUERTE', 'FUERTE', 'DEBIL', 'MEDIO', 'FUERTE', 'MEDIO', 'FUERTE', 'DEBIL'],
    'PESCAR': ['SI', 'SI', 'NO', 'SI', 'NO', 'SI', 'NO', 'NO', 'NO', 'SI']
  }

  #construir una tabla con los datos de entrada
  data_train = pd.DataFrame(data)

  # Transformar variables categóricas a numéricas: One hot encoding 
  data_train_encoded = pd.get_dummies(data_train, columns=['PREVISIÓN', 'TEMPERATURA', 'MAREA', 'VIENTO', 'PESCAR'])

  # nos quedamos con las entradas
  entradas = data_train_encoded.drop(columns=['PESCAR_SI', 'PESCAR_NO'])
  salida = data_train_encoded['PESCAR_SI']

  # Crear el modelo de árbol de decisión
  md1 = DecisionTreeClassifier().fit(entradas, salida)

  # a) Para ver qué característica se ha seleccionado como nodo raíz y su importancia, puedes utilizar:
  root_feature = data_train_encoded.columns[md1.tree_.feature[0]]
  print("La característica seleccionada como nodo raíz es:" + root_feature)

  # DEMOSTRACIÓN DEL APARTADO A)
  # Obtener la importancia de las características
  importance = md1.feature_importances_

  # Obtener las características utilizadas como nodos en el árbol
  feature_names = list(entradas.columns)
  # Imprimir la importancia de las características
  for feature, imp in zip(feature_names, importance):
    print(f'{feature}: {imp}')
  

  # d) Para adjuntar de forma gráfica el árbol de decisión:
  plt.figure(figsize=(15, 10))
  plot_tree(md1, filled=True, feature_names=['PREVISIÓN_LLUVIOSO', 'PREVISIÓN_NUBLADO', 'PREVISIÓN_SOLEADO',
                                                           'TEMPERATURA_ALTA', 'TEMPERATURA_BAJA', 'TEMPERATURA_MEDIA',
                                                           'MAREA_ALTA', 'MAREA_BAJA', 'MAREA_MEDIA',
                                                           'VIENTO_DEBIL', 'VIENTO_FUERTE', 'VIENTO_MEDIO'])
  plt.show()

