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

  # Establecer una semilla para la aleatoriedad en scikit-learn
  random.seed(42)
  np.random.seed(42)
  # Crear el modelo de árbol de decisión
  md1 = DecisionTreeClassifier().fit(data_train_encoded[['PREVISIÓN_LLUVIOSO', 'PREVISIÓN_NUBLADO', 'PREVISIÓN_SOLEADO',
                                                           'TEMPERATURA_ALTA', 'TEMPERATURA_BAJA', 'TEMPERATURA_MEDIA',
                                                           'MAREA_ALTA', 'MAREA_BAJA', 'MAREA_MEDIA',
                                                           'VIENTO_DEBIL', 'VIENTO_FUERTE', 'VIENTO_MEDIO']], data_train_encoded['PESCAR_SI'])


  # a) Para ver qué característica se ha seleccionado como nodo raíz y su importancia, puedes utilizar:
  root_feature = data_train_encoded.columns[md1.tree_.feature[0]]
  print(f"La característica seleccionada como nodo raíz es:" + root_feature)


  # Obtener las importancias de las características

  #root_feature = X.columns[model.tree_.feature[0]]
  #print(f"La característica seleccionada como nodo raíz es: " + root_feature)
  
  #rint("Características y sus importancias:")
  #feature_importances = dict(zip(X.columns, model.feature_importances_))
  #print(feature_importances)

  # b) Para verificar si existe algún valor de alguna característica que permita pescar independientemente de los demás valores, debes examinar el árbol. Esto dependerá del árbol específico generado.

  # c) Si existe algún atributo que no influye en la decisión, también dependerá del árbol generado y la estructura de tus datos.

  # d) Para adjuntar de forma gráfica el árbol de decisión:
  #plt.figure(figsize=(15, 10))
  #plot_tree(model, feature_names=X.columns, class_names=['No', 'Sí'], filled=True, rounded=True)
  #plt.show()

