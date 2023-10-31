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

  # Crear el modelo de árbol de decisión
  md1 = DecisionTreeClassifier().fit(data_train_encoded[['PREVISIÓN_LLUVIOSO', 'PREVISIÓN_NUBLADO', 'PREVISIÓN_SOLEADO',
                                                           'TEMPERATURA_ALTA', 'TEMPERATURA_BAJA', 'TEMPERATURA_MEDIA',
                                                           'MAREA_ALTA', 'MAREA_BAJA', 'MAREA_MEDIA',
                                                           'VIENTO_DEBIL', 'VIENTO_FUERTE', 'VIENTO_MEDIO']], data_train_encoded['PESCAR_SI'])


# Utiliza el modelo para hacer predicciones en los datos de entrenamiento
X_evaluacion = data_train_encoded.drop(columns=['PESCAR_SI', 'PESCAR_NO'])
y_real = data_train_encoded['PESCAR_SI']

# Utilizar el modelo para hacer predicciones en los nuevos datos
y_pred = md1.predict(X_evaluacion)

# Compara las predicciones con las etiquetas reales
accuracy = accuracy_score(y_real, y_pred)

print(accuracy)
    #error_percentage = 100 * (1 - accuracy_score(Y, Ye))
    #print("Error en porcentaje sobre los datos de entrenamiento:", error_percentage, "%")


# Calcula el error en porcentaje
error_porcentaje = 100 * (1 - accuracy)

print(f'Error en porcentaje: {error_porcentaje:.2f}%')