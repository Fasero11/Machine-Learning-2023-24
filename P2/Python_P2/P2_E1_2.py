# Aprendizaje Automático 2023 - 2024

# Práctica 2 - Ejercicio 1

# Julia López Augusto
# Gonzalo Vega Pérez

# Importa las bibliotecas necesarias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import random

# Establecer una semilla para la aleatoriedad en scikit-learn
random.seed(42)
np.random.seed(42)

data_entrenamiento = {
    'PREVISIÓN': ['NUBLADO', 'LLUVIOSO', 'NUBLADO', 'LLUVIOSO', 'SOLEADO', 'LLUVIOSO', 'SOLEADO', 'SOLEADO', 'NUBLADO', 'SOLEADO'],
    'TEMPERATURA': ['MEDIA', 'ALTA', 'BAJA', 'MEDIA', 'ALTA', 'BAJA', 'BAJA', 'ALTA', 'BAJA', 'MEDIA'],
    'MAREA': ['BAJA', 'MEDIA', 'MEDIA', 'ALTA', 'ALTA', 'BAJA', 'ALTA', 'ALTA', 'MEDIA', 'BAJA'],
    'VIENTO': ['MEDIO', 'DEBIL', 'FUERTE', 'FUERTE', 'DEBIL', 'MEDIO', 'FUERTE', 'MEDIO', 'FUERTE', 'DEBIL'],
    'PESCAR': ['SI', 'SI', 'NO', 'SI', 'NO', 'SI', 'NO', 'NO', 'NO', 'SI']
}

# Crear un DataFrame con los datos de entrenamiento
df_entrenamiento = pd.DataFrame(data_entrenamiento)

# Transformar las variables categóricas en variables numéricas
df_entrenamiento = pd.get_dummies(df_entrenamiento, columns=['PREVISIÓN', 'TEMPERATURA', 'MAREA', 'VIENTO', 'PESCAR'])

# Crear las variables de entrada (X_entrenamiento) y la variable de salida (y_entrenamiento)
X_entrenamiento = df_entrenamiento.drop(columns=['PESCAR_SI', 'PESCAR_NO'])
y_entrenamiento = df_entrenamiento['PESCAR_SI']

# Crear y entrenar el modelo de árbol de decisión
clf = DecisionTreeClassifier()
clf.fit(X_entrenamiento, y_entrenamiento)

data_evaluacion = {
    'PREVISIÓN': ['SOLEADO', 'NUBLADO', 'NUBLADO', 'SOLEADO', 'LLUVIOSO', 'LLUVIOSO', 'NUBLADO', 'SOLEADO', 'NUBLADO', 'LLUVIOSO'],
    'TEMPERATURA': ['ALTA', 'BAJA', 'MEDIA', 'ALTA', 'MEDIA', 'MEDIA', 'BAJA', 'ALTA', 'MEDIA', 'BAJA'],
    'MAREA': ['MEDIA', 'MEDIA', 'BAJA', 'ALTA', 'BAJA', 'ALTA', 'MEDIA', 'ALTA', 'BAJA', 'MEDIA'],
    'VIENTO': ['DEBIL', 'MEDIO', 'MEDIO', 'FUERTE', 'MEDIO', 'DEBIL', 'MEDIO', 'DEBIL', 'MEDIO', 'FUERTE'],
    'PESCAR': ['NO', 'NO', 'SI', 'NO', 'SI', 'SI', 'SI', 'NO', 'SI', 'SI']
  }

# Crear un DataFrame con los datos de evaluación
df_evaluacion = pd.DataFrame(data_evaluacion)

# Transformar las variables categóricas en variables numéricas
df_evaluacion = pd.get_dummies(df_evaluacion, columns=['PREVISIÓN', 'TEMPERATURA', 'MAREA', 'VIENTO', 'PESCAR'])

# Crear las variables de entrada (X_evaluacion) y la variable de salida real (y_real)
X_evaluacion = df_evaluacion.drop(columns=['PESCAR_SI', 'PESCAR_NO'])
y_real = df_evaluacion['PESCAR_SI']

# Utilizar el modelo para hacer predicciones en los nuevos datos
y_pred = clf.predict(X_evaluacion)

# Calcular el porcentaje de acierto
porcentaje_acierto = accuracy_score(y_real, y_pred) * 100

print(f'Porcentaje de acierto en los nuevos datos: {porcentaje_acierto:.2f}%')

