# Aprendizaje Automático 2023 - 2024

# Práctica 2 - Ejercicio 1

# Julia López Augusto
# Gonzalo Vega Pérez

# Importa las bibliotecas necesarias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Establecer una semilla para la aleatoriedad en scikit-learn
random.seed(42)
np.random.seed(42)

# Datos de entrenamiento para regresión logística con etiquetas (clase)
data = pd.DataFrame({
    'x1': [0.89, 0.41, 0.04, 0.75, 0.15, 0.14, 0.61, 0.25, 0.32, 0.40, 1.26, 1.68, 1.23, 1.46, 1.38, 1.54, 1.99, 1.76, 1.98, 1.23],
    'x2': [0.41, 0.39, 0.61, 0.17, 0.19, 0.09, 0.32, 0.77, 0.23, 0.74, 1.53, 1.05, 1.76, 1.60, 1.86, 1.99, 1.93, 1.41, 1.00, 1.54],
    'x3': [0.69, 0.82, 0.83, 0.29, 0.31, 0.52, 0.33, 0.83, 0.81, 0.56, 1.21, 1.22, 1.33, 1.10, 1.75, 1.75, 1.54, 1.34, 1.83, 1.55],
     'y': ['+', '+', 'O', '+', 'O', '+', '+', '+', '+', '+', 'O', 'O', 'O', 'O', '+', 'O', '+', 'O', 'O', 'O']
})

# Separar entradas (X) con la salida (y)
X = data[['x1', 'x2', 'x3']]
y = data['y']

# Crear un modelo de árbol de decisión
clf = DecisionTreeClassifier()

# Entrenar el modelo
clf.fit(X, y)

# Obtener las características más importantes
importances = clf.feature_importances_

# Obtener el nombre de las características
feature_names = X.columns

# Imprimir la importancia de cada característica
for feature, importance in zip(feature_names, importances):
    print(f'{feature}: {importance:.2f}')

# Crear una imagen del árbol de decisión
plt.figure(figsize=(10, 6))
tree.plot_tree(clf, filled=True, feature_names=X.columns)
plt.show()
