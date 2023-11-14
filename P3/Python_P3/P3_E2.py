# Aprendizaje Automático 2023 - 2024

# Práctica 3 - Ejercicio 2

# Julia López Augusto
# Gonalo Vega Pérez

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':

    current_dir = os.getcwd()
    train_data = pd.read_csv(str(current_dir) + "/P3/Data_P3/trainingsetPCA.csv")  

    x1 = train_data['X1']
    x2 = train_data['X2']
    x3 = train_data['X3']
    x4 = train_data['X4']
    x5 = train_data['X5']
    x6 = train_data['X6']
    x7 = train_data['X7']
    x8 = train_data['X8']
    x9 = train_data['X9']
    x10 = train_data['X10']

    X = np.column_stack((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10))

    # Normalizamos el set de datos para mejorar el funcionamiento de PCA
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    keep_components = X.shape[1]
    info_threshold = 0.9
    print("Umbral de información: " + str(info_threshold))

    retained_info = 1
    # Vamos quitando características hasta que el porcentaje de información retenida sea menor al umbral (info_threshold)
    while retained_info > info_threshold and keep_components > 0:
      pca = PCA(n_components=keep_components) # Indicamos el número de componentes que queremos quedarnos
      X_reduced = pca.fit_transform(X_normalized)

      # Calculamos el porcentaje de información que aporta cada componente
      variance_explained = pca.explained_variance_ratio_

      # Calculamos el porcentaje de información retenida en total (suma de los porcentajes de cada componente)
      cumulative_variance_explained = variance_explained.cumsum()
      retained_info = cumulative_variance_explained[-1] # El último elemento del vector que devuelve cumsum() es la suma de todos los anteriores

      print("Componentes eliminados: " + str(X.shape[1] - keep_components) + " | Información retenida: "+ str(retained_info))

      keep_components -= 1
