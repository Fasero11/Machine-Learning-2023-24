# Aprendizaje Automático 2023 - 2024
# Práctica 3 - Ejercicio 3 - Parte 1

# Julia López Augusto
# Gonzalo Vega Pérez
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D

# Función para graficar el conjunto de datos y el hiperplano de separación
def plot_hiperplano(X, Y, model, plot_title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Representar datos 
    scatter_class_minus = ax.scatter(X[Y == -1, 0], X[Y == -1, 1], X[Y == -1, 2], c='b', marker='o', label='Clase -1')
    scatter_class_plus = ax.scatter(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], c='r', marker='x', label='Clase 1')

    # Obtener los coeficientes y la intersección del hiperplano
    coef = model.coef_[0]
    intercept = model.intercept_
    # Crear una malla de puntos para el plano XY
    xx, yy = np.meshgrid(X[:, 0], X[:, 1])
    # Calcular los valores en el eje Z para el hiperplano 
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
    # Representar el hiperplano como una superficie
    surface = ax.plot_surface(xx, yy, zz, color='c', alpha=0.3)
    
    # Gestionar manualmente la leyenda
    handles = [scatter_class_minus, scatter_class_plus]
    labels = ['Clase -1', 'Clase 1']

    # Crear un proxy artista para el hiperplano de separación (superficie)
    proxy = plt.Rectangle((0, 0), 1, 1, fc='c', alpha=0.3)
    handles.append(proxy)
    ax.legend(handles, labels)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title(plot_title)
    
    plt.show()

# Función para graficar el conjunto de datos
def plot_dataset(X, Y, plot_title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Representa los datos 
    scatter_class_minus = ax.scatter(X[Y == -1, 0], X[Y == -1, 1], X[Y == -1, 2], c='b', marker='o', label='Clase -1')
    scatter_class_plus = ax.scatter(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], c='r', marker='x', label='Clase 1')
    
    # Gestionar manualmente la leyenda
    handles = [scatter_class_minus, scatter_class_plus]
    labels = ['Clase -1', 'Clase 1']
    ax.legend(handles, labels)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title(plot_title)
    
    plt.show()

if __name__ == '__main__':

    current_dir = os.getcwd()
    train_data = pd.read_csv(str(current_dir) + "/P3/Data_P3/trainingsetSVM1.csv")  

    x1 = train_data['X1']
    x2 = train_data['X2']
    x3 = train_data['X3']

    X = np.column_stack((x1, x2, x3))

    # 2 posibles casos de salida: -1 y 1
    Y = train_data['Y'] 

    ## Ejercicio 3.1
    plot_dataset(X, Y, "Conjunto de entrenamiento")

    ## Ejercicio 3.2
    # Modelo SVM con kernel lineal
    model = svm.SVC(kernel='linear')
    model.fit(X, Y)

    plot_hiperplano(X, Y, model, "Modelo SVM lineal + hiperplano")
