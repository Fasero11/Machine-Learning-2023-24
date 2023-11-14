# Aprendizaje Automático 2023 - 2024
# Práctica 2 - Ejercicio 1

# Julia López Augusto
# Gonzalo Vega Pérez
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D


# Función para graficar el conjunto de datos
def plot_dataset(X, Y, plot_title):
    # Matriz que contiene los colores de cada muestra
    colors = np.zeros((len(Y), 3))
    
    # Si para la muestra n, Y(n) es -1, colors(n) será [0, 0, 1]. Color azul
    # Si para la muestra n, Y(n) es 1, colors(n) será [1, 0, 0]. Color rojo
    colors_class1 = np.tile([0, 0, 1], (np.sum(Y == -1), 1))
    colors_class2 = np.tile([1, 0, 0], (np.sum(Y == 1), 1))
    
    # Cambiar el valor de color original [0,0,0] al valor calculado. Rojo o Azul.
    colors[Y == -1, :] = colors_class1
    colors[Y == 1, :] = colors_class2

    # Graficar el conjunto de datos con los contornos
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Usar plot en lugar de scatter para representar solo contornos
    ax.plot(X[Y == -1, 0], X[Y == -1, 1], X[Y == -1, 2], 'bo', label='Clase -1', markerfacecolor='none')
    ax.plot(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], 'ro', label='Clase 1', markerfacecolor='none')
    
    
    # Graficar el conjunto de datos con los nuevos colores.
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors)
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
    plot_dataset(X, Y, "trainingsetSVM1")
    # Tiene forma de hiperboloide invertida

    ## Ejercicio 3.2



