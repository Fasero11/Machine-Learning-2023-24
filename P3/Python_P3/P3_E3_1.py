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

# Función para graficar el conjunto de datos con contornos y el hiperplano de separación
def plot_hiperplano(X, Y, model, plot_title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Representar contornos como en el ejemplo anterior
    scatter_class_minus = ax.scatter(X[Y == -1, 0], X[Y == -1, 1], X[Y == -1, 2], c='b', marker='o', label='Clase -1')
    scatter_class_plus = ax.scatter(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], c='r', marker='x', label='Clase 1')
    
    # Representar hiperplano de separación óptimo
    coef = model.coef_[0]
    intercept = model.intercept_
    xx, yy = np.meshgrid(X[:, 0], X[:, 1])
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
    
    surface = ax.plot_surface(xx, yy, zz, color='c', alpha=0.3)
    
    # Gestionar manualmente la leyenda
    handles = [scatter_class_minus, scatter_class_plus]
    labels = ['Clase -1', 'Clase 1']

    # Crear un proxy artista para el hiperplano de separación (superficie)
    proxy = plt.Rectangle((0, 0), 1, 1, fc='c', alpha=0.3)
    handles.append(proxy)
    labels.append('Hiperplano de separación')

    ax.legend(handles, labels)

    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title(plot_title)
    
    plt.show()

# Función para graficar el conjunto de datos
"""
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
"""
def plot_dataset(X, Y, plot_title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Representar contornos como en el ejemplo anterior
    scatter_class_minus = ax.scatter(X[Y == -1, 0], X[Y == -1, 1], X[Y == -1, 2], c='b', marker='o', label='Clase -1')
    scatter_class_plus = ax.scatter(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], c='r', marker='x', label='Clase 1')
    
    # Representar hiperplano de separación óptimo
    #coef = model.coef_[0]
    #intercept = model.intercept_
    #xx, yy = np.meshgrid(X[:, 0], X[:, 1])
    #zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
    
    #surface = ax.plot_surface(xx, yy, zz, color='c', alpha=0.3)
    
    # Gestionar manualmente la leyenda
    handles = [scatter_class_minus, scatter_class_plus]
    labels = ['Clase -1', 'Clase 1']

    # Crear un proxy artista para el hiperplano de separación (superficie)
    #proxy = plt.Rectangle((0, 0), 1, 1, fc='c', alpha=0.3)
    #handles.append(proxy)
    labels.append('Hiperplano de separación')

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
    # Tiene forma de hiperboloide invertida

    ## Ejercicio 3.2
    # Modelo SVM con kernel lineal
    model = svm.SVC(kernel='linear')
    model.fit(X, Y)

    # Graficar modelo SVM con contornos y hiperplano de separación
    plot_hiperplano(X, Y, model, "Modelo SVM lineal + hiperplano")
