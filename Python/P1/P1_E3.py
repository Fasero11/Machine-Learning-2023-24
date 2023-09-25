# Aprendizaje Automático 2023 - 2024

# Práctica 1 - Ejercicio 3

# Julia López Augusto
# Gonalo Vega Pérez

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Datos Iniciales
    x1 = [0.89, 0.41, 0.04, 0.75, 0.15, 0.14, 0.61, 0.25, 0.32, 0.40, 1.26,
           1.68, 1.23, 1.46, 1.38, 1.54, 1.99, 1.76, 1.98, 1.23]
    x2 = [0.41, 0.39, 0.61, 0.17, 0.19, 0.09, 0.32, 0.77, 0.23, 0.74, 1.53,
           1.05, 1.76, 1.60, 1.86, 1.99, 1.93, 1.41, 1.00, 1.54]
    x3 = [0.69, 0.82, 0.83, 0.29, 0.31, 0.52, 0.33, 0.83, 0.81, 0.56, 1.21,
          1.22, 1.33, 1.10, 1.75, 1.75, 1.54, 1.34, 1.83, 1.55]
    X = np.column_stack((x1, x2, x3))
    # '+' = 1
    #  O = 0
    Y = np.array([1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,0])


    # Crear Modelo y entrenarlo
    mdl = LogisticRegression(penalty="none").fit(X,Y)

    # Utilizar modelo entrenado para predecir resultados
    Ye = mdl.predict(X)
    Ye_proba = mdl.predict_proba(X)

    # Representar graficamente
    fig = plt.figure
    ax = plt.axes(projection='3d')

    data_num = 0
    for data in X:
        # Si el dato es una cruz
        if Y[data_num] == 1:
            print("plot cross " + str(data_num))
            # Si la predicción es cruz (acierto): pintar la cruz azul
            if Ye[data_num] == 1:
                ax.scatter3D(X[data_num][0], X[data_num][1], X[data_num][2], marker="X", color="blue")
            # Si la predicción es círculo (error): pintar la cruz roja
            else:
                ax.scatter3D(X[data_num][0], X[data_num][1], X[data_num][2], marker="X", color="red") 
        # Si el dato es un círculo
        else:
            print("plot circle " + str(data_num))
            # Si la predicción es círculo (acierto): pintar el círculo azul
            if Ye[data_num] == 0:
                ax.scatter3D(X[data_num][0], X[data_num][1], X[data_num][2], color="blue")
            # Si la predicción es cruz (error): pintar el círculo rojo
            else:
                ax.scatter3D(X[data_num][0], X[data_num][1], X[data_num][2], color="red")

        data_num += 1

    # Nombrar Ejes
    ax.set_xlabel('x1', labelpad=20)
    ax.set_ylabel('x2', labelpad=20)
    ax.set_zlabel('x3', labelpad=20)

    plt.show()