# Aprendizaje Automático 2023 - 2024

# Práctica 1 - Ejercicio 1

# Julia López Augusto
# Gonalo Vega Pérez

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Datos Iniciales
    x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    X = np.column_stack((x1, x2))
    Y = np.array([1.56, 1.95, 2.44, 3.05, 3.81, 4.77, 5.96, 7.45, 9.31, 11.64])


    # Crear Modelo y entrenarlo
    mdl = LinearRegression().fit(X,Y)

    # Resultados (Pesos)
    coef = mdl.coef_
    intercept = mdl.intercept_

    print("Coef:" + str(coef))
    print("Intercept:" + str(intercept))

    # Utilizar modelo entrenado para predecir resultados
    Ye = mdl.predict(X)

    # Representar graficamente
    fig = plt.figure
    ax = plt.axes(projection='3d')

    # Mostrar datos iniciales
    ax.plot3D(x1, x2, Y) 
    # Mostrar predicción
    ax.plot3D(x1, x2, Ye, 'red') 

    # Nombrar Ejes
    ax.set_xlabel('x1', labelpad=20)
    ax.set_ylabel('x2', labelpad=20)
    ax.set_zlabel('y', labelpad=20)

    plt.show()