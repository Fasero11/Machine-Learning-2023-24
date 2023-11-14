# Aprendizaje Automático 2023 - 2024
# Práctica 3 - Ejercicio 3 - Parte 2

# Julia López Augusto
# Gonzalo Vega Pérez
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm

# Función para graficar el conjunto de datos
def plot_dataset(X, Y, plot_title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Representar contornos como en el ejemplo anterior
    scatter_class_minus = ax.scatter(X[Y == -1, 0], X[Y == -1, 1], X[Y == -1, 2], c='b', marker='o', label='Clase -1')
    scatter_class_plus = ax.scatter(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], c='r', marker='x', label='Clase 1')
    
    # Gestionar manualmente la leyenda
    handles = [scatter_class_minus, scatter_class_plus]
    labels = ['Clase -1', 'Clase 1']
    
    labels.append('Hiperplano de separación')

    ax.legend(handles, labels)

    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title(plot_title)
    
    plt.show()

if __name__ == '__main__':

    current_dir = os.getcwd()
    train_data = pd.read_csv(str(current_dir) + "/P3/Data_P3/trainingsetSVM2.csv")  

    x1 = train_data['X1']
    x2 = train_data['X2']
    x3 = train_data['X3']

    X = np.column_stack((x1, x2, x3))

    # 2 posibles casos de salida: -1 y 1
    Y = train_data['Y'] 


    # Plot the original dataset
    plot_dataset(X, Y, "trainingsetSVM2")

    # Model with a linear kernel
    linear_mdl = svm.SVC(kernel='linear')
    linear_mdl.fit(X, Y)
    ye_linear = linear_mdl.predict(X)

    # Plot predictions for the linear kernel
    plot_dataset(X, ye_linear, "Predictions trainingsetSVM1 (linear)")

    # Print error rate for the linear kernel
    error_rate_linear = zero_one_loss(Y, ye_linear)
    num_errors_linear = error_rate_linear * len(Y)
    print(f'Number of errors (linear): {num_errors_linear}. Error rate: {error_rate_linear:.3f}.\n\n')
    # el error es del 17,80% y no ea 0 porque no es linealmente separable 

    # 3.5 
    n = 1
    max_iterations = 10
    error_rate_poly = 1

    while error_rate_poly > 0 and n <= max_iterations:
        # Model with a polynomial kernel of degree n
        poly_mdl = svm.SVC(kernel='poly', degree=n, gamma=2)
        poly_mdl.fit(X, Y)
        ye_poly = poly_mdl.predict(X)

        # Calculate error rate and number of errors
        # calcula la tasa de error y el número de errores 
        error_rate_poly = zero_one_loss(Y, ye_poly)
        num_errors_poly = error_rate_poly * len(Y)
    
        
        print(f'Degree of the polynomial: {n}. Number of errors (polynomial): {num_errors_poly:.0f}. Error rate: {error_rate_poly:.4f}.')
    
        n += 1

