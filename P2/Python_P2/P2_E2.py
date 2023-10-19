# Aprendizaje Automático 2023 - 2024

# Práctica 2 - Ejercicio 2

# Julia López Augusto
# Gonalo Vega Pérez

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data_train = pd.read_csv("/home/alumnos/gvega/Escritorio/AA/Machine-Learning-2023-24/P2/Data_P2/data.csv")  

    x1 = data_train['X1']
    x2 = data_train['X2']
    x3 = data_train['X3']
    Y = data_train['Y']

    X = np.column_stack((x1, x2, x3))


    # Ejercicio 1.

    # Representar graficamente
    fig = plt.figure
    ax = plt.axes(projection='3d')

    # Mostrar datos iniciales
    ax.scatter3D(x1, x2, x3, marker='x') 

    # Nombrar Ejes
    ax.set_xlabel('x1', labelpad=20)
    ax.set_ylabel('x2', labelpad=20)
    ax.set_zlabel('x3', labelpad=20)

    ax.scatter3D(2, 1, 3, marker='o', color="red") 

    plt.show()


    # Ejercicio 2.

    neigh = KNeighborsClassifier(n_neighbors=3)
    mdl = neigh.fit(X, Y)

    data_to_predict = np.array([2,1,3]).reshape(1, -1)
    Ye = mdl.predict(data_to_predict)

    # Clase a la que pertenece el "dato data_to_predict".
    print("New data class prediction: " + str(Ye)) 


    # Ejercicio 3.
    x1_1 = x1[0:1000]
    x2_1 = x2[0:1000]
    x3_1 = x3[0:1000]
    clase_1 = np.stack((x1_1, x2_1, x3_1))

    x1_2 = x1[1000:2000]
    x2_2 = x2[1000:2000]
    x3_2 = x3[1000:2000]
    clase_2 = np.stack((x1_2, x2_2, x3_2))

    x1_3 = x1[2000:3000]
    x2_3 = x2[2000:3000]
    x3_3 = x3[2000:3000]
    clase_3 = np.stack((x1_3, x2_3, x3_3))

    cov1 = np.cov(clase_1)
    cov2 = np.cov(clase_2)
    cov3 = np.cov(clase_3)
    
    print('\n Cov Clase1:')  
    print(cov1)
    print('\n Cov Clase2:')  
    print(cov2)
    print('\n Cov Clase3:')  
    print(cov3)