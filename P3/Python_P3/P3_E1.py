# Aprendizaje Automático 2023 - 2024

# Práctica 3 - Ejercicio 1

# Julia López Augusto
# Gonalo Vega Pérez

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from sklearn.metrics import pairwise_distances

if __name__ == '__main__':

    current_dir = os.getcwd()
    train_data = pd.read_csv(str(current_dir) + "/P3/Data_P3/trainingsetkmeans.csv")  

    x1 = train_data['X1']
    x2 = train_data['X2']
    x3 = train_data['X3']

    X = np.column_stack((x1, x2, x3))

    # Ejercicio 1.1

    # Representar graficamente
    fig = plt.figure
    ax = plt.axes(projection='3d')

    # Mostrar datos iniciales
    ax.scatter3D(x1, x2, x3, marker='x') 

    # Nombrar Ejes
    ax.set_xlabel('x1', labelpad=20)
    ax.set_ylabel('x2', labelpad=20)
    ax.set_zlabel('x3', labelpad=20)

    plt.show()

    # Ejercicio 1.2

    kmeans = KMeans(n_clusters=3, init='k-means++')
    kmeans.fit(X)
    
    # Representar graficamente
    fig = plt.figure
    ax = plt.axes(projection='3d')

    # Graficar centroides
    centroids = kmeans.cluster_centers_

    for centroid in centroids:
      ax.scatter3D(centroid[0], centroid[1], centroid[2], marker='o', color="magenta")

    # Graficar datos por colores según la clase asignada
    id = 0
    for point in X:
      if kmeans.labels_[id] == 0:
        ax.scatter3D(point[0], point[1], point[2], marker='x', color="red")
      elif kmeans.labels_[id] == 1:
        ax.scatter3D(point[0], point[1], point[2], marker='x', color="blue") 
      else:
        ax.scatter3D(point[0], point[1], point[2], marker='x', color="green")  

      id += 1

    # Nombrar Ejes
    ax.set_xlabel('x1', labelpad=20)
    ax.set_ylabel('x2', labelpad=20)
    ax.set_zlabel('x3', labelpad=20)

    plt.show()

    # Ejercicio 1.3
    id = 0
    distances = []
    for point in X:
      if kmeans.labels_[id] == 0:
        centroid = centroids[0]
      elif kmeans.labels_[id] == 1:
        centroid = centroids[1]
      else:
        centroid = centroids[2]
      # Por cada punto guarda la distancia al centroide y la clase a la que pertenece
      distances.append((np.sqrt(np.sum((point - centroid) ** 2)), kmeans.labels_[id])) 

      id += 1

    distances_1 = []
    distances_2 = []
    distances_3 = []
    fig = plt.figure
    id = 0
    for point in distances:
      if point[1] == 0:
        plt.scatter(id, point[0], color="red")
        distances_1.append(point[0])
      elif point[1] == 1:
        plt.scatter(id, point[0], color="blue")
        distances_2.append(point[0])
      else:
        plt.scatter(id, point[0], color="green")
        distances_3.append(point[0])

      id += 1

    plt.xlabel("Número de dato")
    plt.ylabel("Distancia a centroide")

    plt.show()

    avg_dist_1 = np.mean(distances_1)
    avg_dist_2 = np.mean(distances_2)
    avg_dist_3 = np.mean(distances_3)

    print("Class 1 (red) average distance from centroid 1: " + str(avg_dist_1))
    print("Class 2 (blue) average distance from centroid 1: " + str(avg_dist_2))
    print("Class 3 (green) average distance from centroid 1: " + str(avg_dist_3))