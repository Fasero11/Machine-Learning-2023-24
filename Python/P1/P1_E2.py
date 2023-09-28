import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import log
from math import sqrt
from math import pi


# Función de log-verosimilitud para regresión lineal
#def log_likelihood(theta0, theta1, X, Y):
#    n = len(Y)
#    y_pred = theta0 * X[:, 0] + theta1 * X[:, 1]
#    error_squared = (Y - y_pred)**2
#    log_likelihood_value = -0.5 * np.sum(error_squared)
#    return log_likelihood_value



# Datos
x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
X = np.column_stack((x1, x2))
Y = np.array([1.56, 1.95, 2.44, 3.05, 3.81, 4.77, 5.96, 7.45, 9.31, 11.64])
# w0 y w1
W = np.array([-0.705, 1.072])


#print(X[2][1])
#J_vals = np.zeros_like(theta0_mesh)
#mesh = np.meshgrid(len(Y), len(W))
#J_vals = np.zeros_like(mesh)
#print(mesh)
for i in range(len(Y)):
    for j in range(len(W)):
       J_vals[i, j] = len(Y)*log(1/sqrt(2*pi)) - (1/2*len(Y))*(Y[i] - W[j]*X[i][j])

#print(J_vals)
# Malla de parámetros
#theta0_vals = np.linspace(-10, 10, 100)
#theta1_vals = np.linspace(-10, 10, 100)
#theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)

# Calcular la función de coste (negativo del log-verosimilitud)
#J_vals = np.zeros_like(theta0_mesh)
#for i in range(len(theta0_vals)):
#    for j in range(len(theta1_vals)):
#        J_vals[i, j] = -log_likelihood(theta0_vals[i], theta1_vals[j], X, Y)

# Graficar en 3D
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')

#ax.plot_surface(theta0_mesh, theta1_mesh, cmap='viridis')
ax.set_xlabel('w0', labelpad=20)
ax.set_ylabel('w1', labelpad=20)
ax.set_zlabel('log V(w)', labelpad=20)
ax.set_title('Función de Coste')
plt.show()

