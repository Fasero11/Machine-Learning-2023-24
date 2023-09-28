import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import log
from math import sqrt
from math import pi

# Datos
x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
X = np.column_stack((x1, x2))
Y = np.array([1.56, 1.95, 2.44, 3.05, 3.81, 4.77, 5.96, 7.45, 9.31, 11.64])
# w0 y w1
#W = np.array([-0.705, 1.072])

#for i in range(len(Y)):
#    for j in range(len(W)):
#       J_vals[i, j] = len(Y)*log(1/sqrt(2*pi)) - (1/2*len(Y))*(Y[i] - W[j]*X[i][j])

#print(J_vals)
# Malla de parámetros
#w0_vals = np.linspace(-2, 2, 100)
#w1_vals = np.linspace(-2, 2, 100)
#w0_mesh, w1_mesh = np.meshgrid(w0_vals, w1_vals)

# Calcular la función de coste (negativo del log-verosimilitud)
#J_vals = np.zeros_like(w0_mesh)
#for i in range(len(w0_vals)):
#    for j in range(len(w1_vals)):
        #J_vals[i, j] = len(Y)*log(1/sqrt(2*pi)) - (1/2*len(Y))*(Y[i] - (w0_vals[j]*X[i][j]))
#        J_val = len(Y) * log(1 / sqrt(2 * pi)) - (1 / (2 * len(Y))) * np.sum((Y - (W[0] + W[1] * X))**2)
w0_vals = np.linspace(-10, 10, 100)
w1_vals = np.linspace(-10, 10, 100)
w0_mesh, w1_mesh = np.meshgrid(w0_vals, w1_vals)

# Calcular la función de coste para cada combinación de w0 y w1
J_vals = np.zeros_like(w0_mesh)
for i in range(len(w0_vals)):
    for j in range(len(w1_vals)):
        J_vals[i, j] = len(Y) * log(1 / sqrt(2 * pi)) - (1 / (2 * len(Y))) * np.sum((Y - (w0_vals[i]* X[:, 0]+ w1_vals[i] *  X[:, 1]))**2)

# Graficar en 3D
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')

#ax.plot_surface(theta0_mesh, theta1_mesh, cmap='viridis')
ax.plot_surface(w0_mesh, w1_mesh, J_vals, cmap='viridis')
ax.set_xlabel('w0', labelpad=20)
ax.set_ylabel('w1', labelpad=20)
ax.set_zlabel('log V(w)', labelpad=20)
ax.set_title('Función de Coste')
plt.show()

