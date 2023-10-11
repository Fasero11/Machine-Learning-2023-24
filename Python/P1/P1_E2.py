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
# w0 = -0.705, w1 = 1.072 y w2 = 0
#W = np.array([-0.705, 1.072])

# Para poder representarlo en 3D hay que dar un rango para expresar w0 y w1
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
ax = plt.axes(projection='3d')

ax.plot_surface(w0_mesh, w1_mesh, J_vals, cmap='viridis')
ax.set_xlabel('w0', labelpad=20)
ax.set_ylabel('w1', labelpad=20)
ax.set_zlabel('log V(w)', labelpad=20)
ax.set_title('Función de Coste')
plt.show()

