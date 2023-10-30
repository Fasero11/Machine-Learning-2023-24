# Aprendizaje Automático 2023 - 2024

# Práctica 2 - Ejercicio 1

# Julia López Augusto
# Gonzalo Vega Pérez

# Importa las bibliotecas necesarias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


if __name__ == '__main__':
# 1.2. Evaluación del modelo con nuevos datos
  nuevos_datos = {
    'PREVISIÓN': ['SOLEADO', 'NUBLADO', 'NUBLADO', 'SOLEADO', 'LLUVIOSO', 'LLUVIOSO', 'NUBLADO', 'SOLEADO', 'NUBLADO', 'LLUVIOSO'],
    'TEMPERATURA': ['ALTA', 'BAJA', 'MEDIA', 'ALTA', 'MEDIA', 'MEDIA', 'BAJA', 'ALTA', 'MEDIA', 'BAJA'],
    'MAREA': ['MEDIA', 'MEDIA', 'BAJA', 'ALTA', 'BAJA', 'ALTA', 'MEDIA', 'ALTA', 'BAJA', 'MEDIA'],
    'VIENTO': ['DEBIL', 'MEDIO', 'MEDIO', 'FUERTE', 'MEDIO', 'DEBIL', 'MEDIO', 'DEBIL', 'MEDIO', 'FUERTE'],
    'PESCAR': ['NO', 'NO', 'SI', 'NO', 'SI', 'SI', 'SI', 'NO', 'SI', 'SI']
  }

  df_nuevos = pd.DataFrame(nuevos_datos)

  # Transformar variables categóricas a numéricas
  df_nuevos = pd.get_dummies(df_nuevos, columns=['PREVISIÓN', 'TEMPERATURA', 'MAREA', 'VIENTO'], drop_first=True)

  X_nuevos = df_nuevos.drop('PESCAR', axis=1)
  y_nuevos = df_nuevos['PESCAR']

    # Crear el modelo de árbol de decisión
  model = DecisionTreeClassifier()

  # Predecir con el modelo existente
  predicciones = model.predict(X_nuevos)

  # Evalúa el modelo con los nuevos datos
  #accuracy = model.score(X_nuevos, y_nuevos)
  #print("Porcentaje de acierto en los nuevos datos:", accuracy * 100, "%")

  # Calcular el porcentaje de acierto
  exactitud = accuracy_score(y_nuevos, predicciones)
  print("Porcentaje de acierto:", exactitud * 100, "%")

  # 1.3. Modelo de árbol de decisión utilizando los datos de regresión logística
  # No se proporcionaron datos de regresión logística para este ejercicio, por lo que no se puede generar el modelo.

  # 1.4. Cálculo del error en porcentaje sobre los datos de entrenamiento
  error = 1 - model.score(X, y)
  print("Error en porcentaje:", error * 100, "%")
