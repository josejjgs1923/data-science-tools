#!/usr/bin/env python
"""
contiene funciones generales para implementar modelos de machine learning, visualizar metricas, y graficar predicciones de modelos
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from .graficos import heatmap

def mostrar_metricas(y_real, y_predicho, decimales = 4):
    
    mse = metrics.mean_squared_error(y_real, y_predicho)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_real, y_predicho)
    mae =  metrics.mean_absolute_error(y_real, y_predicho)
    print(
    f"R2  : {r2:.{decimales}f}",
    f"MSE : {mse:.{decimales}f}",
    f"RMSE: {rmse:.{decimales}f}",
    f"MAE : {mae:.{decimales}f}",
    sep = "\n",
    )


def matriz_confusion(y_real, y_predicho):
    matriz = metrics.confusion_matrix(y_real, y_predicho)
    heatmap(matriz, "Matriz Confusion", "d")
    ax = plt.gca()
    ax.set_xlabel("Valor Predicho")
    ax.set_ylabel("Valor Real")

def find_elbow_point(x, y):
    # Crear la línea desde el primer hasta el último punto
    line = np.array([x[0], y[0], x[-1], y[-1]])

    # Calcular la distancia de cada punto a la línea
    distances = []
    for i in range(len(x)):
        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[-1], y[-1]])
        p = np.array([x[i], y[i]])
        distance = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
        distances.append(distance)

    # El punto con la máxima distancia es el codo
    elbow_index = np.argmax(distances)
    return x[elbow_index]


if __name__ == "__main__":
    pass



