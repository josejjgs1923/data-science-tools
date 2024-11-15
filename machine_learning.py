#!/usr/bin/env python
"""
contiene funciones generales para implementar modelos de machine learning, visualizar metricas, y graficar predicciones de modelos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from .graficos import heatmap

def mostrar_metricas(y_real, y_predicho, decimales=4, num_predictores=None):
    """
    Mostrar un tabla con resultados  de medidas de resumen de ajuste de
    modelos: calcula el MSE, RMSE, R2, MAE, R2_aj
    """ 
   # calculo del R cuadrado
    r2 = metrics.r2_score(y_real, y_predicho)
    
    summ = [ f"R2    : {r2:.{decimales}f}", ]

    # calculo R2 cuadrado ajustado, si se pasa un numero de predictores
    if num_predictores is not None:
        n = y_real.size
        
        r2_adj = 1 - ( (1 - r2) * (n - 1) ) / (n - num_predictores - 1) 

        summ.append(f"R2_adj: {r2_adj:.{decimales}f}")

    # Calculo de metricas MSE, RMSE, MAE
    
    mse = metrics.mean_squared_error(y_real, y_predicho)
    
    rmse = np.sqrt(mse)
 
    mae =  metrics.mean_absolute_error(y_real, y_predicho)
    
    summ.extend(
        (
        f"MSE   : {mse:.{decimales}f}",
        f"RMSE  : {rmse:.{decimales}f}",
        f"MAE   : {mae:.{decimales}f}",
        ))

    return "\n".join(summ)

def mostrar_parametros(variables, modelo, valores_t, valores_p, nombre_parametros="parametros"):
    """
    Contruir un dataframe para mostrar los resultados de un modelo de regresion.
    muestra los nombres de las variables, valores de parametros estimados,
    valores t y valores p.
    """
    summ = pd.DataFrame(
        {"variables":[*variables, "intercepto"], 
        "parametros":[*modelo.coef_.ravel(), *modelo.intercept_.ravel()],
        "valores t": valores_t,
        "valores_p": valores_p
        })

    return summ


def matriz_confusion(y_real, y_predicho):
    matriz = metrics.confusion_matrix(y_real, y_predicho)
    heatmap(matriz, "Matriz Confusion", "d")
    ax = plt.gca()
    ax.set_xlabel("Valor Predicho")
    ax.set_ylabel("Valor Real")


def punto_codo(x, y):
    """
    hallar el punto de codo de un modelo de clasificacion por agrupación,
    o en el contexto de un PCA
    """
    #colocar x en una matrix con y, donde las filas seran el conjunto de puntos: [[x1, y1], [x2, y2], [x3, y3]...]
    puntos = np.vstack((x, y)).T

    extremo_menor = puntos[0,:]

    # vectores diferencia entre el punto menor y cada punto
    dif = extremo_menor - puntos

    # vector diferencia entre el punto extremo mayor y el menor
    dif_extremos = dif[-1,:]

    # Calcular la distancia de cada punto a la línea, usando el producto punto
    norma_extremos = np.linalg.norm(dif_extremos)

    distancias = np.abs(np.cross(dif_extremos, dif)) / norma_extremos

    # El punto con la máxima distancia es el codo
    indice_codo = np.argmax(distancias)
    
    return x[indice_codo]


if __name__ == "__main__":
    pass



