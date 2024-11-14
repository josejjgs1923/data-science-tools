"""contiene funciones generales para implementar modelos de machine learning, visualizar metricas, y graficar predicciones de modelos"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


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


def heatmap(matriz, titulo, formato=".2f", tamaño = (4, 4), mapa="Reds"):
    plt.figure(figsize=tamaño)
    sns.heatmap(matriz, annot=True, cmap=mapa, fmt=formato)
    plt.title(titulo)

def cluster_plot(data_x, data_y, labels, titulos, data_centroides=None):

    fig = plt.figure(figsize=(5,5))

    ax  =  fig.add_subplot(1, 1, 1)

    # Graficar los datos y los centros de clústeres
    ax.scatter(data_x, data_y, c=labels, cmap='viridis')

    if data_centroides is not None:
        ax.scatter(*data_centroides, s=200, marker='^', c='red')

    ax.set_title(titulos[0])
    ax.set_xlabel(titulos[1])
    ax.set_ylabel(titulos[2])        

    plt.show()


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


def grafico_codo(numero_clusteres, suma_distancias, punto_codo=None):
    fig = plt.figure(figsize=(8, 5))

    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(numero_clusteres, suma_distancias, marker='o', linestyle='-', color='b')
    ax.set_title('Método del Codo')
    ax.set_xlabel('Número de Clusters k')
    ax.set_ylabel('Distorsión (SSE)')

    if punto_codo is not None:
        ax.axvline(x=punto_codo, linestyle='--', color='r', label=f"Codo en k = {punto_codo:d}")
        ax.legend()
        
    ax.grid(True)
    plt.show()
    
   

def conjunto_cluster_plot(data_x, caracteristicas, modelo_kmeans):

    labels = modelo_kmeans.labels_
    cant_caracteristicas = len(caracteristicas)

    try:
        centroides = modelo_kmeans.cluster_centers_
        
        for variable_1 in range(cant_caracteristicas):
            for variable_2 in range(variable_1 + 1, cant_caracteristicas):
    
                cluster_plot(data_x[:, variable_1], data_x[:, variable_2],
                             labels, ('Resultados de K-Means', caracteristicas[variable_1], caracteristicas[variable_2]),
                             data_centroides=(centroides[:, variable_1], centroides[:, variable_2])
                            )

    except AttributeError:
        
        for variable_1 in range(cant_caracteristicas):
            for variable_2 in range(variable_1 + 1, cant_caracteristicas):
    
                cluster_plot(data_x[:, variable_1], data_x[:, variable_2],
                             labels, ('Agglomerative Clustering', caracteristicas[variable_1], caracteristicas[variable_2])
                            )


if __name__ == "__main__":
    pass



