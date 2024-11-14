#!/usr/bin/env python
"""
contiene funciones generales para visualizar los resultados de modelos, EDAs y otras visualizaciones.
"""

import seaborn as sns
import matplotlib.pyplot as plt

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
