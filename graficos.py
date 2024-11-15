#!/usr/bin/env python
"""
contiene funciones generales para visualizar los resultados de modelos, EDAs y otras visualizaciones.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def grafico_linear_simple(x_entre, y_entre, x_prueba, y_prueba, modelo, titulo, titulo_x, titulo_y):
    plt.scatter(x_entre, y_entre, color="c", label="entrenamiento")
    
    plt.scatter(x_prueba, y_prueba, color="r", label="prueba")

    x_reg = np.linspace(np.min(x_entre), np.max(x_entre), 300).reshape(-1, 1)
    y_reg = modelo.predict(x_reg)
    
    plt.plot(x_reg, y_reg, "k--", label="linea regresión")

    plt.title(titulo)

    # Etiquetas de los ejes (opcional)
    plt.xlabel(titulo_x)
    plt.ylabel(titulo_y)
    plt.legend()

    # Mostrar el gráfico
    plt.show()


def grafico_real_vs_predicho(y_prueba, y_prueba_pred, titulo, titulo_x, titulo_y):
    linea = np.linspace(np.min(y_prueba_pred), np.max(y_prueba_pred), 300)
    
    plt.plot(linea, linea, color='k', linestyle='--', label='y = x')
    
    plt.scatter(y_prueba, y_prueba_pred, color="c")
    
    plt.title(titulo)
    
    # Etiquetas de los ejes (opcional)
    plt.xlabel(titulo_x)
    plt.ylabel(titulo_y)
    plt.legend()
    
    # Mostrar el gráfico
    plt.show()


def grafico_residuos(y_real, y_predicho, titulo, titulo_x, titulo_y, color="g"):
    """
    Calcular y graficar residuos de un modelo de regresion versus los valores
    predichos.
    """
    fig = plt.figure()

    ax = fig.add_subplot()

    residuos = y_real - y_predicho
    
    ax.scatter(y_predicho, residuos, color=color, alpha=0.6)

    ax.axhline(y=0, color="k", linestyle="--")

    ax.set_title(titulo)

    ax.set_xlabel(titulo_x)

    ax.set_ylabel(titulo_y)

    plt.show()


def heatmap(matriz, titulo, formato=".2f", tamaño = (4, 4), mapa="Reds"):
    """
    Presentar un vision grafica de una matrix, com un mapa de calor.
    """
    plt.figure(figsize=tamaño)
    sns.heatmap(matriz, annot=True, cmap=mapa, fmt=formato)
    plt.title(titulo)


def cluster_plot(data, labels, titulos, data_centroides=None, ax=None):
    """
    graficar los puntos de datos y los centroides para los resultados 
    de clasificaciones de agrupamiento.
    """
    
    # Graficar los datos y los centros de clústeres
    if ax is None:
        fig = plt.figure()

        ax = fig.add_subplot(1,1,1)
        
    sns.scatterplot(data, x=titulos[1], y=titulos[2], hue=labels, palette="viridis", ax=ax)

    if data_centroides is not None:
        centroides = (data_centroides.loc[:, titulos[1]], data_centroides.loc[:, titulos[2]])
        
        ax.scatter(*centroides, s=100, marker='^', c='red')

    ax.set_title(titulos[0])
    ax.set_xlabel(titulos[1])
    ax.set_ylabel(titulos[2])        

    plt.show()


def grafico_codo(numero_clusteres, suma_distancias, titulos, punto_codo=None):
    """
    realizar un grafico de dispersion de los valores de distorsion o varianza 
    en un analisis de componentes principales. grafica el punto de codo, 
    que supuestamente reduce la dispersion o aumenta la varianza optimamente.
    """
    fig = plt.figure(figsize=(8, 5))

    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(numero_clusteres, suma_distancias, marker='o', linestyle='-', color='b')
    ax.set_title(titulos[0])
    ax.set_xlabel(titulos[1])
    ax.set_ylabel(titulos[2])

    if punto_codo is not None:
        ax.axvline(x=punto_codo, linestyle='--', color='r', label=f"Codo en x = {punto_codo:d}")
        ax.legend()
        
    ax.grid(True)
    plt.show()
    

def conjunto_cluster_plot(data, titulo, modelo, caracteristicas=None, cmap="viridis", tamaño=None):
    """
    Graficar conjunto de cluster plots, creando un grafico de clusters por cada
    para de variables.
    """

    labels = modelo.labels_
    
    try:
        data_centroides = pd.DataFrame(modelo.cluster_centers_, columns=data.columns)

        def grafico(eje, variable_x, variable_y):
            eje.scatter(data=data, x=variable_x, y=variable_y, c=labels, s=20, marker=".", cmap=paleta)

            centroides = (data_centroides.loc[:, variable_x], data_centroides.loc[:, variable_y])
        
            eje.scatter(*centroides, s=100, marker='^', c='red')
        
    except AttributeError:

        def grafico(eje, variable_x, variable_y):
            eje.scatter(data=data, x=variable_x, y=variable_y, c=labels, s=20, marker=".", cmap=paleta)

    # obtener la paleta de colores, con los mapas de seaborn
    paleta = sns.color_palette(cmap, modelo.n_clusters, as_cmap=True)

    # funcion para indices de combinaciones, dos a dos,
    # para los ejes y las caracteristicas
    def comb(num):
        for idx in range(num):
            for idy in range(idx + 1, num):
                yield idx, idy
                
    def perm(num_filas, num_cols):
        for idx in range(num_filas):
            for idy in range(num_cols):
                yield idx, idy

    # definir cuales son las caracteristicas 
    if caracteristicas is None:
        if tamaño is None:
            tamaño = (20, 20)
            
        caracteristicas = data.columns

        num_filas = num_cols = len(caracteristicas)
    
        # generador de los indices
        indices = comb(num_cols)
    
        #construir grafico de plots multiples:
    
        #obtener figura y conjunto de ejes
        fig = plt.figure(figsize=tamaño, layout="constrained")
    
        axes = fig.subplots(num_filas, num_cols)
    
        # eliminar ejes triangulo superior, vacios
        for idx, idy in comb(num_cols):
            axes[idx, idy].remove()
    
        # eliminar ejes vacios en ultima fila: num_filas - 1 
        for idx in range(num_cols):
            axes[num_filas - 1, idx].remove()
    
        # iterar sobre los pares de caracteristicas y graficar
        
        for idx, idy in indices:
            # obtener el eje en el que se va a graficar con indices
            # hay un desfase de  una fila hacia arriba, para aprovechar los
            # graficos de la diagonal
            
            #idx es el indice de columnas, y idy el de filas, por ser grafico de cuadrillas
            eje = axes[idy - 1, idx]
            
            #tambien, idx se refiere al indice variable x, idy indice variable y, en caracteriscas
            variable_x, variable_y = caracteristicas[idx], caracteristicas[idy]
    
            grafico(eje, variable_x, variable_y)
            
            # graficar titulos y en los graficos de la primera columna: 0
            if idx == 0:
                eje.set_ylabel(variable_y)     
    
            # graficar titulos x en los graficos de la ultima fila: num_filas - 1
            if idy == num_filas - 1 :
                eje.set_xlabel(variable_x)        

    else:
        cant = len(caracteristicas)

        if cant < 4:
            num_filas = 1
            
            num_cols = cant

            def mapeo(idx, idy, axes):
                return axes[idy]

        else:
            num_filas = num_cols = int(np.floor(np.sqrt(cant)))
            
            def mapeo(idx, idy, axes):
                return axes[idx, idy]
                
        if tamaño is None:
            tamaño = (5 * num_cols, 5 * num_filas)

        # generador de los indices
        indices = perm(num_filas, num_cols)
    
        #construir grafico de plots multiples:
    
        #obtener figura y conjunto de ejes
        fig = plt.figure(figsize=tamaño, layout="constrained")
    
        axes = fig.subplots(num_filas, num_cols)

        for (idx, idy), (variable_x, variable_y) in zip(indices, caracteristicas):
            # obtener el eje en el que se va a graficar con indices
            #idx es el indice de filas, y idy el de columnas
            eje = mapeo(idx, idy, axes)
            
            grafico(eje, variable_x, variable_y)
    
            eje.set_xlabel(variable_x)        
            
            eje.set_ylabel(variable_y)     

        for idx, idy in indices:
            axes[idx, idy].remove()
 
    #eje = fig.add_subplot(num_filas, num_cols, id_graf)

    # colocar titulo global de los graficos
    plt.suptitle(titulo)

    plt.show()

if __name__ == "__main__":
    pass
