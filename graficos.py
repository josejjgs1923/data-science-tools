#!/usr/bin/env python
"""
contiene funciones generales para visualizar los resultados de modelos, EDAs y otras visualizaciones.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.cluster import (
    AgglomerativeClustering as _AgglomerativeClustering,
    KMeans as _KMeans,
)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as _Axes
from typing import Collection as _Collection, Optional as _Optional


def grafico_linear_simple(
    x_entre: np.ndarray,
    y_entre: np.ndarray,
    x_prueba: np.ndarray,
    y_prueba: np.ndarray,
    modelo: _LinearRegression,
    titulos: tuple[str, str, str],
) -> None:
    """
    graficar un modelo de regresion lineal simple: mostrando un grafico de dispersion de las predicciones
    y de los valores reales (datos de entrenamiento y de prueba). las predicciones se muestran como una linea
    punteada negra.

    parametros:
        x_entre: arreglo de numpy con los datos independientes de entrenamiento.
        y_entre: arreglo de numpy con los datos dependientes de entrenamiento.
        x_prueba: arreglo con los datos de prueba independientes.
        y_prueba: arreglo con los datos de prueba dependientes.
        modelo: modelo de regresion lineal simple de sklearn.
        titulos: tupla con tres titulos: titulo grafico, titulo eje x, titulo eje y.
    )
    """
    plt.scatter(x_entre, y_entre, color='c', label='entrenamiento')

    plt.scatter(x_prueba, y_prueba, color='r', label='prueba')

    x_reg = np.linspace(np.min(x_entre), np.max(x_entre), 300).reshape(-1, 1)
    y_reg = modelo.predict(x_reg)

    plt.plot(x_reg, y_reg, 'k--', label='linea regresión')

    plt.title(titulos[0])

    # Etiquetas de los ejes (opcional)
    plt.xlabel(titulos[1])
    plt.ylabel(titulos[2])
    plt.legend()

    # Mostrar el gráfico
    plt.show()


def grafico_real_vs_predicho(
    y_prueba: np.ndarray,
    y_prueba_pred: np.ndarray,
    titulos: tuple[str, str, str],
) -> None:
    """
    graficar los resultados de un modelo aprendizaje supervisado. Se un grafico de dispersion,
    mostrando puntos de la variable dependiente reales vs los predichos.

    parametros:
        y_prueba: arreglo de numpy con los datos de la variale indepentiente reales (de prueba).
        y_prueba_pred: arreglo de numpy con los datos predichos.
        titulos: tupla con tres titulos: titulo grafico, titulo eje x, titulo eje y.
    """
    linea = np.linspace(np.min(y_prueba_pred), np.max(y_prueba_pred), 300)

    plt.plot(linea, linea, color='k', linestyle='--', label='y = x')

    plt.scatter(y_prueba, y_prueba_pred, color='c')

    plt.title(titulos[0])

    # Etiquetas de los ejes (opcional)
    plt.xlabel(titulos[1])
    plt.ylabel(titulos[2])
    plt.legend()

    # Mostrar el gráfico
    plt.show()


def grafico_residuos(
    y_real: np.ndarray,
    y_predicho: np.ndarray,
    titulos: tuple[str, str, str],
    color: str = 'g',
) -> None:
    """
    Calcular y graficar residuos de un modelo de regresion, y graficar contra
    los valores predichos.

    parametros:
        y_real: arreglo de numpy con los datos de la variable dependiente reales.
        y_predicho: arreglo de numpy con los datos de la varible dependientes predichos.
        titulos: tupla con tres titulos: titulo grafico, titulo eje x, titulo eje y.
        color: color usado para los puntos, sigue la convención usada por matplotlib.
            por defecto es el verde.
    """
    fig = plt.figure()

    ax = fig.add_subplot()

    residuos = y_real - y_predicho

    ax.scatter(y_predicho, residuos, color=color, alpha=0.6)

    ax.axhline(y=0, color='k', linestyle='--')

    ax.set_title(titulos[0])

    ax.set_xlabel(titulos[1])

    ax.set_ylabel(titulos[2])

    plt.show()


def heatmap(
    matriz: np.ndarray,
    titulo: str,
    formato: str = '.2f',
    tamaño: tuple[int, int] = (4, 4),
    mapa: str = 'Reds',
) -> None:
    """
    Presentar un vision grafica de una matriz, com un mapa de calor.

    parametros:
        matriz: arreglo de numpy, siendo una matriz.
        titulo: titulo del grafico
        formato: formato numerico a usar. por defecto usa dos decimales.
        tamaño: tamaño del grafico, tupla en la forma (tamaño x, tamaño y).
        mapa: mapa de color para el grafico. usa los mapas de seaborn.
    """
    plt.figure(figsize=tamaño)

    sns.heatmap(matriz, annot=True, cmap=mapa, fmt=formato)

    plt.title(titulo)

    plt.show()


def cluster_plot(
    data: pd.DataFrame,
    labels: np.ndarray,
    titulos: tuple[str, str, str],
    data_centroides: _Optional[pd.DataFrame] = None,
    ax: _Optional[_Axes] = None,
) -> None:
    """
    graficar puntos de datos (y los centroides opcionalmente) para los resultados
    de clasificaciones de un modelo no supervisado de agrupamiento. Los numero_clusteres
    se colorean.

    parametros:
        data: dataframe con los datos a graficar
        labels: arreglo de numpy con la indicacion (número) del cluster al cual pertenece.
        titulos: tupla con el titulo grafico, titulo eje x, titulo eje y. los titulos x y y deben ser tambien nombres
                en las columnas del dataframe data, y de las columnas del dataframe data_centroides.
        data_centroides: dataframe con la información de los centroides.
        ax: ejes de matplotlib, en caso de que no se quieran generar nuevos ejes.
    """

    # Graficar los datos y los centros de clústeres
    if ax is None:
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)

    sns.scatterplot(
        data, x=titulos[1], y=titulos[2], hue=labels, palette='viridis', ax=ax
    )

    if data_centroides is not None:
        centroides = (
            data_centroides.loc[:, titulos[1]],
            data_centroides.loc[:, titulos[2]],
        )

        ax.scatter(*centroides, s=100, marker='^', c='red')

    ax.set_title(titulos[0])
    ax.set_xlabel(titulos[1])
    ax.set_ylabel(titulos[2])

    plt.show()


def grafico_codo(
    variacion: np.ndarray,
    metricas: np.ndarray,
    titulos: tuple[str, str, str],
    punto_codo: _Optional[float] = None,
) -> None:
    """
    realizar un grafico de codo: se grafica las metricas de optimización contra la variable de variación
    opcionalmente, se puede graficar el punto de codo.

    parametros:
        variacion: arreglo numpy con los cambios para la metrica.
        metricas: arreglo numpy con la metrica calculada para cada variación.
        titulos: tupla con tres titulos: titulo grafico, titulo eje x, titulo eje y.
        punto_codo: numero que representa opcionalmente un punto de codo para graficar.
    """
    fig = plt.figure(figsize=(8, 5))

    ax = fig.add_subplot(1, 1, 1)

    ax.plot(variacion, metricas, marker='o', linestyle='-', color='b')
    ax.set_title(titulos[0])
    ax.set_xlabel(titulos[1])
    ax.set_ylabel(titulos[2])

    if punto_codo is not None:
        ax.axvline(
            x=punto_codo,
            linestyle='--',
            color='r',
            label=f'Codo en x = {punto_codo:d}',
        )
        ax.legend()

    ax.grid(True)
    plt.show()


def conjunto_cluster_plot(
    data: pd.DataFrame,
    titulo: str,
    modelo: _AgglomerativeClustering | _KMeans,
    caracteristicas: _Optional[_Collection[tuple[str, str]]] = None,
    cmap: str = 'viridis',
    tamaño: _Optional[tuple[int, int]] = None,
) -> None :
    """
    Graficar conjunto de cluster plots, creando un grafico de clusters por cada
    par de variables. funciona con un modelo KMeans o AgglomerativeClustering de 
    sklearn. Se grafican las combinaciones de las columnas que se encuentren en el 
    dataframe. 

    parametros:
        data: dataframe conteniendo los datos.
        titulo: titulo del grafico.
        modelo: modelo de sklearn de aglomeración.
        caracteristicas: Iterable opcional con pares de caracteristicas a graficar,
               en lugar de las combinaciones por defecto.
        cmap: mapa de calor a usar para el grafico, usa los de seaborn.
        tamaño: tupla opcional para cambiar el tamaño del grafico.
    """

    labels = modelo.labels_

    try:
        data_centroides = pd.DataFrame(
            modelo.cluster_centers_, columns=data.columns #type: ignore
        )

        def grafico(eje, variable_x, variable_y):
            eje.scatter(
                data=data,
                x=variable_x,
                y=variable_y,
                c=labels,
                s=20,
                marker='.',
                cmap=paleta,
            )

            centroides = (
                data_centroides.loc[:, variable_x],
                data_centroides.loc[:, variable_y],
            )

            eje.scatter(*centroides, s=100, marker='^', c='red')

    except AttributeError:

        def grafico(eje, variable_x, variable_y):
            eje.scatter(
                data=data,
                x=variable_x,
                y=variable_y,
                c=labels,
                s=20,
                marker='.',
                cmap=paleta,
            )

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

        # construir grafico de plots multiples:

        # obtener figura y conjunto de ejes
        fig = plt.figure(figsize=tamaño, layout='constrained')

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

            # idx es el indice de columnas, y idy el de filas, por ser grafico de cuadrillas
            eje = axes[idy - 1, idx]

            # tambien, idx se refiere al indice variable x, idy indice variable y, en caracteriscas
            variable_x, variable_y = caracteristicas[idx], caracteristicas[idy]

            grafico(eje, variable_x, variable_y)

            # graficar titulos y en los graficos de la primera columna: 0
            if idx == 0:
                eje.set_ylabel(variable_y)

            # graficar titulos x en los graficos de la ultima fila: num_filas - 1
            if idy == num_filas - 1:
                eje.set_xlabel(variable_x)

        # axes[0,0].legend()

    else:
        cant = len(caracteristicas)

        if cant < 4:
            num_filas = 1

            num_cols = cant

            def mapeo(idx, idy, axes):
                return axes[idy]

        else:
            num_filas = num_cols = int(np.ceil(np.sqrt(cant)))

            def mapeo(idx, idy, axes):
                return axes[idx, idy]

        if tamaño is None:
            tamaño = (5 * num_cols, 5 * num_filas)

        # generador de los indices
        indices = perm(num_filas, num_cols)

        # construir grafico de plots multiples:

        # obtener figura y conjunto de ejes
        fig = plt.figure(figsize=tamaño, layout='constrained')

        axes = fig.subplots(num_filas, num_cols)

        for (variable_x, variable_y), (idx, idy) in zip(
            caracteristicas, indices
        ):
            # obtener el eje en el que se va a graficar con indices
            # idx es el indice de filas, y idy el de columnas
            eje = mapeo(idx, idy, axes)

            grafico(eje, variable_x, variable_y)

            eje.set_xlabel(variable_x)

            eje.set_ylabel(variable_y)

        for idx, idy in indices:
            axes[idx, idy].remove()

        # mapeo(0, 0, axes).legend()

    # eje = fig.add_subplot(num_filas, num_cols, id_graf)

    # colocar titulo global de los graficos
    plt.suptitle(titulo)

    plt.show()


if __name__ == '__main__':
    pass
