#!/usr/bin/env python
"""
Almacenar funciones y clases que facilitan los EDAs, analisis de hipotesis, probabilidad y en general estadistica.
"""

import numpy as np
import pandas as pd
from scipy import stats


def tamaño_sturgess(data: pd.DataFrame):
    """
    calcular el tamaño y lo limites de las categorias (bins) para la clasificación de una variable numerica continua
    usando la regla de sturgess.

    parametros:

        data: vector o estructura de una sola dimension, conteniendo los valores de los datos.

    retorna:

        tamaño: tamaño calculado de las categorias.
        bins: arreglo con los limites de las categorias.
    """
    maximo = np.max(data)

    minimo = np.min(data)

    tamaño = (maximo - minimo) / (1 + 3.3 * np.log10(len(data)))

    bins = np.arange(minimo, maximo, tamaño)

    return tamaño, bins


def medidas_resumen(datos: pd.DataFrame, campo: str):
    """
    Calcular y mostrar en un dataframe las medidas estatisticas de
    resumen para un campo. Se presenta el promedio, mediana, cuartiles, desviación estandar
    y varianza.

    Parametros:
        datos: dataframe con los datos.
        campo: columna donde se van a realizar los calculos.

    retorna:
        resumen: dataframe, filas siendo las caracteristicas (como indices)
    """
    serie = datos[campo]

    resumen = serie.describe()

    resumen['var'] = serie.var()

    resumen = pd.DataFrame(resumen)

    return resumen


def tabla_frecuencia(datos: pd.DataFrame, campo: str):
    """
    Construye una un dataframe, calculando las frecuencias
    absolutas y relativas del campo dado.

    parametros:
        datos: dataframe con los datos.
        campo: columna de la cual se va a realizar la tabla de frecuencia.
               debe ser un variable categorica.

    retorna:
        frec: dataframe con la tabla de frecuencia.
    """
    frec = pd.crosstab(datos[campo], 'frecuencia')

    relativa = frec / datos.shape[0]

    frec['relativa'] = relativa

    frec.loc['suma'] = [datos.shape[0], relativa['frecuencia'].sum()]

    return frec


def calcular_valor_t(
    X: pd.DataFrame,
    y_real: np.ndarray,
    y_pred: np.ndarray,
    estadistico_estimado: float | int,
    estadistico_esperado: float | int,
):
    """
    Calcular valores t, comparando los valores estimados de un estadistico vs los valores
    esperados. se calculan los valores t y lo valores p correspondientes.

    parametros:
        X: dataframe con los datos de la variable independiente.
        y_real: vector con los datos experimentales o reales.
        estadistico_estimado: valor calculado del estadistico en prueba.
        estadistico_esperado: valor esperado del estadistico en prueba.

    retorna:
        valor_t: vector con los valores del estadistico t-student.
        valor_p: vector con los valores P equivalentes.
    """

    # obtener cantidad de puntos de datos
    n = y_real.shape[0]

    # calculo de resiudos, despues de haber entrenado un modelo
    residuos = y_real - y_pred

    # Calculo de la desviacion estandar de los residuos.
    # grado de libertad n-2 por ser un modelo regresion

    SC = residuos**2

    std_residuos = np.sqrt(SC.sum() / (n - 2))

    # calculo de la desviacion estandar de las caracteriscas

    std_X = X.values.std()

    # calculo del error estandar. se invierte para usarlo directamente despues

    inv_std_error = (np.sqrt(n - 1) / std_residuos) * (std_X)

    # calculo del estadisticos t y p,  con t = (m_estimado - m_esperado)/error_estandar
    valor_t = (estadistico_estimado - estadistico_esperado) * inv_std_error

    valor_p = stats.t.sf(np.abs(valor_t), df=n - 2) * 2

    return valor_t, valor_p


def tabla_contingencia(datos, factores, relativa=False):
    """
    Construir una tabla de contingencia.
    """
    f1, f2 = factores

    tabla = pd.crosstab(datos[f1], datos[f2], margins=True)

    if relativa:
        for campo in tabla.columns[:-1]:
            tabla[campo] /= tabla[campo].iloc[-1]

        tabla['All'] /= tabla['All']

    return tabla


def test_chi_cuadrado(datos):
    """
    crea un dataframe, con las filas siendo los totales de las columnas, duplicado
    la cantidad de filas de los datos
    """

    totales_por_columna_aumentado = pd.DataFrame(
        [datos.iloc[-1, :-1].values] * (datos.shape[0] - 1)
    )

    probabilidad_positiva = datos.iloc[:-1, -1].values / datos.iloc[-1, -1]

    # calculo de la frecuencias esperadas, multiplicando la probabilidad del
    # evento positivo por el total de columna, por cada fila
    frec_esp = totales_por_columna_aumentado.mul(probabilidad_positiva, axis=0)

    # delta de los grados de libertad para el chi cuadrado
    ddof = sum(frec_esp.shape) - 2

    gl = (frec_esp.shape[0] - 1) * (frec_esp.shape[1] - 1)

    # el parametro axis, hace que las frecuencias se tomen como si fuera un vector
    return (
        stats.chisquare(
            datos.iloc[:-1, :-1], f_exp=frec_esp, ddof=ddof, axis=None
        ),
        gl,
    )


def mostrar_chi(resultados):
    """
    mostrar un pequeño resumen de los
    resultados de un prueba chi-cuadrado
    """
    test, gl = resultados

    print(f'Chi Cuadrado    : {test.statistic:.3f}')

    print(f'Valor P         : {test.pvalue:.3f}')

    print(f'Grados libertad : {gl}')


def mostrar_anova(resultados):
    """
    mostrar un pequeño resumen de los
    resultados de un prueba ANOVA
    """

    print(f'Valor F: {resultados.statistic:.3f}')

    print(f'Valor P: {resultados.pvalue:.3f}')


if __name__ == '__main__':
    pass
