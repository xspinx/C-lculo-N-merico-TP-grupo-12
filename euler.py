r"""
================================================================================
METODO DE EULER EXPLICITO  (orden 1)
================================================================================

Archivo separado del de Runge-Kutta, como pide el enunciado.

FORMULA
-------
    y_{n+1} = y_n + dt * f(y_n)

IDEA
----
En cada paso se aproxima la curva solucion por su RECTA TANGENTE en el punto
actual, y se avanza un paso sobre esa recta. Es el metodo mas simple que existe
para resolver un problema de valores iniciales.

DE DONDE SALE
-------------
Del desarrollo de Taylor de la solucion alrededor de t_n:

    y(t_n + dt) = y(t_n) + dt*y'(t_n) + (dt^2/2)*y''(xi)
                  \_______________________/   \__________/
                      lo que Euler calcula    lo que Euler TIRA

Como y'(t_n) = f(y_n), truncar en el termino lineal da exactamente la formula.
El primer termino descartado es O(dt^2): ese es el ERROR LOCAL DE TRUNCAMIENTO.

ORDEN
-----
    error local  (un paso)       : O(dt^2)
    error global (N = T/dt pasos): O(dt)      <- se pierde un orden al acumular

Si se parte dt a la mitad, el error global se parte a la mitad.

LIMITACION
----------
Usa la pendiente SOLO al inicio del intervalo. Si la pendiente cambia durante
el paso (que es lo normal), el metodo se queda corto o se pasa de manera
SISTEMATICA, y ese sesgo se acumula paso a paso.
================================================================================
"""

import numpy as np

from modelo import derivadas


def integrar_euler(estado_inicial, paso_temporal, tiempo_final):
    """
    Integra el sistema de EDOs con el metodo de Euler explicito.

    Parametros
    ----------
    estado_inicial : ndarray de shape (3,)
        Condicion inicial [N0, P0, S0].
    paso_temporal : float
        El dt. Su eleccion NO es libre: ver el analisis de estabilidad.
    tiempo_final : float
        Hasta que tiempo integrar.

    Devuelve
    --------
    tiempos : ndarray de shape (n_pasos+1,)
    trayectoria : ndarray de shape (n_pasos+1, 3)
        Cada fila es el estado [N, P, S] en el instante correspondiente.

    Nota: la funcion es independiente de la DIMENSION del sistema. Sirve igual
    para 2 que para 3 variables porque opera sobre arrays de numpy.
    """
    cantidad_de_pasos = int(round(tiempo_final / paso_temporal))

    tiempos = np.linspace(0.0, tiempo_final, cantidad_de_pasos + 1)
    trayectoria = np.zeros((cantidad_de_pasos + 1, len(estado_inicial)))
    trayectoria[0] = estado_inicial

    for indice_paso in range(cantidad_de_pasos):
        estado_actual = trayectoria[indice_paso]

        # Unica evaluacion del lado derecho: la pendiente al INICIO del intervalo
        pendiente = derivadas(estado_actual)

        # Avance sobre la recta tangente
        trayectoria[indice_paso + 1] = estado_actual + paso_temporal * pendiente

    return tiempos, trayectoria


if __name__ == "__main__":
    estado_inicial = np.array([5.0, 2.0, 1.0])
    tiempos, trayectoria = integrar_euler(estado_inicial, paso_temporal=0.01, tiempo_final=200.0)
    print(f"pasos dados     : {len(tiempos) - 1}")
    print(f"estado inicial  : {trayectoria[0]}")
    print(f"estado final    : {trayectoria[-1]}")
