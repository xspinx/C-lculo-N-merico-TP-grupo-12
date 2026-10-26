"""
================================================================================
RUNGE-KUTTA DE 2do ORDEN — Euler Mejorado / Método del Punto Medio
Universidad Austral - Facultad de Ingeniería - Cálculo Numérico
================================================================================

DEDUCCIÓN DE CÁTEDRA (3 ecuaciones y 4 incógnitas):
--------------------------------------------------
Comparando con la serie de Taylor de segundo orden:
    w1 + w2 = 1
    w2 * alpha = 1/2
    w2 * beta  = 1/2

Con la elección de cátedra: w1 = 0  ==>  w2 = 1,  alpha = 1/2,  beta = 1/2

FÓRMULA (transcripción textual de la diapositiva):
-------------------------------------------------
    F1 = h * f(x_i, y_i)
    F2 = h * f(x_i + 1/2 * h,  y_i + 1/2 * F1)
    
    y_{i+1} = y_i + F2

Difiere de Taylor recién en h^3, por lo tanto el error global es O(h^2) (ORDEN 2).
Esta fórmula se conoce como EULER MEJORADO ó MÉTODO DEL PUNTO MEDIO.

JUSTIFICACIÓN DE SU USO (Diapositiva de clase):
-----------------------------------------------
    "Para mejorar la solución podemos:
        - disminuir el valor h
        - aumentar el orden
     Disminuir h puede ser muy ineficiente, entonces vamos a aumentar el orden."
================================================================================
"""

import numpy as np
from modelo import derivadas


def integrar_rk2(estado_inicial, paso_temporal, tiempo_final):
    """
    Integra el sistema de EDOs con Runge-Kutta 2 (Euler Mejorado / Punto Medio).

    Parametros
    ----------
    estado_inicial : ndarray de shape (3,)
        Condición inicial [N0, P0, S0].
    paso_temporal : float
        El dt (h).
    tiempo_final : float
        Hasta qué tiempo integrar.

    Devuelve
    --------
    tiempos : ndarray de shape (n_pasos+1,)
    trayectoria : ndarray de shape (n_pasos+1, 3)
    """
    cantidad_de_pasos = int(round(tiempo_final / paso_temporal))

    tiempos = np.linspace(0.0, tiempo_final, cantidad_de_pasos + 1)
    trayectoria = np.zeros((cantidad_de_pasos + 1, len(estado_inicial)))
    trayectoria[0] = estado_inicial

    h = paso_temporal

    for indice_paso in range(cantidad_de_pasos):
        estado_actual = trayectoria[indice_paso]

        # F1 = h * f(x_i, y_i)
        F1 = h * derivadas(estado_actual)

        # F2 = h * f(x_i + 1/2*h, y_i + 1/2*F1)
        estado_medio = estado_actual + 0.5 * F1
        F2 = h * derivadas(estado_medio)

        # y_{i+1} = y_i + F2
        trayectoria[indice_paso + 1] = estado_actual + F2

    return tiempos, trayectoria


if __name__ == "__main__":
    estado_inicial = np.array([5.0, 2.0, 1.0])
    tiempos, trayectoria = integrar_rk2(estado_inicial, paso_temporal=0.01, tiempo_final=200.0)
    print(f"Pasos dados     : {len(tiempos) - 1}")
    print(f"Estado inicial  : {trayectoria[0]}")
    print(f"Estado final    : {trayectoria[-1]}")
