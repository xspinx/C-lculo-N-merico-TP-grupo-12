"""
================================================================================
UTILIDADES DE GRAFICADO
================================================================================

Funciones reutilizables para generar las figuras del informe. Cada una recibe
los datos YA CALCULADOS y devuelve la figura; no calcula nada por su cuenta.

CONVENCIONES QUE RESPETAN (y que se evaluan en el informe):
  - ejes rotulados, con unidad cuando corresponde
  - leyenda siempre que haya mas de una curva
  - distinto ESTILO de trazo ademas del color, para que se lea en blanco y negro
  - resolucion 200 dpi
  - el EPIGRAFE no va en la figura: va en el LaTeX, DEBAJO de ella
================================================================================
"""

import matplotlib.pyplot as plt
import numpy as np

RESOLUCION_DPI = 200

# Estilos por especie: color + trazo, para que se distingan impresas en gris
ESTILO_PRESA = {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.7}
ESTILO_DEPREDADOR = {"color": "#d62728", "linestyle": "--", "linewidth": 1.7}
ESTILO_SUPERDEPREDADOR = {"color": "#2ca02c", "linestyle": "-.", "linewidth": 1.7}


def graficar_series_temporales(tiempos, trayectoria, equilibrio=None,
                               titulo="", archivo_salida=None):
    """
    Evolucion de las tres poblaciones contra el tiempo.

    Es la figura que muestra el COMPORTAMIENTO ASINTOTICO: si el equilibrio es
    atractor, las tres curvas se aplanan en sus valores finales.

    Parametros
    ----------
    tiempos : ndarray (n,)
    trayectoria : ndarray (n, 3)   columnas [N, P, S]
    equilibrio : ndarray (3,), opcional
        Si se pasa, se dibujan lineas horizontales de referencia.
    """
    figura, eje = plt.subplots(figsize=(9, 4.5))

    eje.plot(tiempos, trayectoria[:, 0], label="Presa $N(t)$", **ESTILO_PRESA)
    eje.plot(tiempos, trayectoria[:, 1], label="Depredador $P(t)$", **ESTILO_DEPREDADOR)
    eje.plot(tiempos, trayectoria[:, 2], label="Superdepredador $S(t)$", **ESTILO_SUPERDEPREDADOR)

    if equilibrio is not None:
        for valor in equilibrio:
            eje.axhline(valor, color="gray", linestyle=":", linewidth=0.8, zorder=0)

    eje.set_xlabel("Tiempo $t$ [d]")
    eje.set_ylabel("Densidad poblacional [ind]")
    if titulo:
        eje.set_title(titulo)
    eje.legend()
    eje.grid(alpha=0.3)
    figura.tight_layout()

    if archivo_salida:
        figura.savefig(archivo_salida, dpi=RESOLUCION_DPI)
    return figura


def graficar_retrato_de_fase_3d(trayectoria, equilibrio=None,
                                titulo="", archivo_salida=None):
    """
    Retrato de fase en el espacio (N, P, S).  <-- EL GRAFICO 3D DEL INFORME

    Cada punto de la curva es un ESTADO del sistema; el tiempo no se ve, se
    recorre. Si el equilibrio atrae, la trayectoria espirala hacia el y se queda.

    COMO SE LEE:
      - converge a un PUNTO        -> equilibrio estable
      - termina en un LAZO cerrado -> ciclo limite (oscilacion sostenida)
      - llena una REGION sin cerrarse -> posible CAOS
    """
    figura = plt.figure(figsize=(7, 6))
    eje = figura.add_subplot(projection="3d")

    eje.plot(trayectoria[:, 0], trayectoria[:, 1], trayectoria[:, 2],
             color="#1f77b4", linewidth=0.9)

    eje.scatter(*trayectoria[0], color="black", s=45, marker="o",
                label="Condicion inicial")
    if equilibrio is not None:
        eje.scatter(*equilibrio, color="#d62728", s=150, marker="*",
                    label="Equilibrio")

    eje.set_xlabel("Presa $N$ [ind]")
    eje.set_ylabel("Depredador $P$ [ind]")
    eje.set_zlabel("Superdepredador $S$ [ind]")
    if titulo:
        eje.set_title(titulo)
    eje.legend()
    figura.tight_layout()

    if archivo_salida:
        figura.savefig(archivo_salida, dpi=RESOLUCION_DPI)
    return figura


def graficar_comparacion_de_metodos(tiempos_euler, trayectoria_euler,
                                    tiempos_rk2, trayectoria_rk2,
                                    indice_variable=0, nombre_variable="N",
                                    titulo="", archivo_salida=None):
    """
    Superpone Euler y RK2 para la MISMA variable y el MISMO dt.

    Muestra visualmente la diferencia de orden: con dt grande, Euler se separa
    de RK2 de forma sistematica (no al azar: siempre hacia el mismo lado).
    """
    figura, eje = plt.subplots(figsize=(9, 4.5))

    eje.plot(tiempos_euler, trayectoria_euler[:, indice_variable],
             label="Euler", color="#d62728", linestyle="--", linewidth=1.6)
    eje.plot(tiempos_rk2, trayectoria_rk2[:, indice_variable],
             label="RK2 (Heun)", color="#1f77b4", linestyle="-", linewidth=1.6)

    eje.set_xlabel("Tiempo $t$ [d]")
    eje.set_ylabel(f"${nombre_variable}$ [ind]")
    if titulo:
        eje.set_title(titulo)
    eje.legend()
    eje.grid(alpha=0.3)
    figura.tight_layout()

    if archivo_salida:
        figura.savefig(archivo_salida, dpi=RESOLUCION_DPI)
    return figura


def graficar_inestabilidad(resultados_por_paso, paso_maximo_teorico,
                           indice_variable=0, nombre_variable="N",
                           archivo_salida=None):
    """
    LA figura que pide el enunciado: que pasa al agrandar el dt.

    Parametros
    ----------
    resultados_por_paso : lista de tuplas (dt, tiempos, trayectoria)
        Una corrida por cada dt. Conviene incluir al menos uno por DEBAJO de la
        cota y uno por ENCIMA, para que se vea el contraste.
    paso_maximo_teorico : float
        La cota 2/|lambda|. Se usa para rotular cada panel.
    """
    cantidad = len(resultados_por_paso)
    figura, ejes = plt.subplots(1, cantidad, figsize=(5.0 * cantidad, 4.2))
    if cantidad == 1:
        ejes = [ejes]

    for (paso, tiempos, trayectoria), eje in zip(resultados_por_paso, ejes):
        eje.plot(tiempos, trayectoria[:, indice_variable],
                 linewidth=1.3, color="#1f77b4")
        condicion = "ESTABLE" if paso < paso_maximo_teorico else "INESTABLE"
        eje.set_title(r"$\Delta t = " + f"{paso:g}$  ({condicion})")
        eje.set_xlabel("Tiempo $t$ [d]")
        eje.grid(alpha=0.3)

    ejes[0].set_ylabel(f"${nombre_variable}$ [ind]")
    figura.tight_layout()

    if archivo_salida:
        figura.savefig(archivo_salida, dpi=RESOLUCION_DPI)
    return figura


def graficar_convergencia_loglog(pasos_euler, errores_euler,
                                 pasos_rk2, errores_rk2,
                                 archivo_salida=None):
    """
    Error vs dt en escala log-log.

    COMO SE LEE: en log-log, un error O(dt^p) se ve como una RECTA DE PENDIENTE p.
    Euler debe dar pendiente ~1 y RK2 pendiente ~2. Se dibujan rectas de
    referencia para comparar a ojo.

    Es la version visual de la tabla de convergencia, y la justificacion mas
    contundente del dt elegido.
    """
    pasos_euler = np.asarray(pasos_euler, dtype=float)
    errores_euler = np.asarray(errores_euler, dtype=float)
    pasos_rk2 = np.asarray(pasos_rk2, dtype=float)
    errores_rk2 = np.asarray(errores_rk2, dtype=float)

    figura, eje = plt.subplots(figsize=(6.5, 5))

    eje.loglog(pasos_euler, errores_euler, "o-", color="#d62728",
               label="Euler (medido)", linewidth=1.6)
    eje.loglog(pasos_rk2, errores_rk2, "s-", color="#1f77b4",
               label="RK2 (medido)", linewidth=1.6)

    # Rectas de referencia de pendiente 1 y 2, ancladas al primer punto de cada serie
    eje.loglog(pasos_euler, errores_euler[0] * (pasos_euler / pasos_euler[0]) ** 1,
               ":", color="gray", label="pendiente 1 (orden 1)")
    eje.loglog(pasos_rk2, errores_rk2[0] * (pasos_rk2 / pasos_rk2[0]) ** 2,
               "--", color="gray", label="pendiente 2 (orden 2)")

    eje.set_xlabel(r"Paso temporal $\Delta t$")
    eje.set_ylabel("Error global")
    eje.set_title("Convergencia: el orden se lee como la pendiente")
    eje.legend()
    eje.grid(alpha=0.3, which="both")
    figura.tight_layout()

    if archivo_salida:
        figura.savefig(archivo_salida, dpi=RESOLUCION_DPI)
    return figura
