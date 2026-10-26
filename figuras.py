"""
================================================================================
UTILIDADES DE GRAFICADO (EDICIÓN CALIDAD PUBLICACIÓN / LATEX)
================================================================================

Funciones reutilizables para generar las figuras del informe. Cada una recibe
los datos YA CALCULADOS y devuelve la figura; no calcula nada por su cuenta.

CONVENCIONES QUE RESPETAN (y que se evaluan en el informe):
  - ejes rotulados, con unidad cuando corresponde
  - leyenda siempre que haya mas de una curva
  - distinto ESTILO de trazo ademas del color, para que se lea en blanco y negro
  - resolucion 300 dpi (supera el estandar requerido de 200 dpi)
  - el EPIGRAFE no va en la figura: va en el LaTeX, DEBAJO de ella
================================================================================
"""

import matplotlib.pyplot as plt
import numpy as np

# 300 DPI garantiza nitidez perfecta en LaTeX (cumple y supera los 200 DPI)
RESOLUCION_DPI = 300

# Configuración global de estilo académico (armoniza con tipografía LaTeX)
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",          # Fuentes matemáticas Computer Modern (LaTeX)
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.direction": "in",           # Ticks hacia adentro (estilo paper)
    "ytick.direction": "in",
    "axes.linewidth": 0.9,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.dpi": 120,                 # Buena vista previa en pantalla
    "savefig.dpi": RESOLUCION_DPI,
    "savefig.bbox": "tight",           # Evita que se corten etiquetas al exportar
    "savefig.pad_inches": 0.08,
})

# Paleta Okabe-Ito (alto contraste en B/N y apta para daltónicos) + trazos distintos
ESTILO_PRESA = {
    "color": "#0072B2",                # Azul oscuro
    "linestyle": "-",
    "linewidth": 2.0,
    "zorder": 3
}
ESTILO_DEPREDADOR = {
    "color": "#D55E00",                # Rojo/Naranja bermellón
    "linestyle": "--",
    "linewidth": 2.0,
    "zorder": 3
}
ESTILO_SUPERDEPREDADOR = {
    "color": "#009E73",                # Verde azulado
    "linestyle": "-.",
    "linewidth": 2.0,
    "zorder": 3
}


def _estilizar_eje_2d(eje):
    """Aplica limpieza estética de publicación a un eje 2D."""
    eje.spines["top"].set_visible(False)
    eje.spines["right"].set_visible(False)
    eje.grid(True, which="major", color="#CCCCCC", linestyle="--", linewidth=0.6, zorder=0)
    eje.set_axisbelow(True)


def _estilizar_leyenda(eje, **kwargs):
    """Configura una leyenda limpia con fondo semitransparente."""
    opciones = {
        "frameon": True,
        "facecolor": "white",
        "edgecolor": "#E0E0E0",
        "framealpha": 0.92,
        "fancybox": False,
    }
    opciones.update(kwargs)
    return eje.legend(**opciones)


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
    figura, eje = plt.subplots(figsize=(8.5, 4.2))
    _estilizar_eje_2d(eje)

    eje.plot(tiempos, trayectoria[:, 0], label="Presa $N(t)$", **ESTILO_PRESA)
    eje.plot(tiempos, trayectoria[:, 1], label="Depredador $P(t)$", **ESTILO_DEPREDADOR)
    eje.plot(tiempos, trayectoria[:, 2], label="Superdepredador $S(t)$", **ESTILO_SUPERDEPREDADOR)

    if equilibrio is not None:
        colores_eq = [ESTILO_PRESA["color"], ESTILO_DEPREDADOR["color"], ESTILO_SUPERDEPREDADOR["color"]]
        for idx, (valor, col) in enumerate(zip(equilibrio, colores_eq)):
            lbl = "Equilibrio asintótico" if idx == 0 else None
            eje.axhline(valor, color=col, linestyle=":", linewidth=1.2, alpha=0.75, zorder=1, label=lbl)

    eje.set_xlabel("Tiempo $t$ [d]")
    eje.set_ylabel("Densidad poblacional [ind]")
    eje.set_xlim(left=tiempos[0], right=tiempos[-1])
    eje.margins(y=0.08)

    if titulo:
        eje.set_title(titulo, pad=10, fontweight="bold")
    _estilizar_leyenda(eje, loc="best")
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
    """
    figura = plt.figure(figsize=(7.5, 6.2))
    eje = figura.add_subplot(projection="3d")

    # Paneles de fondo más limpios y sutiles para impresión
    for pane in (eje.xaxis.pane, eje.yaxis.pane, eje.zaxis.pane):
        pane.fill = True
        pane.set_facecolor((0.97, 0.97, 0.97, 0.6))
        pane.set_edgecolor("#CCCCCC")

    # Proyección (sombra) en el piso para dar noción de profundidad en papel 2D
    z_piso = np.min(trayectoria[:, 2]) - 0.06 * np.ptp(trayectoria[:, 2])
    eje.plot(trayectoria[:, 0], trayectoria[:, 1],
             np.full_like(trayectoria[:, 2], z_piso),
             color="gray", linestyle=":", linewidth=0.8, alpha=0.45,
             label="Proyección en plano $(N, P)$")

    # Trayectoria principal 3D
    eje.plot(trayectoria[:, 0], trayectoria[:, 1], trayectoria[:, 2],
             color="#0072B2", linewidth=1.5, alpha=0.9, label="Trayectoria $(N, P, S)$")

    # Punto inicial
    eje.scatter(*trayectoria[0], color="#111111", edgecolor="white",
                s=65, marker="o", depthshade=False, zorder=5,
                label="Condición inicial")

    # Punto de equilibrio
    if equilibrio is not None:
        eje.scatter(*equilibrio, color="#D55E00", edgecolor="black", linewidth=0.6,
                    s=180, marker="*", depthshade=False, zorder=6,
                    label="Punto de equilibrio")

    eje.set_xlabel("Presa $N$ [ind]", labelpad=9)
    eje.set_ylabel("Depredador $P$ [ind]", labelpad=9)
    eje.set_zlabel("Superdepredador $S$ [ind]", labelpad=9)
    eje.set_zlim(bottom=z_piso)

    # Ángulo de vista óptimo para distinguir espirales
    eje.view_init(elev=24, azim=-45)

    if titulo:
        eje.set_title(titulo, pad=12, fontweight="bold")
    _estilizar_leyenda(eje, loc="upper left")
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
    figura, eje = plt.subplots(figsize=(8.5, 4.2))
    _estilizar_eje_2d(eje)

    # RK2 como referencia sólida y Euler en trazo discontinuo encima
    eje.plot(tiempos_rk2, trayectoria_rk2[:, indice_variable],
             label="RK2 (Heun - Orden 2)", color="#0072B2",
             linestyle="-", linewidth=2.0, zorder=3)
    eje.plot(tiempos_euler, trayectoria_euler[:, indice_variable],
             label="Euler explícito (Orden 1)", color="#D55E00",
             linestyle="--", linewidth=1.8, zorder=4)

    eje.set_xlabel("Tiempo $t$ [d]")
    eje.set_ylabel(f"Población ${nombre_variable}(t)$ [ind]")
    eje.set_xlim(left=min(tiempos_euler[0], tiempos_rk2[0]),
                 right=max(tiempos_euler[-1], tiempos_rk2[-1]))
    eje.margins(y=0.08)

    if titulo:
        eje.set_title(titulo, pad=10, fontweight="bold")
    _estilizar_leyenda(eje, loc="best")
    figura.tight_layout()

    if archivo_salida:
        figura.savefig(archivo_salida, dpi=RESOLUCION_DPI)
    return figura


def graficar_inestabilidad(resultados_por_paso, paso_maximo_teorico,
                           indice_variable=0, nombre_variable="N",
                           archivo_salida=None):
    """
    LA figura que pide el enunciado: que pasa al agrandar el dt.
    """
    cantidad = len(resultados_por_paso)
    figura, ejes = plt.subplots(1, cantidad, figsize=(4.8 * cantidad, 4.0), squeeze=False)
    ejes = ejes[0]

    for (paso, tiempos, trayectoria), eje in zip(resultados_por_paso, ejes):
        _estilizar_eje_2d(eje)
        es_estable = paso < paso_maximo_teorico

        condicion = "ESTABLE" if es_estable else "INESTABLE"
        color_curva = "#0072B2" if es_estable else "#D55E00"
        estilo_linea = "-" if es_estable else "-."

        eje.plot(tiempos, trayectoria[:, indice_variable],
                 linewidth=1.6, color=color_curva, linestyle=estilo_linea,
                 label=rf"$\Delta t = {paso:g}$ d")

        # Si es inestable, sombrear muy sutilmente el fondo para enfatizar la divergencia
        if not es_estable:
            eje.set_facecolor("#FFF8F6")

        eje.set_title(
            rf"$\Delta t = {paso:g}$ d  [{condicion}]" + "\n" + rf"($\Delta t_{{máx}} \approx {paso_maximo_teorico:.3g}$ d)",
            fontsize=10.5,
            fontweight="bold" if not es_estable else "normal",
            color="#900C3F" if not es_estable else "black",
            pad=8
        )
        eje.set_xlabel("Tiempo $t$ [d]")
        eje.set_xlim(left=tiempos[0], right=tiempos[-1])
        _estilizar_leyenda(eje, loc="best")

    ejes[0].set_ylabel(f"Población ${nombre_variable}(t)$ [ind]")
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
    Euler debe dar pendiente ~1 y RK2 pendiente ~2.
    """
    pasos_euler = np.asarray(pasos_euler, dtype=float)
    errores_euler = np.asarray(errores_euler, dtype=float)
    pasos_rk2 = np.asarray(pasos_rk2, dtype=float)
    errores_rk2 = np.asarray(errores_rk2, dtype=float)

    figura, eje = plt.subplots(figsize=(6.8, 5.0))
    _estilizar_eje_2d(eje)

    # Grilla log-log diferenciada (principal y secundaria)
    eje.grid(True, which="major", color="#BBBBBB", linestyle="--", linewidth=0.7)
    eje.grid(True, which="minor", color="#E5E5E5", linestyle=":", linewidth=0.5)

    # Curvas medidas con marcadores destacados
    eje.loglog(pasos_euler, errores_euler, "o-", color="#D55E00",
               linewidth=2.0, markersize=6.5, markeredgecolor="white",
               markeredgewidth=0.8, label="Euler explícito (medido)", zorder=4)
    eje.loglog(pasos_rk2, errores_rk2, "s-", color="#0072B2",
               linewidth=2.0, markersize=6.5, markeredgecolor="white",
               markeredgewidth=0.8, label="RK2 Heun (medido)", zorder=4)

    # Rectas teóricas de referencia desplazadas un 15% para no tapar los datos
    ref_euler = (errores_euler[0] * 0.85) * (pasos_euler / pasos_euler[0]) ** 1
    ref_rk2 = (errores_rk2[0] * 0.85) * (pasos_rk2 / pasos_rk2[0]) ** 2

    eje.loglog(pasos_euler, ref_euler, ":", color="#444444", linewidth=1.5,
               label=r"Referencia $\mathcal{O}(\Delta t^1)$ (pendiente 1)", zorder=2)
    eje.loglog(pasos_rk2, ref_rk2, "--", color="#666666", linewidth=1.5,
               label=r"Referencia $\mathcal{O}(\Delta t^2)$ (pendiente 2)", zorder=2)

    eje.set_xlabel(r"Paso de integración $\Delta t$ [d]")
    eje.set_ylabel(r"Error global máximo $E(\Delta t)$")
    eje.set_title("Análisis de convergencia y orden numérico", pad=10, fontweight="bold")
    _estilizar_leyenda(eje, loc="upper left")
    figura.tight_layout()

    if archivo_salida:
        figura.savefig(archivo_salida, dpi=RESOLUCION_DPI)
    return figura