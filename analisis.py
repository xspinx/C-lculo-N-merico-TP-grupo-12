"""
================================================================================
ANALISIS — equilibrio, estabilidad y eleccion del paso temporal
================================================================================
Universidad Austral - Ingenieria en Informatica - Calculo Numerico
Problema 12: Cadena trofica con respuestas funcionales mixtas

Este script resuelve los 5 pasos fundamentales:
    1. Encontrar el equilibrio de coexistencia (fsolve).
    2. Verificar que atrae (autovalores del Jacobiano con Re < 0).
    3. Cota teorica de estabilidad: dt <= 2 / |lambda_max|.
    4. Estudio empirico de convergencia en el transitorio (Euler ~2, RK2 ~4).
    5. Generacion de las 5 figuras vectoriales para el informe.
================================================================================
"""

import os
import numpy as np
from scipy.optimize import fsolve

from modelo import derivadas, jacobiano_numerico
from euler import integrar_euler
from rk2 import integrar_rk2
import figuras


# ==============================================================================
# PASO 1 — Encontrar el equilibrio de coexistencia
# ==============================================================================

def encontrar_equilibrio(semilla):
    """
    Encuentra el estado estacionario donde dN/dt = dP/dt = dS/dt = 0
    utilizando fsolve a partir de una semilla inicial.

    Verifica que el residuo sea menor a 1e-8.
    """
    sol, info, ier, mesg = fsolve(derivadas, semilla, full_output=True)
    if ier != 1:
        raise RuntimeError(f"fsolve no convergio: {mesg}")

    residuo = np.linalg.norm(derivadas(sol))
    if residuo > 1e-8:
        raise RuntimeError(f"Residuo elevado en fsolve ({residuo:.2e}): no es un equilibrio real.")

    return sol


# ==============================================================================
# PASO 2 — Verificar que el equilibrio ATRAE
# ==============================================================================

def analizar_estabilidad(equilibrio):
    """
    Calcula los autovalores de la matriz Jacobiana evaluada en el equilibrio.
    Informa si el equilibrio es atractor asintotico o repulsor.
    """
    matriz_J = jacobiano_numerico(equilibrio)
    autovalores = np.linalg.eigvals(matriz_J)
    partes_reales = np.real(autovalores)

    print("\nAutovalores de la matriz Jacobiana:")
    es_atractor = True
    for i, lam in enumerate(autovalores):
        print(f"  lambda_{i+1} = {lam.real:+.6f} {lam.imag:+.6f}j  (Re = {lam.real:+.6f})")
        if lam.real >= 0:
            es_atractor = False

    if es_atractor:
        print(">>> CONCLUSION: Todos los autovalores tienen Re(lambda) < 0.")
        print(">>> El equilibrio es ASINTOTICAMENTE ESTABLE (ATRACTOR LOCAL).")
    else:
        print(">>> ADVERTENCIA: Hay autovalores con Re(lambda) >= 0. El punto NO atrae.")

    return autovalores


# ==============================================================================
# PASO 3 — Cota teorica del paso temporal
# ==============================================================================

def paso_maximo_teorico(equilibrio):
    """
    Calcula la cota maxima de estabilidad para metodos explicitos (Euler):
        dt_max = 2.0 / max|lambda|
    """
    matriz_J = jacobiano_numerico(equilibrio)
    autovalores = np.linalg.eigvals(matriz_J)
    max_modulo = np.max(np.abs(autovalores))
    return 2.0 / max_modulo


# ==============================================================================
# PASO 4 — Estudio de convergencia (la justificacion fuerte del dt)
# ==============================================================================

def estudio_de_convergencia(estado_inicial, tiempo_final, lista_de_pasos, integrador):
    """
    Calcula el error global en el regimen transitorio (t = tiempo_final)
    comparando contra una corrida de referencia ultrafina.

    Devuelve una lista de tuplas: (dt, error, cociente_reduccion)
    """
    # 1. Corrida ultrafina de referencia
    dt_referencia = min(lista_de_pasos) / 8.0
    _, tray_ref = integrador(estado_inicial, dt_referencia, tiempo_final)
    estado_final_ref = tray_ref[-1]

    resultados = []
    error_previo = None

    for dt in lista_de_pasos:
        _, tray = integrador(estado_inicial, dt, tiempo_final)
        estado_final = tray[-1]

        # Medimos el error en la presa N en el tiempo transitorio
        error = abs(estado_final[0] - estado_final_ref[0])

        cociente = (error_previo / error) if (error_previo is not None and error > 1e-15) else None
        resultados.append((dt, error, cociente))
        error_previo = error

    return resultados


# ==============================================================================
# PASO 5 — Generar las figuras y reporte completo
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ANALISIS NUMERICO COMPLETO — PROBLEMA 12")
    print("=" * 80)

    # 1. Semilla y Equilibrio
    SEMILLA = np.array([5.0, 2.0, 1.0])
    equilibrio = encontrar_equilibrio(SEMILLA)
    print(f"\nEquilibrio de coexistencia tripartito:")
    print(f"  Presa           N* = {equilibrio[0]:.4f} ind")
    print(f"  Depredador      P* = {equilibrio[1]:.4f} ind")
    print(f"  Superdepredador S* = {equilibrio[2]:.4f} ind")
    residuo = np.linalg.norm(derivadas(equilibrio))
    print(f"  Residuo ||f(y*)||  = {residuo:.2e}")

    # 2. Estabilidad Espectral
    autovalores = analizar_estabilidad(equilibrio)

    # 3. Cota teorica
    dt_max = paso_maximo_teorico(equilibrio)
    print(f"\nCota teorica de estabilidad (Euler): dt <= {dt_max:.4f} dias")

    # 4. Estudio de convergencia en el transitorio (t = 15 dias)
    TIEMPO_TRANSITORIO = 15.0
    pasos_estudio = [0.08, 0.04, 0.02, 0.01]

    print(f"\n" + "-" * 70)
    print(f"TABLA DE CONVERGENCIA EN REGIMEN TRANSITORIO (t = {TIEMPO_TRANSITORIO} d)")
    print("-" * 70)

    res_euler = estudio_de_convergencia(SEMILLA, TIEMPO_TRANSITORIO, pasos_estudio, integrar_euler)
    res_rk2 = estudio_de_convergencia(SEMILLA, TIEMPO_TRANSITORIO, pasos_estudio, integrar_rk2)

    print(f"{'dt [d]':<10} {'Error Euler':<18} {'Cociente Euler':<16} {'Error RK2':<18} {'Cociente RK2':<16}")
    print("-" * 70)
    for (dt, err_e, coc_e), (_, err_r, coc_r) in zip(res_euler, res_rk2):
        str_coc_e = f"{coc_e:.2f}" if coc_e is not None else "-"
        str_coc_r = f"{coc_r:.2f}" if coc_r is not None else "-"
        print(f"{dt:<10.4f} {err_e:<18.6e} {str_coc_e:<16} {err_r:<18.6e} {str_coc_r:<16}")

    # 5. Generar figuras
    print(f"\n" + "-" * 70)
    print("GENERANDO FIGURAS PARA EL INFORME...")
    print("-" * 70)

    # Directorio de salida para las figuras
    dir_figuras = "figuras_salida"
    os.makedirs(dir_figuras, exist_ok=True)

    # Figura 1: Series Temporales (RK2 con dt fino, muestra comportamiento asintotico)
    t_fin_sim = 150.0
    dt_sim = 0.01
    tiempos_rk2, tray_rk2 = integrar_rk2(SEMILLA, dt_sim, t_fin_sim)
    figuras.graficar_series_temporales(
        tiempos_rk2, tray_rk2, equilibrio=equilibrio,
        titulo=r"Evolución temporal de las tres poblaciones ($\Delta t = 0{,}01\,\mathrm{d}$)",
        archivo_salida=os.path.join(dir_figuras, "figura_1_series_temporales.png")
    )
    print("  [x] Figura 1 generada: figura_1_series_temporales.png")

    # Figura 2: Retrato de Fase 3D
    figuras.graficar_retrato_de_fase_3d(
        tray_rk2, equilibrio=equilibrio,
        titulo="Retrato de fase en el espacio de estados (N, P, S)",
        archivo_salida=os.path.join(dir_figuras, "figura_2_retrato_fase_3d.png")
    )
    print("  [x] Figura 2 generada: figura_2_retrato_fase_3d.png")

    # Figura 3: Inestabilidad numerica al superar dt_max con Euler
    dt_estable = 0.05
    dt_inestable = dt_max * 1.05  # apenas 5% por encima de la cota
    t_inest, tray_est = integrar_euler(SEMILLA, dt_estable, 30.0)
    t_inest2, tray_inest = integrar_euler(SEMILLA, dt_inestable, 30.0)
    resultados_inestabilidad = [
        (dt_estable, t_inest, tray_est),
        (dt_inestable, t_inest2, tray_inest)
    ]
    figuras.graficar_inestabilidad(
        resultados_inestabilidad, dt_max,
        indice_variable=0, nombre_variable="N",
        archivo_salida=os.path.join(dir_figuras, "figura_3_inestabilidad.png")
    )
    print("  [x] Figura 3 generada: figura_3_inestabilidad.png")

    # Figura 4: Comparacion de metodos (Euler vs RK2 con el mismo dt)
    dt_comp = 0.08
    tiempos_e, tray_e = integrar_euler(SEMILLA, dt_comp, 60.0)
    tiempos_r, tray_r = integrar_rk2(SEMILLA, dt_comp, 60.0)
    figuras.graficar_comparacion_de_metodos(
        tiempos_e, tray_e, tiempos_r, tray_r,
        indice_variable=0, nombre_variable="N",
        titulo=r"Comparación Euler vs RK2 para la presa $N(t)$ ($\Delta t = 0{,}08\,\mathrm{d}$)",
        archivo_salida=os.path.join(dir_figuras, "figura_4_comparacion_metodos.png")
    )
    print("  [x] Figura 4 generada: figura_4_comparacion_metodos.png")

    # Figura 5: Convergencia Log-Log
    pasos_euler = [item[0] for item in res_euler]
    errores_euler = [item[1] for item in res_euler]
    pasos_rk2_list = [item[0] for item in res_rk2]
    errores_rk2_list = [item[1] for item in res_rk2]

    figuras.graficar_convergencia_loglog(
        pasos_euler, errores_euler,
        pasos_rk2_list, errores_rk2_list,
        archivo_salida=os.path.join(dir_figuras, "figura_5_convergencia_loglog.png")
    )
    print("  [x] Figura 5 generada: figura_5_convergencia_loglog.png")

    print("\n" + "=" * 80)
    print("¡TODO EL ANALISIS Y LAS FIGURAS FUERON COMPLETADOS CON EXITO!")
    print("=" * 80)
