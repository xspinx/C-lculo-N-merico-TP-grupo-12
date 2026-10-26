"""
================================================================================
EJEMPLO DE REFERENCIA — Modelo presa-depredador con respuesta Holling tipo II
================================================================================

ESTO NO ES EL TP. Es el modelo de DOS especies (el Lotka-Volterra extendido que
figura en el material de la catedra) resuelto completo, para que se vea el patron.
El TP (Problema 12) tiene TRES especies y tres respuestas funcionales distintas:
hay que portar esta misma estructura agregando una ecuacion mas.

MODELO
------
    dN/dt = r*N*(1 - N/K)  -  F_PN(N) * P          (presa, crecimiento logistico)
    dP/dt = b * F_PN(N) * P  -  m*P                 (depredador)

    F_PN(N) = a*N / (1 + a*h*N)                     (Holling tipo II, saturante)

METODOS
-------
    - Euler explicito          : orden 1, error global O(dt)
    - Runge-Kutta 2 (Heun)     : orden 2, error global O(dt^2)
      Tambien llamado "Euler modificado" o "metodo del predictor-corrector".

OJO: el enunciado del TP pide UN ARCHIVO DIFERENTE POR METODO. Aca van juntos
solo porque es material de estudio.

Autor: material de apoyo
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# 1. PARAMETROS DEL MODELO
# ==============================================================================
# Cada parametro tiene un significado biologico. En el informe hay que justificar
# por que se eligio cada valor; no alcanza con decir "los elegimos asi".

tasa_crecimiento_presa      = 1.0    # r  [1/tiempo]  natalidad neta de la presa
capacidad_de_carga          = 10.0   # K  [individuos] techo del ambiente
tasa_de_ataque              = 0.1    # a  [1/(ind*tiempo)] eficacia de busqueda
tiempo_de_manipulacion      = 0.2    # h  [tiempo/ind] lo que tarda en comer una presa
eficiencia_de_conversion    = 0.5    # b  [adimensional] presa consumida -> depredador nuevo
tasa_mortalidad_depredador  = 0.2    # m  [1/tiempo] muerte natural del depredador


def respuesta_funcional_holling_II(densidad_presa):
    """
    Consumo per capita de un depredador en funcion de cuanta presa hay.

    Holling tipo II: crece casi lineal cuando hay poca presa, pero se SATURA
    cuando hay mucha. El techo es 1/h, porque por mas presa que haya el
    depredador necesita h unidades de tiempo para manipular cada una.

    Es la diferencia clave con Holling tipo III (el que usa el superdepredador
    en el Problema 12): el tipo III lleva N^2 en el numerador, asi que arranca
    PLANO en densidades bajas. Eso le da a la presa un refugio cuando es escasa,
    y es lo que suele estabilizar al sistema.
    """
    return (tasa_de_ataque * densidad_presa) / (1.0 + tasa_de_ataque * tiempo_de_manipulacion * densidad_presa)


def derivadas(estado):
    """
    Lado derecho del sistema de EDOs: recibe el estado y devuelve dy/dt.

    Parametros
    ----------
    estado : ndarray de shape (2,)
        [N, P] = [densidad de presa, densidad de depredador]

    Devuelve
    --------
    ndarray de shape (2,) con [dN/dt, dP/dt]

    Nota: el sistema es AUTONOMO (no depende explicitamente de t), por eso la
    funcion no recibe el tiempo. En el TP pasa lo mismo.
    """
    densidad_presa, densidad_depredador = estado

    consumo_per_capita = respuesta_funcional_holling_II(densidad_presa)

    # Presa: crece logisticamente, pierde individuos por depredacion
    crecimiento_logistico = tasa_crecimiento_presa * densidad_presa * (1.0 - densidad_presa / capacidad_de_carga)
    perdida_por_depredacion = consumo_per_capita * densidad_depredador
    variacion_presa = crecimiento_logistico - perdida_por_depredacion

    # Depredador: gana biomasa de lo que come, pierde por mortalidad natural
    ganancia_por_consumo = eficiencia_de_conversion * consumo_per_capita * densidad_depredador
    perdida_por_mortalidad = tasa_mortalidad_depredador * densidad_depredador
    variacion_depredador = ganancia_por_consumo - perdida_por_mortalidad

    return np.array([variacion_presa, variacion_depredador])


# ==============================================================================
# 2. PUNTO DE EQUILIBRIO (analitico)
# ==============================================================================
# El equilibrio es donde dN/dt = dP/dt = 0: el sistema deja de cambiar.
# Saberlo de antemano sirve para DOS cosas:
#   (a) validar el codigo -> la solucion numerica tiene que converger ahi
#   (b) elegir parametros que den un equilibrio biologicamente razonable
#       (ambas especies positivas, y lejos de la zona de caos)

def equilibrio_analitico():
    """
    Despeje a mano del equilibrio de coexistencia.

    De dP/dt = 0 con P != 0:
        b * F(N) = m
        b*a*N / (1 + a*h*N) = m
        N* = m / (a * (b - m*h))

    De dN/dt = 0 con N != 0, reemplazando N*:
        P* = r * (1 - N*/K) * (1 + a*h*N*) / a
    """
    presa_eq = tasa_mortalidad_depredador / (
        tasa_de_ataque * (eficiencia_de_conversion - tasa_mortalidad_depredador * tiempo_de_manipulacion)
    )
    depredador_eq = (
        tasa_crecimiento_presa
        * (1.0 - presa_eq / capacidad_de_carga)
        * (1.0 + tasa_de_ataque * tiempo_de_manipulacion * presa_eq)
        / tasa_de_ataque
    )
    return np.array([presa_eq, depredador_eq])


# ==============================================================================
# 3. METODO DE EULER EXPLICITO  (orden 1)
# ==============================================================================

def integrar_euler(estado_inicial, paso_temporal, tiempo_final):
    """
    Euler explicito (tambien "Euler hacia adelante").

        y_{n+1} = y_n + dt * f(y_n)

    IDEA: aproximar la curva por su recta tangente en el punto actual y avanzar
    un paso sobre esa recta. Es el metodo mas simple que existe.

    ORDEN: el error local por paso es O(dt^2) y el error global acumulado es
    O(dt). Si se parte dt a la mitad, el error se parte a la mitad.

    LIMITACION: usa la pendiente SOLO al inicio del intervalo. Si la pendiente
    cambia durante el paso (que es lo normal), el metodo se queda corto o se
    pasa sistematicamente.
    """
    cantidad_de_pasos = int(round(tiempo_final / paso_temporal))

    tiempos = np.linspace(0.0, tiempo_final, cantidad_de_pasos + 1)
    trayectoria = np.zeros((cantidad_de_pasos + 1, len(estado_inicial)))
    trayectoria[0] = estado_inicial

    for indice_paso in range(cantidad_de_pasos):
        estado_actual = trayectoria[indice_paso]
        pendiente = derivadas(estado_actual)
        trayectoria[indice_paso + 1] = estado_actual + paso_temporal * pendiente

    return tiempos, trayectoria


# ==============================================================================
# 4. METODO DE RUNGE-KUTTA 2 / EULER MODIFICADO  (orden 2)
# ==============================================================================

def integrar_rk2_heun(estado_inicial, paso_temporal, tiempo_final):
    """
    Runge-Kutta de 2do orden, variante de Heun (= "Euler modificado").

        k1 = f(y_n)                      <- pendiente al INICIO del intervalo
        k2 = f(y_n + dt*k1)              <- pendiente al FINAL (usando un predictor)
        y_{n+1} = y_n + (dt/2) * (k1 + k2)   <- se avanza con el PROMEDIO

    IDEA (la que hay que poder explicar en el oral): Euler usa una sola pendiente
    y se equivoca sistematicamente. Heun hace una pasada de prueba con Euler para
    espiar como seria la pendiente al final del intervalo, y despues avanza con
    el promedio de las dos. Es la regla del trapecio aplicada a la integracion
    temporal.

    ORDEN: error global O(dt^2). Si se parte dt a la mitad, el error se divide
    por CUATRO. Eso es lo que se verifica en el estudio de convergencia.

    COSTO: dos evaluaciones de f por paso en vez de una. Pero para igual
    precision permite un dt mucho mas grande, asi que conviene igual.
    """
    cantidad_de_pasos = int(round(tiempo_final / paso_temporal))

    tiempos = np.linspace(0.0, tiempo_final, cantidad_de_pasos + 1)
    trayectoria = np.zeros((cantidad_de_pasos + 1, len(estado_inicial)))
    trayectoria[0] = estado_inicial

    for indice_paso in range(cantidad_de_pasos):
        estado_actual = trayectoria[indice_paso]

        pendiente_inicial = derivadas(estado_actual)                                  # k1
        estado_predictor = estado_actual + paso_temporal * pendiente_inicial          # paso de Euler
        pendiente_final = derivadas(estado_predictor)                                 # k2

        pendiente_promedio = 0.5 * (pendiente_inicial + pendiente_final)
        trayectoria[indice_paso + 1] = estado_actual + paso_temporal * pendiente_promedio

    return tiempos, trayectoria


# ==============================================================================
# 5. ANALISIS DE ESTABILIDAD: POR QUE NO SE PUEDE ELEGIR CUALQUIER dt
# ==============================================================================

def jacobiano_numerico(estado, perturbacion=1e-6):
    """
    Matriz jacobiana J = df/dy calculada por diferencias finitas centradas.

    Sirve para estimar los autovalores del sistema linealizado alrededor del
    equilibrio. Esos autovalores son los que fijan el dt maximo admisible.
    """
    dimension = len(estado)
    matriz_jacobiana = np.zeros((dimension, dimension))

    for columna in range(dimension):
        desplazamiento = np.zeros(dimension)
        desplazamiento[columna] = perturbacion
        derivada_parcial = (derivadas(estado + desplazamiento) - derivadas(estado - desplazamiento)) / (2.0 * perturbacion)
        matriz_jacobiana[:, columna] = derivada_parcial

    return matriz_jacobiana


def paso_maximo_teorico(estado_equilibrio):
    """
    Cota teorica del paso temporal para Euler explicito.

    FUNDAMENTO: cerca del equilibrio el sistema se comporta como y' = J*y.
    Euler avanza y_{n+1} = (I + dt*J) * y_n, asi que cada modo se multiplica
    por el FACTOR DE AMPLIFICACION:

        g(lambda) = 1 + lambda*dt

    El metodo es estable si |g| <= 1 para todos los autovalores lambda. Para
    lambda real negativo eso da:

        dt <= 2 / |lambda_max|

    Si se supera esa cota, |g| > 1 y el error se MULTIPLICA en cada paso: la
    solucion oscila con amplitud creciente y termina desbordando. Eso es
    INESTABILIDAD NUMERICA, no un bug.

    Para RK2 el factor es g = 1 + z + z^2/2 (con z = lambda*dt) y la region de
    estabilidad es mas amplia en el plano complejo, aunque sobre el eje real
    da la misma cota.

    ATENCION (vale para el informe): estabilidad NO es precision. Un dt que
    respeta esta cota puede dar una trayectoria completamente deformada. La
    cota dice "no explota", no dice "esta bien".
    """
    matriz_jacobiana = jacobiano_numerico(estado_equilibrio)
    autovalores = np.linalg.eigvals(matriz_jacobiana)
    radio_espectral = np.max(np.abs(autovalores))
    return 2.0 / radio_espectral, autovalores


def estudio_de_convergencia(estado_inicial, tiempo_final, lista_de_pasos, integrador):
    """
    Verificacion EMPIRICA del orden del metodo. Es la justificacion mas fuerte
    que se puede poner en el informe para defender el dt elegido.

    PROCEDIMIENTO: se integra con dt, dt/2, dt/4... y se compara el estado final
    contra la corrida mas fina (que se toma como referencia). Si el metodo es de
    orden p, al partir dt a la mitad el error se divide por 2^p.

        Euler (p=1) -> el error se divide por 2
        RK2   (p=2) -> el error se divide por 4

    Cuando la solucion deja de cambiar al refinar, se convergio: ese dt sirve.
    """
    paso_mas_fino = min(lista_de_pasos) / 4.0
    _, trayectoria_referencia = integrador(estado_inicial, paso_mas_fino, tiempo_final)
    estado_referencia = trayectoria_referencia[-1]

    filas = []
    error_anterior = None
    for paso in sorted(lista_de_pasos, reverse=True):
        _, trayectoria = integrador(estado_inicial, paso, tiempo_final)
        error = np.linalg.norm(trayectoria[-1] - estado_referencia)
        cociente = (error_anterior / error) if (error_anterior is not None and error > 0) else np.nan
        filas.append((paso, error, cociente))
        error_anterior = error

    return filas


# ==============================================================================
# 6. PROGRAMA PRINCIPAL
# ==============================================================================

def main():
    estado_inicial = np.array([2.0, 1.0])   # [N0, P0]
    tiempo_final = 200.0

    # --- Equilibrio y cota de dt -------------------------------------------
    estado_equilibrio = equilibrio_analitico()
    dt_maximo, autovalores = paso_maximo_teorico(estado_equilibrio)

    print("=" * 70)
    print("EQUILIBRIO Y ESTABILIDAD")
    print("=" * 70)
    print(f"Equilibrio analitico   : N* = {estado_equilibrio[0]:.6f}   P* = {estado_equilibrio[1]:.6f}")
    print(f"Autovalores del jacobiano: {autovalores}")
    print(f"Radio espectral          : {np.max(np.abs(autovalores)):.6f}")
    print(f"dt maximo (Euler)        : {dt_maximo:.6f}")
    print()
    print("Parte real de los autovalores < 0  ->  el equilibrio es ATRACTOR:")
    print("la solucion se acerca asintoticamente. Si fueran complejos con parte")
    print("real negativa, se acerca en espiral (oscilaciones amortiguadas).")
    print()

    # --- Convergencia -------------------------------------------------------
    # OJO: el estudio de convergencia debe medirse en el TRANSITORIO, no en el
    # equilibrio. Si se integra hasta t=200 el sistema ya llego al equilibrio y
    # TODOS los dt dan el mismo resultado: el error da ~1e-14 (ruido de maquina)
    # y los cocientes no significan nada. Hay que cortar mientras la solucion
    # todavia se esta moviendo.
    tiempo_para_convergencia = 20.0
    pasos_a_probar = [0.4, 0.2, 0.1, 0.05]

    for nombre_metodo, integrador, orden_esperado in [
        ("EULER", integrar_euler, 2.0),
        ("RK2 (Heun)", integrar_rk2_heun, 4.0),
    ]:
        print("=" * 70)
        print(f"ESTUDIO DE CONVERGENCIA — {nombre_metodo}  (cociente esperado ~ {orden_esperado:.0f})")
        print("=" * 70)
        print(f"{'dt':>10} {'error':>16} {'cociente':>12}")
        for paso, error, cociente in estudio_de_convergencia(
            estado_inicial, tiempo_para_convergencia, pasos_a_probar, integrador
        ):
            texto_cociente = "---" if np.isnan(cociente) else f"{cociente:.2f}"
            print(f"{paso:>10.4f} {error:>16.3e} {texto_cociente:>12}")
        print()

    # --- Figura 1: series temporales ---------------------------------------
    paso_fino = 0.01
    tiempos, trayectoria = integrar_rk2_heun(estado_inicial, paso_fino, tiempo_final)

    figura, ejes = plt.subplots(figsize=(9, 4.5))
    ejes.plot(tiempos, trayectoria[:, 0], label="Presa $N(t)$", linewidth=1.6)
    ejes.plot(tiempos, trayectoria[:, 1], label="Depredador $P(t)$", linewidth=1.6)
    ejes.axhline(estado_equilibrio[0], linestyle="--", linewidth=0.9, color="gray")
    ejes.axhline(estado_equilibrio[1], linestyle=":", linewidth=0.9, color="gray")
    ejes.set_xlabel("Tiempo $t$")
    ejes.set_ylabel("Densidad poblacional")
    ejes.set_title(f"Evolucion temporal — RK2, $\\Delta t = {paso_fino}$")
    ejes.legend()
    ejes.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig("fig1_series_temporales.png", dpi=200)

    # --- Figura 2: inestabilidad numerica ----------------------------------
    # ESTA es la figura que pide el enunciado: mostrar que con dt grande el
    # metodo NO converge y el sistema se vuelve numericamente inestable.
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    for paso_elegido, eje in zip([0.05, dt_maximo * 1.3], ejes):
        tiempos_p, trayectoria_p = integrar_euler(estado_inicial, paso_elegido, 60.0)
        eje.plot(tiempos_p, trayectoria_p[:, 0], label="$N(t)$", linewidth=1.3)
        eje.plot(tiempos_p, trayectoria_p[:, 1], label="$P(t)$", linewidth=1.3)
        condicion = "ESTABLE" if paso_elegido < dt_maximo else "INESTABLE"
        eje.set_title(f"Euler, $\\Delta t = {paso_elegido:.3f}$  ({condicion})")
        eje.set_xlabel("Tiempo $t$")
        eje.grid(alpha=0.3)
        eje.legend()
    ejes[0].set_ylabel("Densidad poblacional")
    figura.tight_layout()
    figura.savefig("fig2_inestabilidad.png", dpi=200)

    # --- Figura 3: retrato de fase -----------------------------------------
    # Con 2 especies el retrato de fase es 2D. En el TP, con 3 especies, este
    # mismo grafico es 3D: ax = figura.add_subplot(projection="3d") y
    # ax.plot(N, P, S). Es el grafico 3D que hay que mostrar.
    figura, ejes = plt.subplots(figsize=(5.5, 5))
    ejes.plot(trayectoria[:, 0], trayectoria[:, 1], linewidth=1.0)
    ejes.plot(estado_inicial[0], estado_inicial[1], "o", label="Inicio")
    ejes.plot(estado_equilibrio[0], estado_equilibrio[1], "*", markersize=14, label="Equilibrio")
    ejes.set_xlabel("Presa $N$")
    ejes.set_ylabel("Depredador $P$")
    ejes.set_title("Retrato de fase")
    ejes.legend()
    ejes.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig("fig3_retrato_de_fase.png", dpi=200)

    print("Figuras guardadas: fig1_series_temporales.png, fig2_inestabilidad.png, fig3_retrato_de_fase.png")


if __name__ == "__main__":
    main()
