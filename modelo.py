"""
================================================================================
MODELO — Problema 12: cadena trofica de tres niveles con respuestas mixtas
================================================================================

Este modulo contiene SOLO el modelo (las ecuaciones y los parametros).
Los metodos de integracion van en archivos separados (euler.py y rk2.py),
como pide el enunciado.

SISTEMA DE ECUACIONES (transcripcion directa del enunciado)
-----------------------------------------------------------
    dN/dt = r*N*(1 - N/K) - d*N - F_PN*P - F_SN*S
    dP/dt = b*F_PN*P - m*P - F_SP*S
    dS/dt = f*F_SN*S + g*F_SP*S - q*S

    N = presa          P = depredador          S = superdepredador

RESPUESTAS FUNCIONALES ("mixtas": de ahi el nombre del problema)
-----------------------------------------------------------------
    F_PN = a*N   / (1 + a*h1*N)        Holling tipo II   (depredador come presa)
    F_SN = c*N^2 / (1 + c*h2*N^2)      Holling tipo III  (superdep. come presa)
    F_SP = e*P   / (1 + e*h3*P)        Holling tipo II   (superdep. come depredador)
================================================================================
"""

import numpy as np


# ==============================================================================
# PARAMETROS
# ==============================================================================
# ATENCION: estos valores son un PUNTO DE PARTIDA para probar que el codigo
# corre. Hay que ajustarlos y JUSTIFICAR cada uno en el informe (el enunciado
# lo pide explicitamente). Criterios para elegirlos:
#
#   1. Respetar los intervalos tipicos que da el enunciado.
#   2. Que el equilibrio de coexistencia tenga las TRES especies POSITIVAS.
#      Si alguna da negativa, esa especie se extingue y el modelo pierde gracia.
#   3. Evitar la zona de caos que el enunciado advierte. Senal de alarma: la
#      trayectoria nunca se asienta y el retrato de fase llena una region en
#      vez de converger a un punto o a un ciclo.
#   4. Que las escalas de tiempo no sean demasiado dispares entre si, porque
#      eso vuelve al sistema rigido y obliga a un dt chiquisimo.

# --- Presa --------------------------------------------------------------------
TASA_NATALIDAD_PRESA        = 1.0    # r  [1/dia]   natalidad de la presa
CAPACIDAD_DE_CARGA          = 10.0   # K  [ind]     techo que impone el ambiente
TASA_MORTALIDAD_PRESA       = 0.1    # d  [1/dia]   mortalidad natural de la presa

# --- Depredador ---------------------------------------------------------------
EFICIENCIA_PRESA_A_DEPREDADOR = 0.7  # b  [---]     cuanta presa se convierte en depredador
TASA_MORTALIDAD_DEPREDADOR    = 0.15 # m  [1/dia]   mortalidad natural del depredador

# --- Superdepredador ----------------------------------------------------------
EFICIENCIA_PRESA_A_SUPER      = 0.2  # f  [---]     conversion presa -> superdepredador
EFICIENCIA_DEPREDADOR_A_SUPER = 0.5  # g  [---]     conversion depredador -> superdepredador
TASA_MORTALIDAD_SUPER         = 0.15 # q  [1/dia]   mortalidad natural del superdepredador

# --- Tasas de ataque (intervalo tipico del enunciado: 0,01 a 0,5) -------------
ATAQUE_DEPREDADOR_A_PRESA   = 0.20   # a
ATAQUE_SUPER_A_PRESA        = 0.03   # c
ATAQUE_SUPER_A_DEPREDADOR   = 0.10   # e

# --- Tiempos de manipulacion (intervalo tipico: 0,1 a 2) ----------------------
MANIPULACION_PRESA_POR_DEPREDADOR = 0.6   # h1 [dia/ind]
MANIPULACION_PRESA_POR_SUPER      = 0.6   # h2 [dia/ind]
MANIPULACION_DEPREDADOR_POR_SUPER = 0.6   # h3 [dia/ind]



# ==============================================================================
# RESPUESTAS FUNCIONALES
# ==============================================================================
# "Respuesta funcional" = cuanta presa consume UN depredador por unidad de
# tiempo, en funcion de cuanta presa hay disponible.

def respuesta_depredador_sobre_presa(densidad_presa):
    """
    F_PN — Holling tipo II.

        F_PN(N) = a*N / (1 + a*h1*N)

    FORMA: crece casi lineal cuando hay poca presa, y se SATURA cuando hay
    mucha. El techo es 1/h1, porque por mas presa que haya el depredador
    necesita h1 unidades de tiempo para manipular cada una. Es la curva de
    un depredador que se llena.

    EFECTO SOBRE LA ESTABILIDAD: desestabiliza. Al saturarse, la mortalidad
    per capita de la presa CAE cuando la presa abunda, lo que realimenta el
    crecimiento y favorece las oscilaciones (paradoja del enriquecimiento).
    """
    a = ATAQUE_DEPREDADOR_A_PRESA
    h1 = MANIPULACION_PRESA_POR_DEPREDADOR
    return (a * densidad_presa) / (1.0 + a * h1 * densidad_presa)


def respuesta_super_sobre_presa(densidad_presa):
    """
    F_SN — Holling tipo III.  <-- LA DIFERENCIA CLAVE DE ESTE PROBLEMA

        F_SN(N) = c*N^2 / (1 + c*h2*N^2)

    FORMA: sigmoidea. El N^2 del numerador hace que la curva arranque PLANA:
    a densidad baja de presa el superdepredador casi no la consume. Recien
    cuando la presa abunda el consumo despega, y despues satura igual que el
    tipo II.

    INTERPRETACION BIOLOGICA: modela aprendizaje o cambio de dieta. Si la
    presa es escasa, al superdepredador no le rinde buscarla y se dedica a
    otra cosa (en este modelo, al depredador P).

    EFECTO SOBRE LA ESTABILIDAD: estabiliza. Ese arranque plano le da a la
    presa un REFUGIO a densidades bajas: cuando esta por extinguirse deja de
    ser perseguida y puede recuperarse. Es el mecanismo que evita que el
    sistema colapse.

    >>> Si en el oral preguntan "por que tipo III aca y tipo II alla", esta
    >>> es la respuesta: el refugio a densidades bajas.
    """
    c = ATAQUE_SUPER_A_PRESA
    h2 = MANIPULACION_PRESA_POR_SUPER
    presa_al_cuadrado = densidad_presa ** 2
    return (c * presa_al_cuadrado) / (1.0 + c * h2 * presa_al_cuadrado)


def respuesta_super_sobre_depredador(densidad_depredador):
    """
    F_SP — Holling tipo II.

        F_SP(P) = e*P / (1 + e*h3*P)

    Misma forma que F_PN pero sobre el depredador: el superdepredador tambien
    se satura comiendo P. Es el termino que convierte a S en un depredador
    APICE que se alimenta de DOS niveles troficos a la vez (de N y de P).

    Esa doble via de alimentacion ("omnivoria intragremial") es lo que vuelve
    al modelo HIBRIDO, y es tambien la fuente habitual del caos que advierte
    el enunciado.
    """
    e = ATAQUE_SUPER_A_DEPREDADOR
    h3 = MANIPULACION_DEPREDADOR_POR_SUPER
    return (e * densidad_depredador) / (1.0 + e * h3 * densidad_depredador)


# ==============================================================================
# LADO DERECHO DEL SISTEMA
# ==============================================================================

def derivadas(estado):
    """
    Evalua el lado derecho del sistema de EDOs: recibe el estado y devuelve dy/dt.

    Parametros
    ----------
    estado : ndarray de shape (3,)
        [N, P, S] = [presa, depredador, superdepredador]

    Devuelve
    --------
    ndarray de shape (3,) con [dN/dt, dP/dt, dS/dt]

    Nota: el sistema es AUTONOMO (no depende explicitamente del tiempo), por eso
    la funcion no recibe t. Los integradores igual la llaman igual.
    """
    densidad_presa, densidad_depredador, densidad_superdepredador = estado

    # --- Respuestas funcionales evaluadas en el estado actual -----------------
    consumo_depredador_sobre_presa = respuesta_depredador_sobre_presa(densidad_presa)
    consumo_super_sobre_presa      = respuesta_super_sobre_presa(densidad_presa)
    consumo_super_sobre_depredador = respuesta_super_sobre_depredador(densidad_depredador)

    # --- ECUACION 1: la presa -------------------------------------------------
    # Crece logisticamente (frenada por la capacidad de carga), muere sola, y
    # la comen DOS depredadores distintos: P y S.
    crecimiento_logistico = (
        TASA_NATALIDAD_PRESA * densidad_presa
        * (1.0 - densidad_presa / CAPACIDAD_DE_CARGA)
    )
    muerte_natural_presa = TASA_MORTALIDAD_PRESA * densidad_presa
    depredacion_por_depredador = consumo_depredador_sobre_presa * densidad_depredador
    depredacion_por_super      = consumo_super_sobre_presa * densidad_superdepredador

    variacion_presa = (
        crecimiento_logistico
        - muerte_natural_presa
        - depredacion_por_depredador
        - depredacion_por_super
    )

    # --- ECUACION 2: el depredador --------------------------------------------
    # Gana biomasa comiendo presa (con eficiencia b), muere solo, y ademas se
    # lo come el superdepredador.
    ganancia_comiendo_presa = (
        EFICIENCIA_PRESA_A_DEPREDADOR * consumo_depredador_sobre_presa * densidad_depredador
    )
    muerte_natural_depredador = TASA_MORTALIDAD_DEPREDADOR * densidad_depredador
    perdida_por_super = consumo_super_sobre_depredador * densidad_superdepredador

    variacion_depredador = (
        ganancia_comiendo_presa
        - muerte_natural_depredador
        - perdida_por_super
    )

    # --- ECUACION 3: el superdepredador ---------------------------------------
    # Come de los DOS niveles de abajo (presa y depredador) y muere solo.
    # Nadie se lo come: es el depredador apice.
    ganancia_comiendo_presa_super = (
        EFICIENCIA_PRESA_A_SUPER * consumo_super_sobre_presa * densidad_superdepredador
    )
    ganancia_comiendo_depredador = (
        EFICIENCIA_DEPREDADOR_A_SUPER * consumo_super_sobre_depredador * densidad_superdepredador
    )
    muerte_natural_super = TASA_MORTALIDAD_SUPER * densidad_superdepredador

    variacion_superdepredador = (
        ganancia_comiendo_presa_super
        + ganancia_comiendo_depredador
        - muerte_natural_super
    )

    return np.array([variacion_presa, variacion_depredador, variacion_superdepredador])


# ==============================================================================
# UTILIDAD: jacobiano numerico (para el analisis de estabilidad del dt)
# ==============================================================================

def jacobiano_numerico(estado, perturbacion=1e-6):
    """
    Matriz J = df/dy por diferencias finitas centradas.

    Para que sirve: cerca de un equilibrio el sistema se comporta como
    y' = J*y. Los autovalores de J fijan (a) si el equilibrio atrae o repele
    y (b) cual es el dt maximo que tolera un metodo explicito.
    """
    dimension = len(estado)
    matriz = np.zeros((dimension, dimension))
    for columna in range(dimension):
        desplazamiento = np.zeros(dimension)
        desplazamiento[columna] = perturbacion
        matriz[:, columna] = (
            derivadas(estado + desplazamiento) - derivadas(estado - desplazamiento)
        ) / (2.0 * perturbacion)
    return matriz


if __name__ == "__main__":
    # Prueba rapida: que las derivadas se evaluen sin romperse.
    estado_de_prueba = np.array([5.0, 2.0, 1.0])
    print("estado      :", estado_de_prueba)
    print("derivadas   :", derivadas(estado_de_prueba))
    print()
    print("jacobiano:")
    print(jacobiano_numerico(estado_de_prueba))
    print()
    print("autovalores:", np.linalg.eigvals(jacobiano_numerico(estado_de_prueba)))
