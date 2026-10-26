# Teoría del TP y cómo se ve en el código

Documento de estudio. Cada sección explica un concepto y muestra **dónde está implementado**.
Si entendés este archivo, podés defender el trabajo.

---

## 1. Qué es un Problema de Valores Iniciales (PVI)

Un PVI es una ecuación diferencial más una condición inicial:

```
dy/dt = f(t, y)        y(t₀) = y₀
```

La ecuación dice **cómo cambia** el sistema; la condición inicial dice **de dónde parte**.
Juntas determinan una única trayectoria.

En este TP el sistema es de **tres** ecuaciones acopladas, así que `y` es un vector:

```
y = [N, P, S]          dy/dt = [dN/dt, dP/dt, dS/dt]
```

**Acopladas** significa que `dN/dt` depende también de P y S, no solo de N. Por eso no se pueden
resolver de a una: hay que avanzarlas **todas juntas** en cada paso.

**Autónomo:** el tiempo no aparece explícitamente en el lado derecho, solo a través de las
variables. Por eso `derivadas(estado)` no recibe `t`.

> **En el código:** `modelo.py` → función `derivadas(estado)`. Recibe `[N, P, S]`, devuelve
> `[dN/dt, dP/dt, dS/dt]`. Es *el modelo*, separado de *los métodos*.

---

## 2. Por qué hace falta resolver numéricamente

Este sistema **no tiene solución analítica**. Las respuestas funcionales son cocientes no lineales
(`aN/(1+ah₁N)`), y un sistema de tres EDOs acopladas con esas no linealidades no se integra a mano.

No existe una fórmula `N(t) = ...`. Lo único que se puede hacer es **avanzar en el tiempo de a
pasitos**, calculando el estado siguiente a partir del actual. Eso es integración numérica.

> **Pregunta de oral:** "¿Por qué no lo resolvés analíticamente?" → Porque no tiene primitiva;
> el acoplamiento no lineal lo impide. Por eso el problema es *numérico*.

---

## 3. El modelo biológico

### 3.1 Las tres ecuaciones

```
dN/dt = r·N·(1 - N/K) - d·N - F_PN·P - F_SN·S     ← presa
dP/dt = b·F_PN·P - m·P - F_SP·S                   ← depredador
dS/dt = f·F_SN·S + g·F_SP·S - q·S                 ← superdepredador
```

Leídas término a término:

| término | qué significa |
|---|---|
| `r·N·(1 - N/K)` | crecimiento **logístico**: crece exponencial si N≪K, y se frena al acercarse a K |
| `d·N` | muerte natural de la presa |
| `F_PN·P` | lo que el depredador se come |
| `F_SN·S` | lo que el superdepredador se come de la presa |
| `b·F_PN·P` | lo comido se convierte en depredador nuevo, con eficiencia `b` |
| `m·P` | muerte natural del depredador |
| `F_SP·S` | el superdepredador se come al depredador |
| `f·F_SN·S + g·F_SP·S` | S gana comiendo de **los dos** niveles de abajo |
| `q·S` | muerte natural del superdepredador |

**El término logístico.** `(1 - N/K)` es un freno. Si N es chico, vale ≈1 y la presa crece a tasa
`r`. Si N → K, vale ≈0 y el crecimiento se detiene. Si N > K, se vuelve negativo y la población
baja. K es la **capacidad de carga**: lo que el ambiente banca.

**La estructura trófica.** N → P → S, pero además N → S directo. S come de dos niveles: es
**omnivoría intragremial**. Eso es lo que vuelve *híbrido* al modelo y es la fuente habitual del
caos que advierte el enunciado.

> **En el código:** `modelo.py` → cada ecuación es un bloque con los términos nombrados
> (`crecimiento_logistico`, `depredacion_por_super`, …). Si te preguntan por un término, se lee
> del nombre de la variable.

### 3.2 Las respuestas funcionales

Una **respuesta funcional** es cuánta presa consume **un** depredador por unidad de tiempo, en
función de cuánta presa hay.

| | fórmula | tipo | forma |
|---|---|---|---|
| `F_PN` | `a·N / (1 + a·h₁·N)` | Holling **II** | saturante |
| `F_SN` | `c·N² / (1 + c·h₂·N²)` | Holling **III** | sigmoidea |
| `F_SP` | `e·P / (1 + e·h₃·P)` | Holling **II** | saturante |

Que haya tipos distintos en el mismo modelo es lo que se llama **respuestas mixtas**, y es el
nombre del Problema 12.

**Holling II — el depredador que se llena.**
Crece casi lineal con poca presa y se **satura** con mucha. El techo es `1/h`: por más presa que
haya, el depredador necesita `h` unidades de tiempo para manipular cada una.
*Efecto:* **desestabiliza**. Al saturarse, la mortalidad per cápita de la presa *cae* cuando la
presa abunda; eso realimenta el crecimiento y favorece oscilaciones (paradoja del enriquecimiento).

**Holling III — el que aprende.**
El `N²` del numerador hace que la curva **arranque plana**: a densidad baja de presa, el
superdepredador casi no la consume. Recién cuando la presa abunda el consumo despega.
*Interpretación:* aprendizaje o cambio de dieta — si la presa escasea, no rinde buscarla.
*Efecto:* **estabiliza**. Ese arranque plano le da a la presa un **refugio** justo cuando está por
extinguirse. Es el mecanismo que evita el colapso.

> **La pregunta que seguro cae:** "¿Por qué tipo III en el superdepredador y tipo II en los otros?"
> → Por el refugio a densidades bajas. El N² protege a la presa cuando es escasa.

> **En el código:** `modelo.py` → tres funciones, una por respuesta, cada una con su efecto sobre
> la estabilidad documentado en el docstring.

---

## 4. Método de Euler explícito

### 4.1 De dónde sale

Del desarrollo de Taylor de la solución alrededor de `tₙ`:

```
y(tₙ + Δt) = y(tₙ) + Δt·y'(tₙ) + (Δt²/2)·y''(ξ)
             └──────────────────┘   └─────────┘
              lo que Euler calcula   lo que TIRA
```

Como `y'(tₙ) = f(yₙ)`, truncar en el término lineal da:

```
y_{n+1} = yₙ + Δt·f(yₙ)
```

### 4.2 La idea

Aproximar la curva por su **recta tangente** en el punto actual y avanzar sobre ella.

### 4.3 El orden

- **Error local** (un paso): el primer término descartado, `O(Δt²)`
- **Error global** (acumulado en `N = T/Δt` pasos): `O(Δt)`

Se pierde un orden al acumular: el error por paso es Δt², pero hay T/Δt pasos, y
`Δt² · (T/Δt) = T·Δt`.

**Euler es de orden 1.** Partir Δt a la mitad parte el error a la mitad.

### 4.4 Su defecto

Usa la pendiente **solo al inicio** del intervalo. Como la pendiente cambia durante el paso, el
método se queda corto (o se pasa) de forma **sistemática**, y ese sesgo se acumula.

> **En el código:** `euler.py` → una sola llamada a `derivadas()` por paso.

---

## 5. Runge-Kutta 2 (Heun / Euler modificado)

### 5.1 La fórmula

```
k₁ = f(yₙ)                      ← pendiente al INICIO
k₂ = f(yₙ + Δt·k₁)              ← pendiente al FINAL, estimada
y_{n+1} = yₙ + (Δt/2)·(k₁ + k₂) ← avance con el PROMEDIO
```

### 5.2 La idea — predictor / corrector

1. **Predictor:** dar un paso de Euler de prueba para *espiar* dónde caería.
2. **Corrector:** evaluar la pendiente **ahí**, y avanzar con el promedio de las dos.

Es la **regla del trapecio** aplicada al tiempo: en vez de aproximar el área bajo `f` con un
rectángulo (Euler), se la aproxima con un trapecio usando los dos extremos.

**Por qué se llama "Euler modificado":** porque el predictor *es* un paso de Euler. La
modificación es no quedarse con él, sino usarlo para estimar k₂ y promediar.

### 5.3 El orden

- Error local `O(Δt³)`, error global `O(Δt²)` → **orden 2**
- Partir Δt a la mitad divide el error por **4**

### 5.4 El costo

Dos evaluaciones de `f` por paso contra una de Euler. Pero para igual precisión admite un Δt mucho
más grande, así que **hace menos cuentas en total**:

| objetivo | Euler | RK2 |
|---|---|---|
| bajar el error 100× | Δt/100 → 100× pasos | Δt/10 → 10× pasos, 20× evaluaciones |

> **En el código:** `rk2.py` → dos llamadas a `derivadas()` por paso, con los nombres
> `pendiente_inicial` / `pendiente_final` / `pendiente_promedio`.

---

## 6. El punto de equilibrio

### 6.1 Qué es

El estado donde el sistema **deja de cambiar**:

```
dN/dt = dP/dt = dS/dt = 0
```

Si el sistema llega ahí, se queda. Las tres poblaciones se mantienen constantes.

### 6.2 Para qué sirve saberlo

1. **Valida el código.** Si la solución numérica converge al equilibrio calculado a mano, el código
   anda. Es la mejor verificación que tenés.
2. **Guía la elección de parámetros.** Hay que elegirlos para que el equilibrio tenga las **tres
   especies positivas**. Si alguna da negativa, esa especie se extingue.
3. **Fija el Δt.** Los autovalores del jacobiano *en el equilibrio* determinan el paso máximo.

### 6.3 Cómo encontrarlo

**Analíticamente:** igualar las tres a cero y despejar. De `dS/dt = 0` se saca `S` de factor común:

```
S·(f·F_SN + g·F_SP - q) = 0
```

O bien `S = 0` (el superdepredador se extingue) o bien `f·F_SN + g·F_SP = q`. Esa segunda
condición es la que hay que cumplir para que S sobreviva, y acota `f`, `g` y `q`.

**Numéricamente:**

```python
from scipy.optimize import fsolve
equilibrio = fsolve(derivadas, [5.0, 2.0, 1.0])
```

---

## 7. Estabilidad — por qué el Δt no es libre

### 7.1 Linealización

Cerca del equilibrio `y*`, escribo `y = y* + δ` con δ chico. Desarrollando `f` por Taylor:

```
dδ/dt ≈ J·δ          donde J = ∂f/∂y  (matriz jacobiana)
```

El sistema no lineal se comporta **localmente** como uno lineal. Los **autovalores** de J, los λ,
gobiernan todo.

### 7.2 Qué dicen los autovalores

| parte real de λ | qué pasa |
|---|---|
| **negativa** | la perturbación se apaga → el equilibrio **atrae** |
| **positiva** | la perturbación crece → el equilibrio **repele** |
| parte imaginaria ≠ 0 | hay oscilación → se acerca (o aleja) **en espiral** |

**Todos con parte real negativa ⟹ equilibrio asintóticamente estable.** Ese es el caso que querés
reportar: las tres curvas tendiendo a sus valores finales.

### 7.3 La condición sobre Δt

Euler aplicado al sistema linealizado:

```
δ_{n+1} = (I + Δt·J)·δₙ
```

Cada modo se multiplica en cada paso por el **factor de amplificación**:

```
g(λ) = 1 + λ·Δt
```

Para que el error no crezca hace falta `|g| ≤ 1`. Para λ real negativo:

```
|1 + λΔt| ≤ 1   ⟹   Δt ≤ 2/|λ|
```

**Si se supera esa cota, |g| > 1 y el error se MULTIPLICA en cada paso.** La solución oscila con
amplitud creciente y termina desbordando. Eso es **inestabilidad numérica** — no es un bug del
código, es el método rompiéndose.

Para RK2 el factor es `g = 1 + z + z²/2` con `z = λΔt`. Su región de estabilidad es **más amplia**
en el plano complejo, aunque sobre el eje real da la misma cota.

> **Nota fina para el oral:** si los autovalores son **complejos puros** (oscilación sin
> amortiguación), Euler explícito es inestable **para todo Δt**, porque su región de estabilidad es
> un círculo tangente al eje imaginario. RK2 tolera un tramo del eje imaginario. Es un argumento
> fuerte a favor de RK2 en sistemas oscilatorios como este.

### 7.4 ⚠️ Estabilidad NO es precisión

Un Δt que respeta `Δt ≤ 2/|λ|` garantiza que **no explota**. No garantiza que esté **bien**.
La trayectoria puede estar completamente deformada y seguir siendo "estable".

Por eso el Δt **no se elige con la cota**: se elige con el estudio de convergencia.

---

## 8. Estudio de convergencia — la justificación fuerte del Δt

### 8.1 La idea

Integrar con Δt, Δt/2, Δt/4… y comparar contra la corrida más fina. Si el método es de orden `p`,
al partir Δt a la mitad el error se divide por `2^p`:

| método | orden p | cociente esperado |
|---|---|---|
| Euler | 1 | **≈ 2** |
| RK2 | 2 | **≈ 4** |

### 8.2 Para qué sirve

1. **Valida la implementación.** Si Euler da 2 y RK2 da 4, los métodos están bien programados.
   Si RK2 te da 2, hay un error en el corrector.
2. **Justifica el Δt.** Cuando la solución deja de cambiar al refinar, convergiste. Ese Δt sirve.

### 8.3 ⚠️ El error que arruina el estudio

**Medir en el equilibrio.** Si integrás hasta que el sistema ya se asentó, *todos* los Δt dan el
mismo resultado y el error da ~10⁻¹⁴ (ruido de máquina). Los cocientes salen cualquier cosa.

**Hay que medir en el TRANSITORIO**, mientras la solución todavía se mueve.

---

## 9. El caos

El enunciado advierte: *"Los modelos con respuestas mixtas presentan caos con mucha frecuencia."*

**Qué es.** Una trayectoria acotada que nunca se repite y que es **extremadamente sensible a las
condiciones iniciales**: dos arranques casi idénticos divergen exponencialmente.

**Por qué aparece acá.** La omnivoría intragremial (S comiendo de dos niveles) más la saturación
de Holling II en el depredador ápice generan atractores extraños para ciertos parámetros.

**Cómo detectarlo:**
- La serie temporal nunca se asienta ni cae en un ciclo limpio
- El retrato de fase 3D llena una región en vez de converger a un punto o a un lazo cerrado
- Cambiar la condición inicial en 10⁻⁶ da una trayectoria totalmente distinta

**Por qué es un problema numérico.** En régimen caótico **no se puede validar el código con la
trayectoria**: cualquier error de redondeo se amplifica y la solución numérica se separa de la
exacta. Por eso conviene elegir parámetros fuera de la zona de caos.

> **Pregunta de oral:** "¿Cómo sabés que no estás en caos?" → Porque la trayectoria converge a un
> punto fijo, y perturbando la condición inicial se llega al mismo punto.

---

## 10. Mapa: concepto → archivo

| concepto | archivo | qué mirar |
|---|---|---|
| El modelo, las 3 EDOs | `modelo.py` | `derivadas()` |
| Respuestas funcionales | `modelo.py` | las 3 funciones `respuesta_*` |
| Parámetros | `modelo.py` | bloque de constantes arriba |
| Euler | `euler.py` | una evaluación por paso |
| RK2 / Heun | `rk2.py` | predictor + corrector |
| Jacobiano | `modelo.py` | `jacobiano_numerico()` |
| Equilibrio, autovalores, Δt | `analisis.py` | **pendiente — lo hacés vos** |
| Figuras | `figuras.py` | helpers de graficado |

---

## 11. Las preguntas que van a caer

| pregunta | respuesta corta |
|---|---|
| ¿Por qué numérico y no analítico? | No tiene solución cerrada: 3 EDOs acopladas no lineales |
| ¿Por qué ese Δt? | Cota 2/\|λ\| + tabla de convergencia (cociente 2 y 4) |
| ¿Qué pasa si agrando Δt? | \|1+λΔt\| > 1 → el error se multiplica cada paso → diverge |
| ¿Por qué RK2 mejor que Euler? | Orden 2 vs 1; promedia dos pendientes en vez de usar una |
| ¿Por qué Holling III en S? | El N² da refugio a la presa a densidades bajas → estabiliza |
| ¿Cómo validás el código? | Converge al equilibrio analítico + el orden medido coincide |
| ¿Por qué esos parámetros? | Intervalos del enunciado + equilibrio positivo + fuera del caos |
| ¿Qué es el equilibrio? | dy/dt = 0; atrae si todos los Re(λ) < 0 |
| ¿Es asintótico? | Sí, si Re(λ) < 0 para los tres |
| Explicá esta línea | ← practicá con cualquier línea al azar |
