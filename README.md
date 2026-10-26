# Simulación Numérica y Análisis de Estabilidad de un Ecosistema Trófico con Respuestas Mixtas
### Trabajo Práctico de Cálculo Numérico — Problema 12: Respuestas Mixtas / Híbridas
**Facultad de Ingeniería — Universidad Austral (2026)**  
**Autores:** Bruno Cornejo y María Paz Fogliato

---

## 📋 Descripción del Proyecto
Este repositorio contiene la resolución integral, código fuente, figuras y el paper científico en PDF correspondiente al **Problema 12** de Cálculo Numérico. El trabajo modela la dinámica no lineal de una red trófica tripartita con omnivoría intragremial integrada por:
* **Presa basal ($):** Sujeta a crecimiento logístico y capacidad de carga $.
* **Depredador intermedio ($):** Depreda sobre la presa con respuesta funcional de Holling tipo II.
* **Superdepredador ($):** Depreda sobre el depredador intermedio (Holling tipo II) y sobre la presa mediante una **respuesta sigmoidea de Holling tipo III**, la cual provee un *efecto refugio* a bajas densidades que estabiliza el ecosistema.

---

## ⚙️ Métodos Numéricos Implementados
1. **Euler Explícito (Orden 1):** Proyección tangencial en el extremo inicial ((h)$ global).
2. **Euler Mejorado ó Método del Punto Medio (Runge-Kutta 2, Orden 2):**
   * Deducción a partir de serie de Taylor ( = 0 \implies w_2 = 1, lpha = 1/2, eta = 1/2$):
     F_1 = h f(x_i, y_i)
     F_2 = h f\left(x_i + \frac{1}{2}h, \, y_i + \frac{1}{2}F_1\right)
     y_{i+1} = y_i + F_2
   * Error global cuadrático (h^2)$, validando que aumentar el orden es sustancialmente más eficiente que reducir el paso $.
3. **Cota Analítica de Estabilidad:** Linealización y cálculo espectral del Jacobiano ( \le 2/|\lambda_{\max}| \approx 8{,}6676\,\text{d}$).
4. **Estudio de Convergencia:** Verificación de órdenes asintóticos empíricos en régimen transitorio frente a solución de referencia ({\text{ref}} = 10^{-4}\,\text{d}$).

---

## 🗂️ Estructura del Repositorio
`	ext
├── modelo.py                   # Sistema de EDOs, derivadas y cálculo del Jacobiano
├── euler.py                    # Integrador numérico de Euler explícito
├── rk2.py                      # Integrador de Runge-Kutta 2 (Euler Mejorado / Punto Medio)
├── analisis.py                 # Script central: equilibrio, estabilidad, convergencia
├── figuras.py                  # Generación automatizada de las 5 figuras científicas
├── referencia_2especies.py     # Script complementario de validación
├── figuras_salida/             # Gráficos generados en alta resolución (PNG)
│   ├── figura_1_series_temporales.png
│   ├── figura_2_retrato_fase_3d.png
│   ├── figura_3_inestabilidad.png
│   ├── figura_4_comparacion_metodos.png
│   └── figura_5_convergencia_loglog.png
├── informe/                    # Paper y entregables finales
│   ├── informe_final.pdf       # Paper publicado en formato Anales AFA (3 páginas)
│   ├── informe.tex             # Código fuente LaTeX (doble columna, Times New Roman)
│   └── figuras/                # Copia de figuras vinculadas al informe
├── TEORIA.md                   # Resumen teórico y fórmulas de la cátedra
└── README.md                   # Descripción del repositorio
`

---

## 🚀 Instrucciones de Ejecución

### Requisitos
* Python 3.10 o superior
* Paquetes científicos: 
umpy, scipy, matplotlib, 
eportlab, pymupdf

`ash
pip install numpy scipy matplotlib reportlab pymupdf
`

### Ejecutar Simulación y Análisis Completo
`ash
python analisis.py
`

El script imprimirá en consola el punto de coexistencia (^*, P^*, S^*$), los autovalores del Jacobiano, la cota crítica de estabilidad de Euler, la tabla de convergencia transitoria y generará automáticamente todas las figuras en la carpeta iguras_salida/.
