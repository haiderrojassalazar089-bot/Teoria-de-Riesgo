# RiskLab · USTA

## Tablero Interactivo de Análisis de Riesgo Financiero

**Universidad Santo Tomás · Bogotá · Facultad de Estadística**  
**Curso:** Teoría del Riesgo Financiero  
**Profesor:** Javier Mauricio Sierra  
**Proyecto Integrador:** Dashboard de Riesgo con datos en tiempo real  

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Stack Tecnológico](#stack-tecnológico)
3. [Activos Analizados](#activos-analizados)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Instalación y Uso](#instalación-y-uso)
6. [Módulos Implementados](#módulos-implementados)
   - [Vista General](#-vista-general)
   - [Módulo 1 · Análisis Técnico](#módulo-1--análisis-técnico-12)
   - [Módulo 2 · Rendimientos](#módulo-2--rendimientos-8)
   - [Módulo 3 · ARCH/GARCH](#módulo-3--archgarch-12)
   - [Módulo 4 · CAPM & Beta](#módulo-4--capm--beta-8)
   - [Módulo 5 · VaR & CVaR](#módulo-5--var--cvar-12)
   - [Módulo 6 · Markowitz](#módulo-6--markowitz-12)
   - [Módulo 7 · Señales & Alertas](#módulo-7--señales--alertas-)
   - [Módulo 8 · Macro & Benchmark](#módulo-8--macro--benchmark-)
7. [APIs y Fuentes de Datos](#apis-y-fuentes-de-datos)
8. [Diseño Visual](#diseño-visual)
9. [Fundamentos Teóricos](#fundamentos-teóricos)
10. [Decisiones de Diseño y Limitaciones](#decisiones-de-diseño-y-limitaciones)
11. [Referencias Bibliográficas](#referencias-bibliográficas)

---

## Descripción General

**RiskLab** es un tablero web interactivo construido en Python y Streamlit que implementa los modelos de análisis de riesgo financiero del curso de Teoría del Riesgo de la Universidad Santo Tomás. Cubre desde el análisis técnico clásico hasta la optimización de portafolios de Markowitz, modelos GARCH de volatilidad condicional, métricas de VaR y CVaR, y evaluación del desempeño frente al mercado.

**Filosofía de diseño:**
- Sin datasets estáticos: todos los datos se descargan en tiempo real desde **Yahoo Finance** mediante `yfinance`.
- Código modular: cada módulo del curso corresponde a un archivo Python independiente en `pages/`.
- Interpretación académica integrada: cada sección incluye paneles explicativos alineados con los contenidos del curso.
- Caching inteligente: los datos se cachean 30 minutos para evitar solicitudes redundantes a la API.

---

## Stack Tecnológico

| Componente | Tecnología | Versión |
|------------|-----------|---------|
| Framework web | Streamlit | 1.43.2 |
| Visualización interactiva | Plotly | 5.22.0 |
| Datos de mercado | yfinance | ≥ 0.2.55 (1.2.0 recomendada) |
| Modelos ARCH/GARCH | arch | 7.0.0 |
| Optimización de portafolio | PyPortfolioOpt | 1.5.6 |
| Estadística inferencial | scipy | 1.13.1 |
| Modelos estadísticos | statsmodels | 0.14.2 |
| Procesamiento de datos | pandas | 2.2.2 |
| Álgebra lineal / simulación | numpy | 1.26.4 |
| HTTP / APIs externas | requests | 2.32.3 |
| Lenguaje | Python | 3.12.1 |

---

## Activos Analizados

El portafolio de referencia está compuesto por cinco acciones del S&P 500, diversificadas en sectores distintos, con el índice como benchmark:

| Ticker | Empresa | Sector | Justificación |
|--------|---------|--------|---------------|
| AAPL | Apple Inc. | Tecnología | Empresa de mayor capitalización; alta liquidez y cobertura analítica |
| JPM | JPMorgan Chase & Co. | Financiero | Banco más grande de EE.UU.; sensible a política monetaria |
| XOM | ExxonMobil Corporation | Energía | Proxy del precio del petróleo; correlación con commodities |
| JNJ | Johnson & Johnson | Salud | Activo defensivo; baja correlación con ciclo económico |
| AMZN | Amazon.com Inc. | Consumo discrecional | Alta exposición al consumidor; ciclo de crecimiento volátil |
| ^GSPC | S&P 500 Index | Benchmark | Referencia del mercado amplio estadounidense |

**Horizonte de análisis:** 3 años de datos diarios  
**Tasa libre de riesgo:** `^IRX` — T-Bill 3 meses, obtenida en tiempo real desde Yahoo Finance  
**Frecuencia de actualización:** cada 30 minutos (caché de Streamlit)  
**Tipo de rendimientos:** logarítmicos (`ln(Pₜ / Pₜ₋₁)`) — salvo indicación contraria

---

## Estructura del Proyecto

```
Teoria-de-Riesgo/
│
├── app.py                      # Punto de entrada principal — configuración y navegación
├── requirements.txt            # Dependencias con versiones fijadas
├── README.md                   # Este archivo
│
├── data/
│   ├── __init__.py             # Exporta funciones públicas del módulo
│   └── loader.py               # Descarga, caché y preprocesamiento de datos (yfinance 1.2.0)
│                               # → get_prices(), get_ohlcv(), get_returns(), get_risk_free_rate()
│
├── utils/
│   ├── __init__.py
│   ├── theme.py                # Paleta COLORS, TICKER_COLORS, función plotly_base()
│   └── styles.py               # CSS global inyectado en Streamlit (GLOBAL_CSS)
│
└── pages/
    ├── __init__.py
    ├── overview.py             # ◈ Vista General — KPIs, rendimiento normalizado, correlación
    ├── m1_technical.py         # Módulo 1 — Análisis Técnico (SMA, EMA, BB, RSI, MACD, Estocástico)
    ├── m2_returns.py           # Módulo 2 — Rendimientos (distribución, hechos estilizados, normalidad)
    ├── m3_garch.py             # Módulo 3 — ARCH/GARCH (ARCH·GARCH·GJR·EGARCH, pronóstico)
    ├── m4_capm.py              # Módulo 4 — CAPM & Beta (MCO, SML, descomposición de riesgo)
    ├── m5_var.py               # Módulo 5 — VaR & CVaR (Paramétrico · Histórico · MC · Kupiec)
    ├── m6_markowitz.py         # Módulo 6 — Markowitz (frontera eficiente, portafolios óptimos)
    ├── m7_signals.py           # Módulo 7 — Señales & Alertas (semáforo, umbrales configurables)
    └── m8_macro.py             # Módulo 8 — Macro & Benchmark (Alpha Jensen, TE, IR, VIX, FX)
```

---

## Instalación y Uso

### Requisitos previos

- Python 3.12.x ([descargar](https://www.python.org/downloads/))
- pip actualizado: `python -m pip install --upgrade pip`

### Instalación paso a paso

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd Teoria-de-Riesgo

# 2. (Recomendado) Crear entorno virtual
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el tablero
streamlit run app.py
```

El tablero estará disponible automáticamente en `http://localhost:8501`

### Solución de problemas comunes

| Problema | Causa probable | Solución |
|----------|---------------|----------|
| `ModuleNotFoundError: arch` | arch no instalado | `pip install arch==7.0.0` |
| `yfinance download returns empty` | Cambio en API de Yahoo | `pip install --upgrade yfinance` |
| Gráficos en blanco | Plotly y Streamlit incompatibles | Usar versiones fijadas en `requirements.txt` |
| Error de tipo de cambio USD/COP | Sin conexión a internet | El módulo 8 usa fallback a "N/D" automáticamente |

---

## Módulos Implementados

### ◈ Vista General

Panel resumen del portafolio equi-ponderado (20% por activo).

**KPIs calculados:**
- **Retorno acumulado:** `(1+rₜ).cumprod() - 1` sobre rendimientos diarios
- **Volatilidad anual:** `σ_diaria × √252`
- **Ratio de Sharpe:** `(E[R] - Rf) / σ_anual` con Rf obtenida de `^IRX`
- **Máximo Drawdown:** `min[(Pₜ - Pmáx) / Pmáx]` — caída máxima desde máximo

**Visualizaciones:**
- Rendimientos normalizados en base 100 (5 activos comparables)
- Heatmap de correlación de log-rendimientos (paleta gold-navy)
- Tabla resumen: precio actual, variación diaria, YTD, volatilidad anual

---

### Módulo 1 · Análisis Técnico `12%`

Implementación completa de indicadores técnicos clásicos sobre datos OHLCV en tiempo real.

**Indicadores implementados:**

| Indicador | Parámetros ajustables | Señal generada |
|-----------|----------------------|----------------|
| SMA (Media Móvil Simple) | Períodos 1 y 2 (5–200) | Golden/Death Cross |
| EMA (Media Móvil Exponencial) | Período (5–100) | Confirmación de tendencia |
| Bandas de Bollinger | Período 20, k=2 | Squeeze, ruptura de bandas |
| RSI | Período (7–30) | Sobrecompra > 70, sobreventa < 30 |
| MACD | (12, 26, 9) | Cruce de líneas, histograma |
| Oscilador Estocástico | %K=14, %D=3 | Cruce en zonas extremas |

**Tipos de gráfico:** Línea y Velas japonesas (OHLC completo)  
**Períodos disponibles:** 1, 2 y 3 años

**Fórmulas clave:**
```
SMA(n)  = (1/n) · Σ Pₜ₋ᵢ
EMA(n)  = α·Pₜ + (1-α)·EMA(n-1),  α = 2/(n+1)
BB±     = SMA(20) ± 2·σ₂₀
RSI     = 100 - 100/(1 + RS),  RS = Ḡ/L̄  (medias de alzas y bajas)
MACD    = EMA(12) - EMA(26);  Señal = EMA(9) del MACD
%K      = 100·(C - L_n) / (H_n - L_n)
```

---

### Módulo 2 · Rendimientos `8%`

Caracterización estadística completa de los rendimientos y verificación de los hechos estilizados de los mercados financieros.

**Estadísticas descriptivas calculadas:**
- Media diaria y anualizada (`μ × 252`)
- Desviación estándar diaria y anualizada (`σ × √252`)
- Asimetría (sesgo): `E[(X-μ)³/σ³]`
- Curtosis de exceso: `E[(X-μ)⁴/σ⁴] - 3`
- Mínimo, máximo, número de observaciones

**Visualizaciones:**
- Histograma de log-rendimientos con curva normal superpuesta
- Q-Q Plot vs distribución normal (desviaciones en colas)
- Gráfico de r² (volatility clustering)
- Boxplot comparativo de los 5 activos

**Pruebas de normalidad:**

| Prueba | H₀ | Estadístico | Limitación |
|--------|-----|-------------|------------|
| Jarque-Bera | Distribución normal | χ² basado en asimetría y curtosis | Potencia baja en muestras pequeñas |
| Shapiro-Wilk | Distribución normal | Correlación con cuantiles normales | Muestra ≤ 5,000 obs. |

**Hechos estilizados documentados:**
1. **Colas pesadas (leptocurtosis):** curtosis > 3; Q-Q plot diverge en extremos
2. **Agrupamiento de volatilidad:** autocorrelación positiva en r²
3. **Asimetría negativa:** caídas más frecuentes y pronunciadas que subidas
4. **Efecto apalancamiento:** correlación negativa entre rₜ y σₜ₊₁

---

### Módulo 3 · ARCH/GARCH `12%`

Modelado de la heterocedasticidad condicional para capturar el agrupamiento de volatilidad.

**Justificación formal — Test ARCH-LM:**
- H₀: no existe efecto ARCH (varianza constante — homocedasticidad)
- Estadístico: Multiplicadores de Lagrange con 5 rezagos
- Rechazo de H₀ → justifica el uso de modelos GARCH

**Especificaciones ajustadas:**

| Modelo | Ecuación de varianza | Característica |
|--------|---------------------|----------------|
| ARCH(1) | σ²ₜ = ω + α·ε²ₜ₋₁ | Línea base; sin memoria larga |
| GARCH(1,1) | σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁ | Estándar de la industria |
| GJR-GARCH(1,1,1) | + γ·ε²ₜ₋₁·𝟙[εₜ₋₁<0] | Captura efecto apalancamiento |
| EGARCH(1,1) | ln(σ²ₜ) = ω + α·\|zₜ₋₁\| + γ·zₜ₋₁ + β·ln(σ²ₜ₋₁) | Varianza siempre positiva |

**Selección de modelo:** AIC/BIC (menor es mejor) y Log-Likelihood (mayor es mejor)

**Parámetro de persistencia:** `α + β` — si → 1 indica memoria larga de la volatilidad (proceso integrado IGARCH)

**Distribuciones disponibles:** Normal, t-Student, t asimétrica (skew-t)

**Diagnóstico de residuos:**
- Residuos estandarizados `εₜ/σₜ` deben ser ruido blanco con varianza unitaria
- Q-Q plot de residuos
- Test Jarque-Bera sobre residuos (evalúa si la distribución elegida es adecuada)

**Pronóstico:** volatilidad h-pasos con reversión a la media de largo plazo

---

### Módulo 4 · CAPM & Beta `8%`

Estimación del riesgo sistemático y cálculo del rendimiento esperado según el Capital Asset Pricing Model.

**Modelo:** `E[Rᵢ] = Rf + βᵢ · (E[Rm] - Rf)`

**Estimación del Beta por MCO:**
```
βᵢ = Cov(Rᵢ, Rm) / Var(Rm)
αᵢ = Ē[Rᵢ] - βᵢ · Ē[Rm]   (Alpha de Jensen)
```

- Intervalo de confianza del 95% para β
- R² como medida de ajuste del modelo
- p-valor para la significancia estadística de β

**Clasificación de activos:**
- **Agresivo:** β > 1.2 — amplifica movimientos del mercado
- **Defensivo:** β < 0.8 — amortigua movimientos del mercado  
- **Neutro:** 0.8 ≤ β ≤ 1.2 — se mueve en línea con el mercado

**Descomposición del riesgo:**
- Riesgo sistemático: `β² × Var(Rm)` — no diversificable, compensado por el mercado
- Riesgo idiosincrático: `Var(Rᵢ) - β² × Var(Rm)` — diversificable, no compensado

**Visualizaciones:**
- Scatter activo vs S&P 500 con línea de regresión e IC 95%
- Security Market Line (SML) con todos los activos posicionados
- Barra apilada de descomposición del riesgo total

**Limitaciones del CAPM discutidas:**
- Asume mercados eficientes y distribuciones normales
- Factor único de riesgo (ignora tamaño, valor, momentum)
- El alfa de Jensen distinto de cero viola las predicciones del modelo
- Modelos alternativos: Fama-French 3 factores, Carhart 4 factores

---

### Módulo 5 · VaR & CVaR `12%`

Cuantificación de la pérdida potencial máxima bajo distintos métodos y niveles de confianza.

**Definición formal:**  
`VaR_α = -inf{x : P(R ≤ x) > 1-α}` — cuantil (1-α) de la distribución de pérdidas

**Métodos implementados:**

#### VaR Paramétrico (varianza-covarianza)
Asume normalidad de los rendimientos:
```
VaR_α = -(μ + z_α · σ)
CVaR_α = -(μ - σ · φ(z_α) / (1-α))
```
donde `z_α = Φ⁻¹(1-α)` es el cuantil normal y `φ` es la PDF normal estándar.

- **Ventaja:** analítico, rápido, generalizable a múltiples activos
- **Limitación:** subestima el riesgo real dado que los rendimientos tienen colas pesadas

#### VaR por Simulación Histórica
```
VaR_α = -percentil(R, (1-α)·100)
CVaR_α = -E[R | R ≤ -VaR_α]
```
Sin supuestos distribucionales; captura los patrones reales de la muestra.

- **Ventaja:** no paramétrico; incluye eventos históricos extremos
- **Limitación:** limitado al período de muestra; no extrapola escenarios no vistos

#### VaR por Simulación Montecarlo (≥ 10,000 escenarios)
Se generan `N` escenarios bajo distribución normal con `μ` y `σ` históricos:
```
R_sim ~ N(μ, σ²)
VaR_α = -percentil(R_sim, (1-α)·100)
```
- **Ventaja:** flexible; extensible a distribuciones no normales y procesos complejos
- **Limitación:** tan bueno como sus supuestos distribucionales; costoso computacionalmente

#### Expected Shortfall (CVaR / ES)
Medida coherente de riesgo (Artzner et al., 1999):
```
ES_α = -E[R | R ≤ -VaR_α]
```
Promedia las pérdidas de la cola, superando la limitación del VaR de no capturar la magnitud de las pérdidas extremas. Adoptado como estándar regulatorio en Basilea III (FRTB, α=97.5%).

**Tabla comparativa:** VaR diario/anual al 95% y 99% por los tres métodos, con equivalente en USD.

**Backtesting — Test de Kupiec (POF):**
Verifica si la tasa de violaciones observada coincide con el nivel de confianza declarado:
```
H₀: p_obs = 1 - α
LR = -2·[V·ln(p_esp/p_obs) + (T-V)·ln((1-p_esp)/(1-p_obs))] ~ χ²(1)
```
donde `V` = violaciones observadas, `T` = total de días.

---

### Módulo 6 · Markowitz `12%`

Construcción de la frontera eficiente y optimización de portafolios basada en la teoría de media-varianza (Markowitz, 1952).

**Fundamento teórico:**  
Un portafolio `w` es eficiente si no existe otro con mayor retorno esperado para igual volatilidad, ni menor volatilidad para igual retorno esperado.

**Métricas del portafolio:**
```
E[Rₚ] = wᵀ μ
σ²ₚ   = wᵀ Σ w
Sharpe = (E[Rₚ] - Rf) / σₚ
```

**Simulación de portafolios aleatorios (10,000+):**
Se generan vectores de pesos aleatorios mediante la distribución de Dirichlet, garantizando `Σwᵢ = 1, wᵢ ≥ 0`. El mapa de colores refleja el ratio de Sharpe.

**Optimización numérica (scipy.optimize.minimize, método SLSQP):**

| Portafolio | Objetivo | Restricciones |
|------------|---------|---------------|
| Mínima Varianza (MV) | min σₚ | Σwᵢ = 1, wᵢ ≥ 0 |
| Máximo Sharpe (MS) | max (E[Rₚ]-Rf)/σₚ | Σwᵢ = 1, wᵢ ≥ 0 |
| Rendimiento objetivo | min σₚ | Σwᵢ = 1, wᵢ ≥ 0, E[Rₚ] = objetivo |

**Frontera eficiente:** calculada variando el retorno objetivo entre el rendimiento mínimo y máximo factibles, resolviendo el problema de mínima varianza para cada punto.

**Visualizaciones:**
- Heatmap interactivo de correlaciones con valores numéricos
- Conjunto factible coloreado por ratio de Sharpe
- Frontera eficiente resaltada sobre el conjunto factible
- Activos individuales posicionados en el espacio riesgo-retorno
- Gráficos de torta con composición de portafolios MV y MS
- Tabla de pesos con selector de rendimiento objetivo

---

### Módulo 7 · Señales & Alertas ★

Sistema automatizado de señales de compra/venta basado en indicadores técnicos clásicos, con umbrales completamente configurables por el usuario.

**Indicadores evaluados automáticamente:**

#### 1. Cruce del MACD
- **COMPRA:** línea MACD cruza la señal al alza (entre t-1 y t)
- **VENTA:** línea MACD cruza la señal a la baja
- **NEUTRAL+/−:** posición relativa sin cruce reciente

#### 2. RSI en zonas extremas
- **COMPRA:** RSI < umbral de sobreventa (configurable, default 30)
- **VENTA:** RSI > umbral de sobrecompra (configurable, default 70)
- **NEUTRAL:** RSI entre ambos umbrales

#### 3. Bandas de Bollinger
- **COMPRA:** precio de cierre toca o rompe la banda inferior
- **VENTA:** precio de cierre toca o rompe la banda superior
- **NEUTRAL:** precio dentro de las bandas

#### 4. Golden Cross / Death Cross
- **COMPRA:** SMA corta cruza SMA larga al alza (Golden Cross)
- **VENTA:** SMA corta cruza SMA larga a la baja (Death Cross)
- **NEUTRAL+/−:** posición relativa sin cruce reciente

#### 5. Oscilador Estocástico
- **COMPRA:** %K cruza %D al alza en zona de sobreventa
- **VENTA:** %K cruza %D a la baja en zona de sobrecompra
- **PRECAUCIÓN / ATENCIÓN:** zona extrema sin cruce confirmado

**Resumen consolidado por activo:**
Score numérico compuesto de las 5 señales → clasificación: COMPRA FUERTE / SESGO ALCISTA / NEUTRAL / SESGO BAJISTA / VENTA FUERTE

**Umbrales configurables:**
- RSI: período (7–30), sobrecompra (60–85), sobreventa (15–40)
- Bandas de Bollinger: período (10–50), k (1.0–3.0σ)
- Cruces de medias: SMA corta (5–50), SMA larga (20–200)
- Estocástico: zonas alta (70–90) y baja (10–30)

**Panel visual:** tarjetas tipo semáforo por activo con badges de color (verde=compra, rojo=venta, gris=neutral) y gráfico multi-panel detallado por activo seleccionado.

> **Nota académica:** las señales técnicas no garantizan predicciones de mercado. Deben combinarse con análisis fundamental y gestión de riesgo.

---

### Módulo 8 · Macro & Benchmark ★

Evaluación del desempeño del portafolio frente al mercado en contexto macroeconómico.

**Panel macroeconómico (datos en tiempo real vía Yahoo Finance):**

| Indicador | Ticker / Fuente | Descripción |
|-----------|----------------|-------------|
| Tasa libre de riesgo | `^IRX` | T-Bill 3M anualizado |
| Rendimiento Tesoro 10A | `^TNX` | Proxy de expectativas de inflación |
| Spread 10Y−3M | Calculado | Pendiente de la curva de rendimientos |
| VIX | `^VIX` | Índice de volatilidad implícita del mercado |
| USD / COP | `USDCOP=X` | Tipo de cambio peso colombiano |
| EUR / USD | `EURUSD=X` | Tipo de cambio euro / dólar |

**Portafolios comparables:**
- Equi-ponderado: 20% por activo
- Máximo Sharpe: pesos optimizados desde el módulo de Markowitz

**Métricas de desempeño activo:**

#### Alpha de Jensen
```
E[Rᵢ] = αⱼ + Rf + βⱼ·(E[Rm] - Rf) + εᵢ
αⱼ = retorno no explicado por el CAPM (anualizado)
```
Positivo y estadísticamente significativo (p<5%) → el gestor agrega valor real.

#### Tracking Error
```
TE = σ(Rₚ - Rm) × √252
```
Volatilidad del retorno activo (diferencia diaria portafolio − benchmark).

#### Information Ratio
```
IR = (E[Rₚ] - E[Rm]) × 252 / TE
```
Retorno activo por unidad de riesgo activo. Benchmark de competencia: IR > 0.5 (Grinold & Kahn, 2000).

**Visualizaciones:**
- Rendimiento acumulado base 100 con área de diferencia (portafolio vs S&P 500)
- Retorno activo diario (barras verde/rojo)
- Alpha de Jensen rodante (ventana configurable 20–120 días)

**Veredicto automático:** el módulo emite un juicio cuantitativo sobre si el portafolio supera al benchmark, con análisis de significancia estadística del alpha.

---

## APIs y Fuentes de Datos

| Fuente | Ticker / Endpoint | Uso | Acceso | Caché |
|--------|------------------|-----|--------|-------|
| Yahoo Finance | AAPL, JPM, XOM, JNJ, AMZN | Precios OHLCV históricos | Gratuito, sin API key | 30 min |
| Yahoo Finance | ^GSPC | Benchmark S&P 500 | Gratuito, sin API key | 30 min |
| Yahoo Finance | ^IRX | Tasa libre de riesgo (T-Bill 3M) | Gratuito, sin API key | 30 min |
| Yahoo Finance | ^TNX | Rendimiento Tesoro 10 años | Gratuito, sin API key | 30 min |
| Yahoo Finance | ^VIX | Índice de volatilidad implícita | Gratuito, sin API key | 30 min |
| Yahoo Finance | USDCOP=X, EURUSD=X | Tipos de cambio | Gratuito, sin API key | 30 min |

> **Fallback:** si la descarga de la tasa libre de riesgo falla (sin conexión), el sistema usa automáticamente 5.25% como valor de referencia manual.

---

## Diseño Visual

**Estética:** Minimalismo Académico Refinado — inspirado en terminales Bloomberg y publicaciones de Journal of Finance.

**Paleta de colores:**

| Rol | Hex | Uso |
|-----|-----|-----|
| Fondo principal | `#EEF1F6` | Background de la app |
| Superficie | `#FFFFFF` | Cards, gráficos, sidebar |
| Acento principal | `#8B6914` | Dorado antiguo — líneas, badges, énfasis |
| Positivo | `#1A6B4A` | Verde musgo — retornos positivos, señales de compra |
| Negativo | `#8B2A2A` | Rojo vino — retornos negativos, señales de venta |
| Cielo | `#1A4F6E` | Azul profundo — JPM, elementos secundarios |
| Violeta | `#3D2F6B` | Violeta oscuro — JNJ, elementos terciarios |
| Texto principal | `#1A2035` | Títulos y datos |
| Texto secundario | `#4A5568` | Párrafos |
| Texto muted | `#8896A8` | Labels, metadata |

**Tipografía:**
- **Playfair Display** (serif): títulos de módulos y valores KPI — legibilidad y autoridad académica
- **IBM Plex Mono** (monoespaciada): labels, tickers, estadísticas — precisión y contexto técnico
- **Inter** (sans-serif): párrafos de interpretación — claridad en lectura

**Componentes UI:**
- Sidebar con logo y navegación por radio buttons estilizados
- Cards con borde superior de color por activo
- Badges de señal tipo semáforo (verde/rojo/gris/azul)
- Métricas `st.metric` con Playfair Display
- Tabs en IBM Plex Mono en minúsculas con acento dorado activo
- Scrollbar personalizada en gold al hacer hover

---

## Fundamentos Teóricos

| Módulo | Teoría principal | Referencia |
|--------|----------------|------------|
| M1 | Análisis técnico | Murphy (1999) — Technical Analysis of the Financial Markets |
| M2 | Hechos estilizados | Cont (2001) — Empirical properties of asset returns |
| M3 | Volatilidad condicional | Engle (1982) — ARCH; Bollerslev (1986) — GARCH |
| M4 | CAPM | Sharpe (1964), Lintner (1965), Mossin (1966) |
| M5 | VaR y coherencia | Artzner et al. (1999) — Coherent Measures of Risk |
| M5 | Backtesting | Kupiec (1995) — Techniques for Verifying the Accuracy of VaR |
| M6 | Teoría de portafolios | Markowitz (1952) — Portfolio Selection |
| M7 | Señales técnicas | Elder (1993) — Trading for a Living |
| M8 | Performance attribution | Jensen (1968), Grinold & Kahn (2000) |

---

## Decisiones de Diseño y Limitaciones

**Por qué log-rendimientos:**  
`rₜ = ln(Pₜ/Pₜ₋₁)` son aditivos en el tiempo y más estables estadísticamente. Para horizontes cortos, `rₜ ≈ pct_change`. Se ofrece la opción de rendimientos simples en el Módulo 2.

**Por qué yfinance y no Quandl/Bloomberg:**  
Acceso gratuito sin API key, con datos suficientes para el horizonte académico (3 años diarios). Las versiones recientes (≥1.2.0) estabilizan la API tras cambios en Yahoo Finance.

**Limitaciones conocidas:**
- Los datos de Yahoo Finance pueden tener gaps en días festivos de EE.UU.; se aplica `ffill()` para imputar valores faltantes.
- La optimización de Markowitz asume pesos no negativos (no se permite posición corta).
- El test de Kupiec tiene baja potencia en muestras pequeñas (< 250 días).
- Los indicadores macro (USDCOP=X) pueden no estar disponibles en todos los entornos de red.
- La distribución normal en Montecarlo subestima colas pesadas; en producción se reemplazaría por distribución t-Student.

---

## Referencias Bibliográficas

- Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203–228.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236.
- Elder, A. (1993). *Trading for a Living*. Wiley.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
- Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
- Jensen, M. C. (1968). The performance of mutual funds in the period 1945–1964. *Journal of Finance*, 23(2), 389–416.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- Lintner, J. (1965). The valuation of risk assets and the selection of risky investments in stock portfolios and capital budgets. *Review of Economics and Statistics*, 47(1), 13–37.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
- Mossin, J. (1966). Equilibrium in a capital asset market. *Econometrica*, 34(4), 768–783.
- Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.
- Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347–370.
- Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under conditions of risk. *Journal of Finance*, 19(3), 425–442.
- Sharpe, W. F. (1966). Mutual fund performance. *Journal of Business*, 39(1), 119–138.

---

*RiskLab · Universidad Santo Tomás · Bogotá · 2026*  
*Proyecto Integrador — Teoría del Riesgo Financiero*
