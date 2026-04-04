# RiskLab · USTA
### Tablero Interactivo de Análisis de Riesgo Financiero
**Universidad Santo Tomás · Teoría del Riesgo · Prof. Javier Mauricio Sierra**

---

## Descripción

RiskLab es un tablero web interactivo construido con **Streamlit** y **Python 3.12** que implementa los principales modelos de análisis de riesgo financiero vistos en el curso. Los datos se obtienen dinámicamente desde **Yahoo Finance** sin necesidad de datasets estáticos.

---

## Stack Tecnológico

| Componente | Tecnología |
|------------|-----------|
| Framework web | Streamlit 1.43.2 |
| Visualización | Plotly 5.22.0 |
| Datos de mercado | yfinance 1.2.0 |
| Modelos ARCH/GARCH | arch 7.0.0 |
| Optimización de portafolio | PyPortfolioOpt 1.5.6 |
| Estadística | scipy 1.13.1 · statsmodels 0.14.2 |
| Lenguaje | Python 3.12.1 |

---

## Activos Analizados

| Ticker | Empresa | Sector |
|--------|---------|--------|
| AAPL | Apple Inc. | Tecnología |
| JPM | JPMorgan Chase | Financiero |
| XOM | ExxonMobil | Energía |
| JNJ | Johnson & Johnson | Salud |
| AMZN | Amazon | Consumo discrecional |
| ^GSPC | S&P 500 | Benchmark |

**Horizonte:** 3 años de datos diarios · **Tasa libre de riesgo:** ^IRX (T-Bill 3M, Yahoo Finance)

---

## Estructura del Proyecto

```
Teoria-de-Riesgo/
├── app.py                    # Punto de entrada principal
├── requirements.txt          # Dependencias
├── README.md
├── data/
│   ├── __init__.py
│   └── loader.py             # Descarga y caché de datos (yfinance 1.2.0)
├── utils/
│   ├── __init__.py
│   ├── theme.py              # Paleta de colores y configuración Plotly
│   └── styles.py             # CSS global del tablero
└── pages/
    ├── __init__.py
    ├── overview.py           # Vista General del portafolio
    ├── m1_technical.py       # Módulo 1: Análisis Técnico
    ├── m2_returns.py         # Módulo 2: Rendimientos
    ├── m3_garch.py           # Módulo 3: ARCH/GARCH
    ├── m4_capm.py            # Módulo 4: CAPM & Beta
    ├── m5_var.py             # Módulo 5: VaR & CVaR (próximo)
    ├── m6_markowitz.py       # Módulo 6: Markowitz (próximo)
    ├── m7_signals.py         # Módulo 7: Señales & Alertas (próximo)
    └── m8_macro.py           # Módulo 8: Macro & Benchmark (próximo)
```

---

## Módulos Implementados

### ◈ Vista General
- KPIs del portafolio: retorno acumulado, volatilidad anual, Ratio de Sharpe, máximo drawdown
- Gráfico de rendimientos normalizados (base 100)
- Matriz de correlación entre activos
- Tabla resumen con precios, variación diaria, YTD y volatilidad anual

### Módulo 1 · Análisis Técnico `12%`
- Selector dinámico de activos con datos en tiempo real
- Gráfico de precios con línea o velas japonesas
- Medias móviles: SMA y EMA con parámetros ajustables
- Bandas de Bollinger
- RSI con zonas de sobrecompra/sobreventa
- MACD con histograma y línea de señal
- Oscilador Estocástico %K/%D
- Paneles de interpretación académica por indicador

### Módulo 2 · Rendimientos `8%`
- Cálculo de rendimientos simples y logarítmicos
- Estadísticas descriptivas: media, desviación estándar, asimetría, curtosis
- Histograma con curva normal superpuesta
- Q-Q Plot vs distribución normal
- Volatility clustering (gráfico r²)
- Boxplot comparativo de los 5 activos
- Pruebas de normalidad: Jarque-Bera y Shapiro-Wilk
- Discusión de hechos estilizados: colas pesadas, agrupamiento de volatilidad, asimetría negativa, efecto apalancamiento

### Módulo 3 · ARCH/GARCH `12%`
- Prueba ARCH-LM para justificación formal
- Ajuste de 4 especificaciones: ARCH(1), GARCH(1,1), GJR-GARCH(1,1,1), EGARCH(1,1)
- Tabla comparativa: Log-Likelihood, AIC y BIC
- Gráfico de rendimientos y volatilidad condicional estimada
- Parámetros estimados con persistencia α+β
- Diagnóstico: residuos estandarizados, Q-Q plot, Jarque-Bera sobre residuos
- Pronóstico de volatilidad N-pasos con banda de incertidumbre

### Módulo 4 · CAPM & Beta `8%`
- Beta calculado por MCO con intervalo de confianza 95%
- Gráfico de dispersión activo vs S&P 500 con línea de regresión
- Tasa libre de riesgo automática desde ^IRX (Yahoo Finance API)
- Tabla resumen: β, α, R², E[R] anual, clasificación agresivo/defensivo/neutro
- Security Market Line (SML) con todos los activos posicionados
- Descomposición del riesgo sistemático vs idiosincrático
- Discusión de diversificación y limitaciones del CAPM

---

## Instalación y Uso

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd Teoria-de-Riesgo

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el tablero
streamlit run app.py
```

El tablero estará disponible en `http://localhost:8501`

---

## APIs y Fuentes de Datos

| Fuente | Uso | Acceso |
|--------|-----|--------|
| Yahoo Finance (`yfinance`) | Precios históricos OHLCV | Gratuito, sin API key |
| Yahoo Finance (`^IRX`) | Tasa libre de riesgo (T-Bill 3M) | Gratuito, sin API key |
| Yahoo Finance (`^GSPC`) | Benchmark S&P 500 | Gratuito, sin API key |

> Los datos se actualizan automáticamente con caché de 30 minutos (`st.cache_data(ttl=1800)`).

---

## Diseño

Paleta **Minimalismo Académico Refinado** — inspirada en terminales financieras profesionales:

- Fondo: `#EEF1F6` (gris azulado claro)
- Superficie: `#FFFFFF` (blanco limpio)
- Acento principal: `#8B6914` (dorado antiguo)
- Positivo: `#1A6B4A` (verde musgo)
- Negativo: `#8B2A2A` (rojo vino)
- Tipografía: **Playfair Display** (títulos) · **IBM Plex Mono** (datos) · **Inter** (UI)

---

*RiskLab · Universidad Santo Tomás · 2026*
