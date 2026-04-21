# RiskLab · USTA
### Tablero Interactivo de Análisis de Riesgo Financiero

**Universidad Santo Tomás · Teoría del Riesgo · Prof. Javier Mauricio Sierra**

> Proyecto integrador que implementa un tablero de análisis de riesgo financiero con arquitectura backend/frontend separada. El backend FastAPI sirve como motor de cálculo y el frontend Streamlit consume los endpoints para visualizar los resultados.

---

## Autores

| Nombre | Programa |
|--------|---------|
| Haider Rojas Salazar | Estadística · USTA |
| Natalia González | Estadística · USTA |
| Camilo Ostios | Estadística · USTA |

---

## Descripción

RiskLab es un sistema de análisis de riesgo financiero compuesto por:

- **Backend FastAPI** — motor de cálculo con 14 endpoints, modelos Pydantic, inyección de dependencias y configuración con BaseSettings
- **Frontend Streamlit** — tablero interactivo con 11 módulos temáticos (M1–M8 + M9 Monte Carlo Visual, M10 Duelo de Portafolios, M11 Máquina del Tiempo) que consume el backend via HTTP
- **Selector dinámico del S&P 500** — el usuario puede seleccionar entre 2 y 10 empresas del S&P 500; todos los módulos se adaptan automáticamente

Los datos se obtienen dinámicamente desde **Yahoo Finance** sin datasets estáticos. La tasa libre de riesgo se actualiza automáticamente desde `^IRX` (T-Bill 3M).

---

## Stack Tecnológico

| Componente | Tecnología | Versión |
|------------|-----------|---------|
| Backend API | FastAPI | 0.136.0 |
| Servidor ASGI | Uvicorn | 0.45.0 |
| Validación | Pydantic + pydantic-settings | 2.13.x |
| Frontend | Streamlit | 1.56.0 |
| Visualización | Plotly | 6.7.0 |
| Datos de mercado | yfinance | 1.3.0 |
| Modelos ARCH/GARCH | arch | 8.0.0 |
| Optimización | PyPortfolioOpt | 1.6.0 |
| Estadística | scipy + statsmodels | 1.17.x / 0.14.x |
| Lenguaje | Python | 3.14 (Render) / 3.12 (local) |

---

## Activos Analizados

El portafolio es **dinámico** — el usuario selecciona entre 2 y 10 empresas del S&P 500 desde el Selector de Activos. Las sugerencias predefinidas incluyen:

| Grupo | Tickers |
|-------|---------|
| Big Tech | AAPL, MSFT, GOOGL, META, AMZN |
| Financiero | JPM, BAC, GS, MS, V |
| Salud | JNJ, UNH, PFE, ABBV, MRK |
| Energía | XOM, CVX, COP, SLB, EOG |
| Consumo | WMT, HD, MCD, SBUX, NKE |
| Telecomunicaciones | T, VZ, CMCSA, NFLX, DIS |

**Benchmark:** ^GSPC (S&P 500) · **Tasa libre de riesgo:** ^IRX (T-Bill 3M)

> **Nota sobre costo computacional:** Se recomienda un máximo de **8–10 empresas**. Con más activos, los módulos M3 (GARCH), M6 (Markowitz con 50k simulaciones) y M8 pueden tardar 30–60 segundos en instancias gratuitas de Render (0.1 CPU, 512 MB RAM). En entorno local el límite práctico es 10 activos sin degradación notable.

---

## Estructura del Proyecto

```
Teoria-de-Riesgo/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app — 14 endpoints async
│   │   ├── nuevos_endpoints.py   # Endpoints M9–M11: Monte Carlo, Duelo, Máquina del Tiempo
│   │   ├── models.py             # Modelos Pydantic request/response
│   │   ├── services.py           # Lógica de negocio — clases POO + @timer_log
│   │   ├── dependencies.py       # Depends() — inyección de servicios
│   │   ├── config.py             # BaseSettings — carga desde .env
│   │   └── sp500_service.py      # Lista dinámica del S&P 500 con fallback robusto
│   └── Procfile                  # Comando de arranque para Render
├── frontend/
│   ├── data/
│   │   ├── __init__.py
│   │   └── client.py             # Cliente HTTP → backend FastAPI
│   ├── pages/
│   │   ├── selector.py           # Selector dinámico de activos del S&P 500
│   │   ├── overview.py           # Vista General — tickers dinámicos
│   │   ├── m1_technical.py       # Módulo 1: Análisis Técnico
│   │   ├── m2_returns.py         # Módulo 2: Rendimientos — KPIs en tarjetas
│   │   ├── m3_garch.py           # Módulo 3: ARCH/GARCH — tarjetas por modelo
│   │   ├── m4_capm.py            # Módulo 4: CAPM & Beta
│   │   ├── m5_var.py             # Módulo 5: VaR & CVaR
│   │   ├── m6_markowitz.py       # Módulo 6: Markowitz — matriz corr. + KPIs
│   │   ├── m7_signals.py         # Módulo 7: Señales & Alertas — selector de período
│   │   ├── m8_macro.py           # Módulo 8: Macro & Benchmark
│   │   ├── m9_montecarlo.py      # Módulo 9: Monte Carlo Visual
│   │   ├── m10_duelo.py          # Módulo 10: Duelo de Portafolios independientes
│   │   └── m11_tiempo.py         # Módulo 11: Máquina del Tiempo
│   ├── utils/
│   │   ├── theme.py              # Paleta de colores y configuración Plotly
│   │   ├── styles.py             # CSS global del tablero
│   │   └── dynamic_tickers.py    # Utilidades para tickers dinámicos
│   ├── app.py                    # Punto de entrada Streamlit
│   └── Procfile                  # Comando de arranque para Render
├── runtime.txt                   # Versión de Python para Render
├── .gitignore
├── README.md
└── requirements.txt              # Dependencias unificadas
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/haiderrojassalazar089-bot/Teoria-de-Riesgo.git
cd Teoria-de-Riesgo
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecución

El proyecto requiere **dos terminales** corriendo simultáneamente.

### Terminal 1 — Backend FastAPI

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

Disponible en:
- API: `http://localhost:8002`
- Swagger UI: `http://localhost:8002/docs`

### Terminal 2 — Frontend Streamlit

```bash
cd frontend
python -m streamlit run app.py
```

Tablero en: `http://localhost:8501`

---

## Endpoints del Backend

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/activos` | GET | Activos del portafolio con precios actuales |
| `/precios/{ticker}` | GET | Precios históricos OHLCV |
| `/rendimientos/{ticker}` | GET | Rendimientos simples y logarítmicos |
| `/indicadores/{ticker}` | GET | SMA, EMA, Bollinger, RSI, MACD, Estocástico |
| `/capm` | GET | Beta y rendimiento esperado CAPM |
| `/macro` | GET | Indicadores macroeconómicos (Rf, S&P 500) |
| `/alertas` | GET | Señales activas de compra/venta |
| `/var` | POST | VaR paramétrico, histórico, Montecarlo y CVaR |
| `/frontera-eficiente` | POST | Frontera eficiente de Markowitz |
| `/sp500/tickers` | GET | Lista completa del S&P 500 con nombre y sector |
| `/montecarlo` | POST | Simulación Monte Carlo con trayectorias |
| `/duelo` | POST | Duelo entre dos portafolios independientes |
| `/maquina-tiempo` | POST | Portafolio en período histórico personalizado |

---

## Módulos del Tablero

| Módulo | Contenido |
|--------|-----------|
| Selector de Activos | Selección dinámica de 2–10 empresas del S&P 500 |
| Vista General | KPIs del portafolio, rendimiento normalizado, correlaciones |
| M1 Análisis Técnico | SMA, EMA, Bollinger, RSI, MACD, Estocástico |
| M2 Rendimientos | Estadísticas en tarjetas, histograma, Q-Q plot, Jarque-Bera |
| M3 ARCH/GARCH | ARCH, GARCH, GJR-GARCH, EGARCH — tarjetas comparativas |
| M4 CAPM & Beta | Beta MCO, SML, descomposición de riesgo |
| M5 VaR & CVaR | Paramétrico, histórico, Montecarlo, CVaR, Kupiec |
| M6 Markowitz | Frontera eficiente, matriz correlación, portafolios óptimos |
| M7 Señales & Alertas | Semáforo por activo · Selector de período (1 mes a 3 años) |
| M8 Macro & Benchmark | Alpha Jensen, Tracking Error, Information Ratio, VIX |
| M9 Monte Carlo Visual | 1,000 trayectorias, percentiles, probabilidad de pérdida |
| M10 Duelo de Portafolios | Dos portafolios independientes del S&P 500 enfrentados |
| M11 Máquina del Tiempo | Período histórico libre — incluye Guerra/Aranceles 2025 |

---

## Buenas Prácticas Implementadas

- ✅ **Decorador personalizado** `@timer_log` — logging de tiempo en `services.py`
- ✅ **POO** — clases `DataService`, `TechnicalIndicators`, `RiskCalculator`, `PortfolioAnalyzer`, `AlertasService`
- ✅ **Type hints** — anotaciones completas en backend y cliente HTTP
- ✅ **`@field_validator`** — validación de pesos, fechas y tickers S&P 500
- ✅ **`Depends()`** — inyección de dependencias para todos los servicios
- ✅ **`BaseSettings`** — configuración desde `.env`
- ✅ **Rutas `async`** — todos los endpoints son asíncronos
- ✅ **Tickers dinámicos** — `session_state` propaga la selección a todos los módulos
- ✅ **Fallback robusto** — `sp500_service.py` con 100+ empresas hardcodeadas si Wikipedia no responde
- ✅ **MultiIndex yfinance** — manejo correcto de columnas en `nuevos_endpoints.py`
- ✅ **Caché** — `st.cache_data(ttl=1800)` en el cliente HTTP
- ✅ **CORS** — configurado para permitir el frontend

---

## Uso de IA

Este proyecto utilizó **Claude (Anthropic)** como asistente de desarrollo para:
- Arquitectura del sistema backend/frontend
- Implementación de modelos financieros (VaR, GARCH, Markowitz, Monte Carlo)
- Módulos avanzados M9–M11 (Monte Carlo Visual, Duelo de Portafolios, Máquina del Tiempo)
- Selector dinámico del S&P 500 con fallback
- Debugging de errores yfinance MultiIndex
- Diseño del CSS y paleta de colores del tablero
- Despliegue en Render con Python 3.14

Todos los conceptos financieros y estadísticos fueron validados por el equipo.

---

*RiskLab · Universidad Santo Tomás · Bogotá, Colombia · 2026*