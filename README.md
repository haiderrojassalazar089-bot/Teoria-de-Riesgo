# RiskLab · USTA
### Tablero Interactivo de Análisis de Riesgo Financiero

**Universidad Santo Tomás · Teoría del Riesgo · Prof. Javier Mauricio Sierra**

> Proyecto integrador que implementa un tablero de análisis de riesgo financiero con arquitectura backend/frontend separada. El backend FastAPI sirve como motor de cálculo y el frontend Streamlit consume los endpoints para visualizar los resultados.

---

## Autores

| Nombre | Programa |
|--------|---------|
| Haider Rojas Salazar | Estadística · USTA |

---

## Descripción

RiskLab es un sistema de análisis de riesgo financiero compuesto por:

- **Backend FastAPI** — motor de cálculo con 9 endpoints, modelos Pydantic, inyección de dependencias y configuración con BaseSettings
- **Frontend Streamlit** — tablero interactivo con 8 módulos temáticos que consume el backend via HTTP

Los datos se obtienen dinámicamente desde **Yahoo Finance** sin datasets estáticos. La tasa libre de riesgo se actualiza automáticamente desde `^IRX` (T-Bill 3M).

---

## Stack Tecnológico

| Componente | Tecnología | Versión |
|------------|-----------|---------|
| Backend API | FastAPI | 0.136.0 |
| Servidor ASGI | Uvicorn | 0.44.0 |
| Validación | Pydantic + pydantic-settings | 2.13.1 |
| Frontend | Streamlit | 1.43.2 |
| Visualización | Plotly | 5.22.0 |
| Datos de mercado | yfinance | 1.2.0 |
| Modelos ARCH/GARCH | arch | 7.0.0 |
| Optimización | PyPortfolioOpt | 1.5.6 |
| Estadística | scipy + statsmodels | 1.13.1 / 0.14.2 |
| Lenguaje | Python | 3.12.1 |

---

## Activos Analizados

| Ticker | Empresa | Sector |
|--------|---------|--------|
| AAPL | Apple Inc. | Tecnología |
| JPM | JPMorgan Chase & Co. | Financiero |
| XOM | ExxonMobil Corporation | Energía |
| JNJ | Johnson & Johnson | Salud |
| AMZN | Amazon.com Inc. | Consumo discrecional |
| ^GSPC | S&P 500 | Benchmark |

**Horizonte:** 3 años de datos diarios · **Tasa libre de riesgo:** ^IRX (T-Bill 3M)

---

## Estructura del Proyecto

```
Teoria-de-Riesgo/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI app — 9 endpoints async
│   │   ├── models.py         # Modelos Pydantic request/response
│   │   ├── services.py       # Lógica de negocio — clases POO + @timer_log
│   │   ├── dependencies.py   # Depends() — inyección de servicios
│   │   └── config.py         # BaseSettings — carga desde .env
│   ├── .env                  # Variables de entorno (NO subir a Git)
│   ├── .env.example          # Plantilla de variables de entorno
│   └── requirements.txt      # Dependencias del backend
├── frontend/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # Loader directo (legacy)
│   │   └── client.py         # Cliente HTTP → backend FastAPI
│   ├── pages/
│   │   ├── overview.py       # Vista General
│   │   ├── m1_technical.py   # Módulo 1: Análisis Técnico
│   │   ├── m2_returns.py     # Módulo 2: Rendimientos
│   │   ├── m3_garch.py       # Módulo 3: ARCH/GARCH
│   │   ├── m4_capm.py        # Módulo 4: CAPM & Beta
│   │   ├── m5_var.py         # Módulo 5: VaR & CVaR
│   │   ├── m6_markowitz.py   # Módulo 6: Markowitz
│   │   ├── m7_signals.py     # Módulo 7: Señales & Alertas
│   │   └── m8_macro.py       # Módulo 8: Macro & Benchmark
│   ├── utils/
│   │   ├── theme.py          # Paleta de colores y configuración Plotly
│   │   └── styles.py         # CSS global del tablero
│   └── app.py                # Punto de entrada Streamlit
├── .env.example              # Plantilla global de variables de entorno
├── .gitignore                # Excluye .env, __pycache__, etc.
├── README.md
└── requirements.txt          # Dependencias del frontend
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/haiderrojassalazar089/Teoria-de-Riesgo.git
cd Teoria-de-Riesgo
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
# .venv\Scripts\activate         # Windows
```

### 3. Instalar dependencias

```bash
# Frontend
pip install -r requirements.txt

# Backend
pip install -r backend/requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example backend/.env
# Editar backend/.env con tus API keys (opcional — yfinance no requiere key)
```

---

## Ejecución

El proyecto requiere **dos terminales** corriendo simultáneamente.

### Terminal 1 — Backend FastAPI

```bash
cd backend
uvicorn app.main:app --reload --port 8002
```

El backend estará disponible en:
- API: `http://localhost:8002`
- Documentación Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`

### Terminal 2 — Frontend Streamlit

```bash
cd frontend
python -m streamlit run app.py
```

El tablero estará disponible en: `http://localhost:8501`

---

## Variables de Entorno

Crear el archivo `backend/.env` basado en `backend/.env.example`:

| Variable | Descripción | Requerida | Default |
|----------|-------------|-----------|---------|
| `ALPHA_VANTAGE_KEY` | API key de Alpha Vantage | No | `demo` |
| `FRED_API_KEY` | API key de FRED (Federal Reserve) | No | `""` |
| `DEFAULT_YEARS` | Años de historia por defecto | No | `3` |
| `VAR_CONFIDENCE` | Nivel de confianza VaR | No | `0.95` |
| `MC_SIMULATIONS` | Simulaciones Montecarlo | No | `10000` |
| `SMA_PERIOD` | Período SMA por defecto | No | `20` |
| `EMA_PERIOD` | Período EMA por defecto | No | `21` |
| `RSI_PERIOD` | Período RSI por defecto | No | `14` |
| `CACHE_TTL_SECONDS` | TTL del caché en segundos | No | `1800` |

> **Nota:** yfinance no requiere API key. Los datos se obtienen gratuitamente desde Yahoo Finance.

---

## Endpoints del Backend

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Health check — verifica que la API está activa |
| `/activos` | GET | Lista los 5 activos del portafolio con precios actuales |
| `/precios/{ticker}` | GET | Precios históricos OHLCV de un activo |
| `/rendimientos/{ticker}` | GET | Rendimientos simples y logarítmicos con estadísticas |
| `/indicadores/{ticker}` | GET | SMA, EMA, Bollinger, RSI, MACD, Estocástico |
| `/capm` | GET | Beta y rendimiento esperado CAPM de todos los activos |
| `/macro` | GET | Indicadores macroeconómicos actualizados (Rf, S&P 500) |
| `/alertas` | GET | Señales activas de compra/venta por activo |
| `/var` | POST | VaR paramétrico, histórico, Montecarlo y CVaR |
| `/frontera-eficiente` | POST | Frontera eficiente de Markowitz y portafolios óptimos |

Documentación interactiva completa en `/docs` (Swagger UI).

---

## Módulos del Tablero

| Módulo | Peso | Contenido |
|--------|------|-----------|
| Vista General | — | KPIs del portafolio, rendimiento normalizado, correlaciones, tabla de activos |
| M1 Análisis Técnico | 10% | SMA, EMA, Bollinger, RSI, MACD, Estocástico — interactivos |
| M2 Rendimientos | 6% | Estadísticas descriptivas, histograma, Q-Q plot, boxplot, Jarque-Bera, Shapiro-Wilk |
| M3 ARCH/GARCH | 10% | ARCH(1), GARCH(1,1), GJR-GARCH, EGARCH — AIC/BIC, diagnóstico, pronóstico |
| M4 CAPM & Beta | 6% | Beta MCO, SML, descomposición de riesgo, clasificación activos |
| M5 VaR & CVaR | 10% | Paramétrico, histórico, Montecarlo (10,000 sim.), CVaR, backtesting Kupiec |
| M6 Markowitz | 10% | Frontera eficiente, mínima varianza, máximo Sharpe, composición óptima |
| M7 Señales & Alertas | 7% | RSI, MACD, Bollinger, SMA cross, Estocástico — semáforo por activo |
| M8 Macro & Benchmark | 6% | Alpha Jensen, Tracking Error, Information Ratio, VIX, USD/COP |

---

## APIs y Fuentes de Datos

| Fuente | Uso | Acceso |
|--------|-----|--------|
| Yahoo Finance (`yfinance`) | Precios históricos OHLCV | Gratuito, sin API key |
| Yahoo Finance (`^IRX`) | Tasa libre de riesgo T-Bill 3M | Gratuito, sin API key |
| Yahoo Finance (`^GSPC`) | Benchmark S&P 500 | Gratuito, sin API key |
| Yahoo Finance (`^VIX`, `^TNX`) | Indicadores macro | Gratuito, sin API key |
| Yahoo Finance (`USDCOP=X`, `EURUSD=X`) | Tipos de cambio | Gratuito, sin API key |

---

## Arquitectura

```
┌─────────────────────────────────────────────────┐
│              Frontend (Streamlit)                │
│  Vista General · M1-M8 · Navegación sidebar     │
└──────────────────┬──────────────────────────────┘
                   │ HTTP (requests)
                   ▼
┌─────────────────────────────────────────────────┐
│              Backend (FastAPI)                   │
│  9 endpoints · Pydantic · Depends() · async     │
│  BaseSettings · @timer_log · POO                │
└──────────────────┬──────────────────────────────┘
                   │ yfinance
                   ▼
┌─────────────────────────────────────────────────┐
│           Yahoo Finance API                      │
│  Precios · Tasas · Macro · Volatilidad          │
└─────────────────────────────────────────────────┘
```

---

## Buenas Prácticas Implementadas

- ✅ **Decorador personalizado** `@timer_log` — logging de tiempo de ejecución en `services.py`
- ✅ **POO** — clases `DataService`, `TechnicalIndicators`, `RiskCalculator`, `PortfolioAnalyzer`, `AlertasService`
- ✅ **Type hints** — anotaciones de tipo en todo el backend y cliente HTTP
- ✅ **`@field_validator`** — validación personalizada en `PortfolioRequest` (pesos sumen 1.0)
- ✅ **`Depends()`** — inyección de dependencias para todos los servicios
- ✅ **`BaseSettings`** — configuración desde `.env`, nunca hardcodeada
- ✅ **Rutas `async`** — todos los endpoints del backend son asíncronos
- ✅ **Status codes HTTP** — 400, 404, 503 apropiados
- ✅ **Caché** — `st.cache_data(ttl=1800)` en el cliente HTTP
- ✅ **CORS** — configurado para permitir el frontend
- ✅ **`.gitignore`** — excluye `.env`, `__pycache__`, datos temporales
- ✅ **`requirements.txt`** — versiones fijas en backend y frontend
- ✅ **Documentación automática** — Swagger UI en `/docs`, enriquecida con `Field(description=...)`

---

## Uso de IA

Este proyecto utilizó **Claude (Anthropic)** como asistente de desarrollo para:
- Arquitectura del sistema backend/frontend
- Implementación de modelos financieros (VaR, GARCH, Markowitz)
- Debugging y resolución de errores
- Diseño del CSS y paleta de colores del tablero

Todos los conceptos financieros y estadísticos fueron validados por el equipo.

---

*RiskLab · Universidad Santo Tomás · Bogotá, Colombia · 2026*