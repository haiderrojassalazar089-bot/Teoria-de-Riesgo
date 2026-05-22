# 📊 RiskLab — Sistema de Análisis de Riesgo Financiero

**Universidad Santo Tomás**  
**Teoría de Riesgo — 2026**

**Autores:**  
- Haider Rojas Salazar  
- Natalia González  


**Profesor:** Javier Mauricio Sierra

---

## 📋 Descripción

RiskLab es una plataforma completa de análisis de riesgo financiero que integra **11 módulos temáticos** para gestión de portafolios, valoración de derivados y análisis cuantitativo avanzado. El sistema está construido sobre una **arquitectura de 5 capas** con backend FastAPI deployado en Render, frontend Streamlit interactivo, base de datos SQLAlchemy, y pipeline de Machine Learning para señales de trading.

### Activos Seleccionados

| Ticker | Nombre | Sector | Justificación |
|--------|--------|--------|---------------|
| **JPM** | JPMorgan Chase | Financiero | Representante del sector bancario, alta liquidez |
| **BAC** | Bank of America | Financiero | Diversificación dentro del sector financiero |
| **GS** | Goldman Sachs | Financiero | Exposición a banca de inversión |
| **MS** | Morgan Stanley | Financiero | Complemento en servicios financieros |
| **V** | Visa | Tecnología Financiera | Crecimiento secular en pagos digitales |

**Justificación del Portafolio:** Enfoque sectorial en servicios financieros con diversificación entre banca comercial, banca de inversión y fintech. Alta correlación intra-sector permite demostrar técnicas de optimización de Markowitz y análisis de riesgo sistémico.

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA 5: PRESENTACIÓN                      │
│              Streamlit Frontend (Puerto 8501)                │
│     11 Módulos: Técnico, Vol, VaR, CAPM, Markowitz, etc.    │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CAPA 4: API (FastAPI)                      │
│              Backend RESTful (Puerto 8002)                   │
│        14+ Endpoints | Swagger /docs | ReDoc /redoc         │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                CAPA 3: LÓGICA DE NEGOCIO                     │
│   Services: DataService, RiskCalculator, TechnicalIndicators│
│   ML: ModelPredictor (Singleton), OptionPricer, StressTester│
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              CAPA 2: PERSISTENCIA (SQLAlchemy)               │
│    SQLite: PrecioCache, ConsultaLog, PortafolioGuardado,    │
│                      PredictionLog                           │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              CAPA 1: FUENTES EXTERNAS                        │
│     Yahoo Finance (yfinance) | FRED API | Alpha Vantage     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Instalación Local

### Prerrequisitos

- Python 3.11.9+
- Git
- (Opcional) Docker 20.10+

### 1. Clonar repositorio

```bash
git clone https://github.com/tu-usuario/Teoria-de-Riesgo.git
cd Teoria-de-Riesgo
```

### 2. Configurar variables de entorno

```bash
cp backend/.env.example backend/.env
```

Edita `backend/.env` con tus API keys:

```env
# API Keys (obtener en los siguientes enlaces)
FRED_API_KEY=tu_key_aqui          # https://fred.stlouisfed.org/docs/api/api_key.html
ALPHAVANTAGE_API_KEY=tu_key_aqui  # https://www.alphavantage.co/support/#api-key

# Base de datos
DATABASE_URL=sqlite:///./risklab.db

# Configuración
ALLOWED_ORIGINS=http://localhost:8501,http://localhost:3000
```

### 3. Backend — Instalación

```bash
cd backend

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Entrenar modelo ML
python -m app.ml.train

# Iniciar servidor
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

**Backend disponible en:** http://localhost:8002  
**Swagger UI:** http://localhost:8002/docs  
**ReDoc:** http://localhost:8002/redoc

### 4. Frontend — Instalación

```bash
cd frontend

# Instalar dependencias (si no usas el mismo venv del backend)
pip install -r requirements.txt

# Iniciar Streamlit
streamlit run app.py
```

**Frontend disponible en:** http://localhost:8501

---

## 🐳 Instalación con Docker

### Opción 1: docker-compose (Recomendado)

```bash
# Desde la raíz del proyecto
docker-compose up

# Detener
docker-compose down
```

### Opción 2: Docker manual

```bash
# Build
cd backend
docker build -t risklab-backend:latest .

# Run
docker run -d \
  --name risklab-backend \
  -p 8002:8000 \
  -e DATABASE_URL=sqlite:///./risklab.db \
  risklab-backend:latest

# Logs
docker logs -f risklab-backend
```

Ver [DOCKER.md](DOCKER.md) para documentación completa de Docker.

---

## 🧪 Tests

```bash
cd backend

# Ejecutar todos los tests
pytest tests/ -v

# Con coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# Solo tests unitarios
pytest tests/test_unit.py -v

# Solo tests de integración
pytest tests/test_integration.py -v
```

**Tests implementados (8):**
- ✅ Unit: RSI calculation
- ✅ Unit: VaR paramétrico vs analítico
- ✅ Unit: Black-Scholes paridad put-call
- ✅ Unit: SMA calculation
- ✅ Integration: GET /precios/{ticker}
- ✅ Integration: POST /var con validación de pesos
- ✅ Integration: GET / health check
- ✅ Integration: GET /activos

---

## 🤖 Machine Learning

### Propósito Analítico

**Clasificación de señales de trading (BUY / HOLD / SELL)** basado en indicadores técnicos.

**Features:**
1. RSI (14 períodos)
2. MACD Histogram
3. Retorno 5 días
4. Retorno 10 días
5. Volatilidad 20 días

**Modelo:** RandomForestClassifier (100 árboles, max_depth=10)  
**Pipeline:** StandardScaler → RandomForest  
**Accuracy:** ~74.5% en test set

### Entrenar el modelo

```bash
cd backend
python -m app.ml.train
```

Output esperado:
```
[1/5] Construyendo features...
  → 2000 muestras, 5 features
[5/5] Guardando modelo en model.joblib...
  → Modelo guardado ✓
```

### Endpoint ML

```bash
# Ejemplo de llamada
curl -X POST http://localhost:8002/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "features": [65.5, 1.2, 0.015, 0.022, 0.18]
  }'

# Response
{
  "ticker": "AAPL",
  "signal": "BUY",
  "signal_code": 2,
  "confidence": 0.87,
  "probabilities": {"SELL": 0.05, "HOLD": 0.08, "BUY": 0.87},
  "model_version": "v1.0.0"
}
```

**Patrón Singleton:** El modelo se carga UNA sola vez al iniciar el servidor y se reutiliza en todos los requests, verificable en los logs con `[ModelPredictor] ✅ Modelo cargado`.

---

## 🌐 Despliegue en Producción

### Backend Deployado (Render)

**URL:** `https://tu-app.onrender.com` _(reemplazar con tu URL real)_

**Endpoints disponibles:**
- GET `/` — Health check
- GET `/docs` — Swagger UI
- GET `/redoc` — ReDoc
- GET `/activos` — Lista de activos
- GET `/precios/{ticker}` — Precios históricos
- POST `/var` — Cálculo de VaR
- POST `/ml/predict` — Predicción ML

### CI/CD

GitHub Actions ejecuta automáticamente:
- ✅ Tests con pytest
- ✅ Linting con flake8
- ✅ Build de imagen Docker

Ver: `.github/workflows/ci.yml`

---

## 📊 Módulos del Sistema

### Backend (14+ Endpoints)

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/activos` | GET | Lista de activos con precios actuales |
| `/precios/{ticker}` | GET | Precios históricos OHLCV |
| `/rendimientos/{ticker}` | GET | Retornos logarítmicos y simples |
| `/indicadores/{ticker}` | GET | Indicadores técnicos (RSI, MACD, etc.) |
| `/var` | POST | VaR paramétrico y Monte Carlo |
| `/capm` | GET | Beta, alpha, Sharpe ratio |
| `/frontera-eficiente` | POST | Optimización de Markowitz |
| `/alertas` | GET | Señales de trading |
| `/macro` | GET | Tasa libre de riesgo y benchmark |
| `/ml/predict` | POST | Predicción de señal de trading |
| `/ml/model-info` | GET | Metadata del modelo ML |

### Frontend (11 Módulos)

1. **M1 · Portafolio** — Selección de activos
2. **M2 · Análisis Técnico** — RSI, MACD, Bandas de Bollinger
3. **M3 · Rendimientos** — Estadística descriptiva
4. **M4 · CAPM** — Beta, Alpha, Sharpe Ratio
5. **M5 · VaR + Kupiec** — VaR paramétrico, histórico, Monte Carlo
6. **M6 · Optimización** — Frontera eficiente de Markowitz
7. **M7 · Volatilidad** — EWMA y GARCH(1,1)
8. **M8 · Alertas** — Señales de compra/venta
9. **M11 · Macro** — Benchmark y tasa libre de riesgo
10. **M12 · Renta Fija** — Nelson-Siegel + Bonos
11. **M13 · Opciones** — Black-Scholes + Greeks
12. **M14 · Stress Testing** — Escenarios extremos

---

## 🛠️ Stack Tecnológico

**Backend:**
- FastAPI 0.115.0
- Pydantic 2.9.2 (validación)
- SQLAlchemy 2.0.35 (ORM)
- scikit-learn 1.5.2 (ML)
- yfinance 0.2.55 (datos)
- pytest 9.0.3 (testing)

**Frontend:**
- Streamlit 1.38.0
- Plotly 5.22.0
- Pandas 2.2.2
- NumPy 1.26.4

**DevOps:**
- Docker + docker-compose
- GitHub Actions (CI/CD)
- Render (deployment)

---

## 🤖 Uso de Herramientas de IA

Este proyecto fue desarrollado con asistencia de **Claude (Anthropic)** para:

- **Arquitectura del código:** Diseño del patrón de inyección de dependencias y estructura de carpetas modular
- **Debugging:** Resolución de errores de validación Pydantic y queries SQLAlchemy
- **Optimización:** Refactorización del código para reducir duplicación
- **Documentación:** Generación de docstrings y este README
- **Tests:** Implementación de fixtures pytest y casos de prueba

**Contribución humana:**
- Decisiones metodológicas (selección de activos, modelos GARCH, VaR)
- Interpretación de resultados financieros
- Validación de cálculos contra referencias académicas
- Diseño de la experiencia de usuario del frontend
- Integración y testing end-to-end

---

## 📄 Licencia

Este proyecto es material académico para el curso de Teoría de Riesgo en la Universidad Santo Tomás.

---

## 📞 Contacto

Para preguntas sobre el proyecto, contactar a:
- Haider Rojas Salazar
- Natalia González
