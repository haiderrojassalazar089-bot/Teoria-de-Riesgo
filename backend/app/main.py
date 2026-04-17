"""
backend/app/main.py
FastAPI — Motor de cálculo del RiskLab USTA
9 endpoints con rutas async, Pydantic, Depends() y manejo de errores HTTP.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .models import (
    ActivosResponse, ActivoInfo,
    PreciosResponse, PrecioItem,
    RendimientosResponse, RendimientoStats,
    IndicadoresResponse,
    VaRResponse, PortfolioRequest,
    CAPMResponse,
    FronteraResponse, FronteraRequest,
    AlertasResponse, AlertaItem,
    MacroResponse, MacroIndicador,
    ErrorResponse,
)
from .dependencies import (
    get_config, get_data_service, get_technical_indicators,
    get_risk_calculator, get_portfolio_analyzer, get_alertas_service,
)
from .services import (
    DataService, TechnicalIndicators, RiskCalculator,
    PortfolioAnalyzer, AlertasService, SECTOR_MAP, NOMBRE_MAP,
)
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "RiskLab API — USTA",
    description = (
        "Backend de análisis de riesgo financiero. "
        "Proporciona indicadores técnicos, modelos de riesgo, CAPM y optimización "
        "de portafolios para el tablero RiskLab."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ──────────────────────────────────────────────────────
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.allowed_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Utilidades internas ───────────────────────────────────────

def _safe_float(v) -> float:
    """Convierte a float manejando NaN/inf."""
    try:
        f = float(v)
        return 0.0 if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return 0.0

def _descriptive(r: pd.Series) -> RendimientoStats:
    jb_s, jb_p = stats.jarque_bera(r.values)
    return RendimientoStats(
        media_diaria      = round(_safe_float(r.mean()), 6),
        media_anualizada  = round(_safe_float(r.mean() * 252), 6),
        std_diaria        = round(_safe_float(r.std()), 6),
        std_anualizada    = round(_safe_float(r.std() * np.sqrt(252)), 6),
        asimetria         = round(_safe_float(stats.skew(r.values)), 4),
        curtosis          = round(_safe_float(stats.kurtosis(r.values)), 4),
        minimo            = round(_safe_float(r.min()), 6),
        maximo            = round(_safe_float(r.max()), 6),
        jarque_bera_stat  = round(_safe_float(jb_s), 4),
        jarque_bera_p     = round(_safe_float(jb_p), 6),
        n_obs             = len(r),
    )


# ════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════

@app.get("/", summary="Health check")
async def root():
    """Verifica que la API está activa."""
    return {
        "status"  : "ok",
        "servicio": "RiskLab API — USTA",
        "version" : "1.0.0",
        "docs"    : "/docs",
    }


# ── GET /activos ──────────────────────────────────────────────
@app.get(
    "/activos",
    response_model = ActivosResponse,
    summary        = "Lista activos del portafolio",
    tags           = ["Portafolio"],
)
async def get_activos(
    cfg  : Settings    = Depends(get_config),
    data : DataService = Depends(get_data_service),
) -> ActivosResponse:
    """Retorna los activos disponibles con precio actual y variación del día."""
    try:
        prices = data.get_multi_close(cfg.tickers, years=1)
        activos_list = []
        for t in cfg.tickers:
            if t not in prices.columns:
                continue
            px   = prices[t].dropna()
            ult  = _safe_float(px.iloc[-1])
            prev = _safe_float(px.iloc[-2]) if len(px) >= 2 else ult
            chg  = (ult - prev) / prev if prev != 0 else 0.0
            activos_list.append(ActivoInfo(
                ticker     = t,
                sector     = SECTOR_MAP.get(t, "Otro"),
                nombre     = NOMBRE_MAP.get(t, t),
                ultimo     = round(ult, 2),
                cambio_hoy = round(chg, 6),
            ))
        return ActivosResponse(
            activos   = activos_list,
            benchmark = cfg.benchmark,
            total     = len(activos_list),
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ── GET /precios/{ticker} ─────────────────────────────────────
@app.get(
    "/precios/{ticker}",
    response_model = PreciosResponse,
    summary        = "Precios históricos OHLCV",
    tags           = ["Mercado"],
)
async def get_precios(
    ticker : str,
    years  : int        = Query(default=3, ge=1, le=10, description="Años de historia"),
    data   : DataService = Depends(get_data_service),
    cfg    : Settings    = Depends(get_config),
) -> PreciosResponse:
    """Retorna serie OHLCV completa para el ticker solicitado."""
    ticker = ticker.upper()
    valid  = cfg.tickers + [cfg.benchmark]
    if ticker not in valid:
        raise HTTPException(
            status_code = 404,
            detail      = f"Ticker '{ticker}' no encontrado. Disponibles: {valid}",
        )
    try:
        df = data.get_ohlcv(ticker, years=years)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Sin datos para '{ticker}'")

    precios = [
        PrecioItem(
            fecha  = idx.strftime("%Y-%m-%d"),
            open   = round(_safe_float(row["Open"]),   2),
            high   = round(_safe_float(row["High"]),   2),
            low    = round(_safe_float(row["Low"]),    2),
            close  = round(_safe_float(row["Close"]),  2),
            volume = int(row["Volume"]) if not np.isnan(row["Volume"]) else 0,
        )
        for idx, row in df.iterrows()
    ]
    return PreciosResponse(
        ticker     = ticker,
        start_date = df.index[0].strftime("%Y-%m-%d"),
        end_date   = df.index[-1].strftime("%Y-%m-%d"),
        n_dias     = len(df),
        precios    = precios,
    )


# ── GET /rendimientos/{ticker} ────────────────────────────────
@app.get(
    "/rendimientos/{ticker}",
    response_model = RendimientosResponse,
    summary        = "Rendimientos simples y logarítmicos",
    tags           = ["Análisis"],
)
async def get_rendimientos(
    ticker : str,
    years  : int         = Query(default=3, ge=1, le=10),
    data   : DataService = Depends(get_data_service),
    cfg    : Settings    = Depends(get_config),
) -> RendimientosResponse:
    """Retorna rendimientos calculados con estadísticas descriptivas."""
    ticker = ticker.upper()
    if ticker not in cfg.tickers + [cfg.benchmark]:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado.")
    try:
        df = data.get_ohlcv(ticker, years=years)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

    close   = df["Close"]
    log_r   = np.log(close / close.shift(1)).dropna()
    sim_r   = close.pct_change().dropna()
    fechas  = [d.strftime("%Y-%m-%d") for d in log_r.index]

    return RendimientosResponse(
        ticker         = ticker,
        log_returns    = [round(_safe_float(v), 6) for v in log_r],
        simple_returns = [round(_safe_float(v), 6) for v in sim_r],
        fechas         = fechas,
        stats_log      = _descriptive(log_r),
        stats_simple   = _descriptive(sim_r),
    )


# ── GET /indicadores/{ticker} ─────────────────────────────────
@app.get(
    "/indicadores/{ticker}",
    response_model = IndicadoresResponse,
    summary        = "Indicadores técnicos completos",
    tags           = ["Análisis"],
)
async def get_indicadores(
    ticker : str,
    years  : int                 = Query(default=2, ge=1, le=5),
    data   : DataService         = Depends(get_data_service),
    ti     : TechnicalIndicators = Depends(get_technical_indicators),
    cfg    : Settings            = Depends(get_config),
) -> IndicadoresResponse:
    """SMA, EMA, Bollinger, RSI, MACD y Estocástico para el ticker."""
    ticker = ticker.upper()
    if ticker not in cfg.tickers + [cfg.benchmark]:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado.")
    try:
        df  = data.get_ohlcv(ticker, years=years)
        ind = ti.compute_all(df)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return IndicadoresResponse(ticker=ticker, **ind)


# ── POST /var ─────────────────────────────────────────────────
@app.post(
    "/var",
    response_model = VaRResponse,
    summary        = "VaR y CVaR del portafolio",
    tags           = ["Riesgo"],
)
async def post_var(
    body : PortfolioRequest,
    risk : RiskCalculator   = Depends(get_risk_calculator),
    cfg  : Settings         = Depends(get_config),
) -> VaRResponse:
    """Calcula VaR paramétrico, histórico, Montecarlo y CVaR."""
    try:
        result = risk.compute_var(
            tickers    = body.tickers,
            weights    = body.weights,
            confidence = body.confidence,
            years      = body.years,
            n_sim      = cfg.mc_simulations,
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en cálculo VaR: {e}")

    return VaRResponse(
        tickers    = body.tickers,
        weights    = body.weights,
        confidence = body.confidence,
        **result,
    )


# ── GET /capm ─────────────────────────────────────────────────
@app.get(
    "/capm",
    response_model = CAPMResponse,
    summary        = "CAPM y Beta de todos los activos",
    tags           = ["Riesgo"],
)
async def get_capm(
    years    : int               = Query(default=3, ge=1, le=10),
    analyzer : PortfolioAnalyzer = Depends(get_portfolio_analyzer),
    cfg      : Settings          = Depends(get_config),
) -> CAPMResponse:
    """Beta, alpha, R², rendimiento esperado y clasificación para cada activo."""
    try:
        result = analyzer.compute_capm(cfg.tickers, cfg.benchmark, years)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

    from .models import CAPMItem
    return CAPMResponse(
        rf_display = result["rf_display"],
        rf_annual  = result["rf_annual"],
        rf_source  = result["rf_source"],
        rf_date    = result["rf_date"],
        benchmark  = result["benchmark"],
        rm_annual  = result["rm_annual"],
        activos    = [CAPMItem(**a) for a in result["activos"]],
    )


# ── POST /frontera-eficiente ──────────────────────────────────
@app.post(
    "/frontera-eficiente",
    response_model = FronteraResponse,
    summary        = "Frontera eficiente de Markowitz",
    tags           = ["Optimización"],
)
async def post_frontera(
    body     : FronteraRequest,
    analyzer : PortfolioAnalyzer = Depends(get_portfolio_analyzer),
) -> FronteraResponse:
    """Simula portafolios aleatorios y construye la frontera eficiente."""
    try:
        result = analyzer.compute_frontera(
            tickers       = body.tickers,
            years         = body.years,
            n_portfolios  = body.n_portfolios,
            target_return = body.target_return,
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en optimización: {e}")

    from .models import PortafolioOptimo
    return FronteraResponse(
        tickers          = result["tickers"],
        n_simulaciones   = result["n_simulaciones"],
        retornos         = result["retornos"],
        volatilidades    = result["volatilidades"],
        sharpes          = result["sharpes"],
        pesos_simulados  = result["pesos_simulados"],
        min_varianza     = PortafolioOptimo(**result["min_varianza"]),
        max_sharpe       = PortafolioOptimo(**result["max_sharpe"]),
        objetivo         = PortafolioOptimo(**result["objetivo"]) if result["objetivo"] else None,
    )


# ── GET /alertas ──────────────────────────────────────────────
@app.get(
    "/alertas",
    response_model = AlertasResponse,
    summary        = "Señales activas de compra/venta",
    tags           = ["Señales"],
)
async def get_alertas(
    alertas_svc : AlertasService = Depends(get_alertas_service),
    cfg         : Settings       = Depends(get_config),
) -> AlertasResponse:
    """Evalúa RSI, MACD, Bollinger, SMA y Estocástico para todos los activos."""
    try:
        raw = alertas_svc.compute_alertas(cfg.tickers)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    alertas = [AlertaItem(**a) for a in raw]
    resumen = {
        "COMPRA" : sum(1 for a in alertas if a.señal == "COMPRA"),
        "VENTA"  : sum(1 for a in alertas if a.señal == "VENTA"),
        "NEUTRAL": sum(1 for a in alertas if a.señal == "NEUTRAL"),
    }
    return AlertasResponse(
        fecha   = datetime.today().strftime("%Y-%m-%d"),
        alertas = alertas,
        resumen = resumen,
    )


# ── GET /macro ────────────────────────────────────────────────
@app.get(
    "/macro",
    response_model = MacroResponse,
    summary        = "Indicadores macroeconómicos actualizados",
    tags           = ["Macro"],
)
async def get_macro(
    data : DataService = Depends(get_data_service),
    cfg  : Settings    = Depends(get_config),
) -> MacroResponse:
    """Tasa libre de riesgo, retorno y volatilidad del benchmark desde API."""
    try:
        rf     = data.get_rf()
        prices = data.get_ohlcv(cfg.benchmark, years=3)
        bm     = np.log(prices["Close"] / prices["Close"].shift(1)).dropna()
        bm_ret = float(bm.mean() * 252)
        bm_vol = float(bm.std() * np.sqrt(252))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return MacroResponse(
        tasa_libre_riesgo = MacroIndicador(
            nombre  = "Tasa Libre de Riesgo",
            valor   = round(rf["annual"], 6),
            display = rf["display"],
            fuente  = rf["source"],
            fecha   = rf["date"],
            unidad  = "% anual",
        ),
        benchmark_retorno = MacroIndicador(
            nombre  = f"Retorno anualizado {cfg.benchmark}",
            valor   = round(bm_ret, 6),
            display = f"{bm_ret:.2%}",
            fuente  = "Yahoo Finance",
            fecha   = datetime.today().strftime("%Y-%m-%d"),
            unidad  = "% anual",
        ),
        benchmark_vol = MacroIndicador(
            nombre  = f"Volatilidad anualizada {cfg.benchmark}",
            valor   = round(bm_vol, 6),
            display = f"{bm_vol:.2%}",
            fuente  = "Yahoo Finance",
            fecha   = datetime.today().strftime("%Y-%m-%d"),
            unidad  = "% anual",
        ),
    )