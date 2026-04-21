"""
backend/app/main.py
FastAPI — Motor de cálculo del RiskLab USTA
Endpoints originales + nuevos endpoints dinámicos S&P 500.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy import stats

from .config import Settings, get_settings
from .models import (
    ActivosResponse, ActivoInfo,
    PreciosResponse, PrecioItem,
    RendimientosResponse, RendimientoStats,
    IndicadoresResponse,
    VaRResponse, PortfolioRequest,
    CAPMResponse, CAPMItem,
    FronteraResponse, FronteraRequest, PortafolioOptimo,
    AlertasResponse, AlertaItem,
    MacroResponse, MacroIndicador,
    SP500ListResponse, SP500TickerInfo,
    MonteCarloRequest, MonteCarloResponse,
    DueloRequest, DueloResponse, DueloMetricas,
    MaquinaTiempoResponse, DynamicAnalysisRequest,
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
from .sp500_service import get_sp500_info, get_sp500_tickers
from .nuevos_endpoints import (
    get_sp500_list, post_montecarlo, post_duelo, post_maquina_tiempo
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "RiskLab API — USTA",
    description = "Backend de análisis de riesgo financiero con tickers dinámicos del S&P 500.",
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.allowed_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


def _safe_float(v) -> float:
    try:
        f = float(v)
        return 0.0 if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return 0.0


def _descriptive(r: pd.Series) -> RendimientoStats:
    jb_s, jb_p = stats.jarque_bera(r.values)
    return RendimientoStats(
        media_diaria     = round(_safe_float(r.mean()), 6),
        media_anualizada = round(_safe_float(r.mean() * 252), 6),
        std_diaria       = round(_safe_float(r.std()), 6),
        std_anualizada   = round(_safe_float(r.std() * np.sqrt(252)), 6),
        asimetria        = round(_safe_float(stats.skew(r.values)), 4),
        curtosis         = round(_safe_float(stats.kurtosis(r.values)), 4),
        minimo           = round(_safe_float(r.min()), 6),
        maximo           = round(_safe_float(r.max()), 6),
        jarque_bera_stat = round(_safe_float(jb_s), 4),
        jarque_bera_p    = round(_safe_float(jb_p), 6),
        n_obs            = len(r),
    )


# ════════════════════════════════════════════════
# ENDPOINTS ORIGINALES
# ════════════════════════════════════════════════

@app.get("/", summary="Health check")
async def root():
    return {"status": "ok", "servicio": "RiskLab API — USTA", "version": "2.0.0", "docs": "/docs"}


@app.get("/activos", response_model=ActivosResponse, tags=["Portafolio"])
async def get_activos(
    tickers: str = Query(default="AAPL,JPM,XOM,JNJ,AMZN", description="Tickers separados por coma"),
    cfg: Settings    = Depends(get_config),
    data: DataService = Depends(get_data_service),
) -> ActivosResponse:
    """Retorna activos con precio actual. Acepta tickers dinámicos del S&P 500."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        ticker_list = cfg.tickers
    try:
        prices = data.get_multi_close(ticker_list, years=1)
        activos_list = []
        sp500_info = {i["ticker"]: i for i in get_sp500_info()}
        for t in ticker_list:
            if t not in prices.columns:
                continue
            px   = prices[t].dropna()
            ult  = _safe_float(px.iloc[-1])
            prev = _safe_float(px.iloc[-2]) if len(px) >= 2 else ult
            chg  = (ult - prev) / prev if prev != 0 else 0.0
            info = sp500_info.get(t, {})
            activos_list.append(ActivoInfo(
                ticker     = t,
                sector     = info.get("sector", SECTOR_MAP.get(t, "S&P 500")),
                nombre     = info.get("name",   NOMBRE_MAP.get(t, t)),
                ultimo     = round(ult, 2),
                cambio_hoy = round(chg, 6),
            ))
        return ActivosResponse(activos=activos_list, benchmark=cfg.benchmark, total=len(activos_list))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/precios/{ticker}", response_model=PreciosResponse, tags=["Mercado"])
async def get_precios(
    ticker: str,
    years:  int         = Query(default=3, ge=1, le=10),
    data:   DataService = Depends(get_data_service),
) -> PreciosResponse:
    """Precios históricos OHLCV — acepta cualquier ticker del S&P 500."""
    ticker = ticker.upper().replace(".", "-")
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


@app.get("/rendimientos/{ticker}", response_model=RendimientosResponse, tags=["Análisis"])
async def get_rendimientos(
    ticker: str,
    years:  int          = Query(default=3, ge=1, le=10),
    data:   DataService  = Depends(get_data_service),
) -> RendimientosResponse:
    ticker = ticker.upper().replace(".", "-")
    try:
        df = data.get_ohlcv(ticker, years=years)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    close  = df["Close"]
    log_r  = np.log(close / close.shift(1)).dropna()
    sim_r  = close.pct_change().dropna()
    fechas = [d.strftime("%Y-%m-%d") for d in log_r.index]
    return RendimientosResponse(
        ticker         = ticker,
        log_returns    = [round(_safe_float(v), 6) for v in log_r],
        simple_returns = [round(_safe_float(v), 6) for v in sim_r],
        fechas         = fechas,
        stats_log      = _descriptive(log_r),
        stats_simple   = _descriptive(sim_r),
    )


@app.get("/indicadores/{ticker}", response_model=IndicadoresResponse, tags=["Análisis"])
async def get_indicadores(
    ticker: str,
    years:  int                  = Query(default=2, ge=1, le=5),
    data:   DataService          = Depends(get_data_service),
    ti:     TechnicalIndicators  = Depends(get_technical_indicators),
) -> IndicadoresResponse:
    ticker = ticker.upper().replace(".", "-")
    try:
        df  = data.get_ohlcv(ticker, years=years)
        ind = ti.compute_all(df)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return IndicadoresResponse(ticker=ticker, **ind)


@app.post("/var", response_model=VaRResponse, tags=["Riesgo"])
async def post_var(
    body: PortfolioRequest,
    risk: RiskCalculator = Depends(get_risk_calculator),
    cfg:  Settings       = Depends(get_config),
) -> VaRResponse:
    """VaR con tickers dinámicos validados contra el S&P 500."""
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
        raise HTTPException(status_code=400, detail=f"Error VaR: {e}")
    return VaRResponse(tickers=body.tickers, weights=body.weights, confidence=body.confidence, **result)


@app.get("/capm", response_model=CAPMResponse, tags=["Riesgo"])
async def get_capm(
    tickers:  str               = Query(default="AAPL,JPM,XOM,JNJ,AMZN"),
    years:    int               = Query(default=3, ge=1, le=10),
    analyzer: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
    cfg:      Settings          = Depends(get_config),
) -> CAPMResponse:
    """CAPM con tickers dinámicos."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    try:
        result = analyzer.compute_capm(ticker_list, cfg.benchmark, years)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return CAPMResponse(
        rf_display = result["rf_display"],
        rf_annual  = result["rf_annual"],
        rf_source  = result["rf_source"],
        rf_date    = result["rf_date"],
        benchmark  = result["benchmark"],
        rm_annual  = result["rm_annual"],
        activos    = [CAPMItem(**a) for a in result["activos"]],
    )


@app.post("/frontera-eficiente", response_model=FronteraResponse, tags=["Optimización"])
async def post_frontera(
    body:     FronteraRequest,
    analyzer: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
) -> FronteraResponse:
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
        raise HTTPException(status_code=400, detail=f"Error optimización: {e}")
    return FronteraResponse(
        tickers         = result["tickers"],
        n_simulaciones  = result["n_simulaciones"],
        retornos        = result["retornos"],
        volatilidades   = result["volatilidades"],
        sharpes         = result["sharpes"],
        pesos_simulados = result["pesos_simulados"],
        min_varianza    = PortafolioOptimo(**result["min_varianza"]),
        max_sharpe      = PortafolioOptimo(**result["max_sharpe"]),
        objetivo        = PortafolioOptimo(**result["objetivo"]) if result["objetivo"] else None,
    )


@app.get("/alertas", response_model=AlertasResponse, tags=["Señales"])
async def get_alertas(
    tickers:     str           = Query(default="AAPL,JPM,XOM,JNJ,AMZN"),
    alertas_svc: AlertasService = Depends(get_alertas_service),
) -> AlertasResponse:
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    try:
        raw = alertas_svc.compute_alertas(ticker_list)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    alertas = [AlertaItem(**a) for a in raw]
    resumen = {
        "COMPRA" : sum(1 for a in alertas if a.señal == "COMPRA"),
        "VENTA"  : sum(1 for a in alertas if a.señal == "VENTA"),
        "NEUTRAL": sum(1 for a in alertas if a.señal == "NEUTRAL"),
    }
    return AlertasResponse(fecha=datetime.today().strftime("%Y-%m-%d"), alertas=alertas, resumen=resumen)


@app.get("/macro", response_model=MacroResponse, tags=["Macro"])
async def get_macro(
    data: DataService = Depends(get_data_service),
    cfg:  Settings    = Depends(get_config),
) -> MacroResponse:
    try:
        rf     = data.get_rf()
        prices = data.get_ohlcv(cfg.benchmark, years=3)
        bm     = np.log(prices["Close"] / prices["Close"].shift(1)).dropna()
        bm_ret = float(bm.mean() * 252)
        bm_vol = float(bm.std() * np.sqrt(252))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return MacroResponse(
        tasa_libre_riesgo = MacroIndicador(nombre="Tasa Libre de Riesgo", valor=round(rf["annual"], 6),
            display=rf["display"], fuente=rf["source"], fecha=rf["date"], unidad="% anual"),
        benchmark_retorno = MacroIndicador(nombre=f"Retorno anualizado {cfg.benchmark}",
            valor=round(bm_ret, 6), display=f"{bm_ret:.2%}", fuente="Yahoo Finance",
            fecha=datetime.today().strftime("%Y-%m-%d"), unidad="% anual"),
        benchmark_vol = MacroIndicador(nombre=f"Volatilidad anualizada {cfg.benchmark}",
            valor=round(bm_vol, 6), display=f"{bm_vol:.2%}", fuente="Yahoo Finance",
            fecha=datetime.today().strftime("%Y-%m-%d"), unidad="% anual"),
    )


# ════════════════════════════════════════════════
# NUEVOS ENDPOINTS
# ════════════════════════════════════════════════

@app.get("/sp500/tickers", response_model=SP500ListResponse, tags=["S&P 500"])
async def sp500_list(
    sector: str = Query(default=None),
    search: str = Query(default=None),
):
    """Lista completa del S&P 500 con nombre, sector y opción de búsqueda."""
    return await get_sp500_list(sector=sector, search=search)


@app.post("/montecarlo", response_model=MonteCarloResponse, tags=["Riesgo"])
async def montecarlo(
    body: MonteCarloRequest,
    data: DataService = Depends(get_data_service),
):
    """Simulación Monte Carlo con trayectorias del portafolio."""
    return await post_montecarlo(body=body, data=data)


@app.post("/duelo", response_model=DueloResponse, tags=["Análisis"])
async def duelo(
    body: DueloRequest,
    data: DataService = Depends(get_data_service),
):
    """Enfrenta dos portafolios en todas las métricas de riesgo."""
    return await post_duelo(body=body, data=data)


@app.post("/maquina-tiempo", response_model=MaquinaTiempoResponse, tags=["Análisis"])
async def maquina_tiempo(
    body: DynamicAnalysisRequest,
    data: DataService = Depends(get_data_service),
):
    """Reconstruye el portafolio en un período histórico con calendario."""
    return await post_maquina_tiempo(body=body, data=data)