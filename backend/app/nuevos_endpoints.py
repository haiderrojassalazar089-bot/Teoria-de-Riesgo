"""
backend/app/main_nuevos_endpoints.py
NUEVOS ENDPOINTS para tickers dinámicos del S&P 500.
Agrega estos endpoints al main.py existente.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy import stats

from .config import Settings, get_settings
from .models import (
    SP500ListResponse, SP500TickerInfo,
    MonteCarloRequest, MonteCarloResponse,
    DueloRequest, DueloResponse, DueloMetricas,
    MaquinaTiempoResponse, DynamicAnalysisRequest,
    PortfolioRequest, VaRResponse,
    FronteraRequest, FronteraResponse, PortafolioOptimo,
)
from .dependencies import get_data_service, get_risk_calculator, get_portfolio_analyzer
from .services import DataService, RiskCalculator, PortfolioAnalyzer
from .sp500_service import get_sp500_info, get_sp500_tickers

logger = logging.getLogger(__name__)


def _safe_float(v) -> float:
    try:
        f = float(v)
        return 0.0 if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return 0.0

# ── Pega estos endpoints en tu main.py existente ─────────────


# GET /sp500/tickers — lista completa del S&P 500
async def get_sp500_list(
    sector: str = Query(default=None, description="Filtrar por sector GICS"),
    search: str = Query(default=None, description="Buscar por ticker o nombre"),
) -> SP500ListResponse:
    """
    Retorna la lista completa de tickers del S&P 500 con nombre y sector.
    Permite filtrar por sector o buscar por texto.
    """
    infos = get_sp500_info()
    if sector:
        infos = [i for i in infos if sector.lower() in i["sector"].lower()]
    if search:
        q = search.lower()
        infos = [i for i in infos if q in i["ticker"].lower() or q in i["name"].lower()]
    return SP500ListResponse(
        total=len(infos),
        tickers=[SP500TickerInfo(**i) for i in infos]
    )


# POST /montecarlo — simulación de trayectorias
async def post_montecarlo(
    body: MonteCarloRequest,
    data: DataService = Depends(get_data_service),
) -> MonteCarloResponse:
    """
    Simula N trayectorias del portafolio para el horizonte indicado.
    Retorna percentiles 5, 50, 95 y probabilidad de pérdida.
    """
    try:
        prices = data.get_multi_close(body.tickers, years=body.years_history)
        prices = prices[body.tickers]
        log_ret = np.log(prices / prices.shift(1)).dropna()
        w = np.array(body.weights)
        port_r = log_ret.values @ w
        mu = port_r.mean()
        sigma = port_r.std()
        cov = log_ret.cov().values

        np.random.seed(42)
        H = body.horizon_days
        N = body.n_simulations
        n_assets = len(body.tickers)

        # Cholesky para correlaciones
        try:
            L = np.linalg.cholesky(cov * 252 / 252)
        except np.linalg.LinAlgError:
            L = None

        trayectorias = []
        for _ in range(N):
            if L is not None:
                z = np.random.standard_normal((H, n_assets))
                r_sim = (log_ret.mean().values + (z @ L.T))
                port_sim = r_sim @ w
            else:
                port_sim = np.random.normal(mu, sigma, H)
            # Valor acumulado base 100
            tray = 100 * np.cumprod(1 + port_sim)
            trayectorias.append(tray.tolist())

        arr = np.array(trayectorias)
        p5  = np.percentile(arr, 5,  axis=0).tolist()
        p50 = np.percentile(arr, 50, axis=0).tolist()
        p95 = np.percentile(arr, 95, axis=0).tolist()

        # Fechas simuladas (días hábiles desde hoy)
        hoy = datetime.today()
        fechas = []
        d = hoy
        count = 0
        while count < H:
            d += timedelta(days=1)
            if d.weekday() < 5:
                fechas.append(d.strftime("%Y-%m-%d"))
                count += 1

        prob_perdida = float((arr[:, -1] < 100).mean())
        retorno_esp  = float(arr[:, -1].mean() / 100 - 1)
        var_h = float(-np.percentile(arr[:, -1] / 100 - 1, 5))

        # Downsample trayectorias para no enviar demasiados datos
        step = max(1, N // 200)
        tray_sample = [trayectorias[i] for i in range(0, N, step)]

        return MonteCarloResponse(
            tickers=body.tickers,
            weights=body.weights,
            horizon_days=H,
            n_simulations=N,
            trayectorias=tray_sample,
            fechas_sim=fechas,
            percentil_5=p5,
            percentil_50=p50,
            percentil_95=p95,
            prob_perdida=round(prob_perdida, 4),
            retorno_esperado=round(retorno_esp, 4),
            var_horizonte=round(var_h, 4),
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error Monte Carlo: {e}")


# POST /duelo — enfrentar dos portafolios
async def post_duelo(
    body: DueloRequest,
    data: DataService = Depends(get_data_service),
) -> DueloResponse:
    """
    Enfrenta dos portafolios en Sharpe, VaR, Beta, Alpha y Drawdown.
    Retorna veredicto, puntos y resumen narrativo.
    """
    try:
        all_tickers = list(set(body.portafolio_a.tickers + body.portafolio_b.tickers + ["^GSPC"]))
        prices = data.get_multi_close(all_tickers, years=body.years)
        log_ret = np.log(prices / prices.shift(1)).dropna()
        rf_data = data.get_rf()
        rf_daily = rf_data["daily"]
        bench = log_ret["^GSPC"] if "^GSPC" in log_ret.columns else log_ret.iloc[:, 0]

        def calc_metrics(req, name):
            w = np.array(req.weights)
            t = req.tickers
            avail = [x for x in t if x in log_ret.columns]
            r = log_ret[avail].values @ w[:len(avail)]
            ann_ret = float(r.mean() * 252)
            vol = float(r.std() * np.sqrt(252))
            sharpe = (ann_ret - rf_daily * 252) / vol if vol > 0 else 0
            cum = (1 + pd.Series(r)).cumprod()
            dd = float((cum / cum.cummax() - 1).min())
            var95 = float(-np.percentile(r, 5))
            aligned = pd.concat([pd.Series(r), bench], axis=1).dropna()
            slope, intercept, *_ = stats.linregress(aligned.iloc[:, 1], aligned.iloc[:, 0])
            return DueloMetricas(
                nombre=name, tickers=t, weights=list(w),
                retorno_anual=round(ann_ret, 4),
                volatilidad=round(vol, 4),
                sharpe=round(sharpe, 4),
                max_drawdown=round(dd, 4),
                var_95=round(var95, 4),
                beta=round(float(slope), 4),
                alpha=round(float(intercept * 252), 4),
                ganador_metricas={},
            )

        a = calc_metrics(body.portafolio_a, "Portafolio A")
        b = calc_metrics(body.portafolio_b, "Portafolio B")

        # Determinar ganador por métrica
        comparaciones = {
            "Retorno anual":  (a.retorno_anual,  b.retorno_anual,  True),
            "Sharpe":         (a.sharpe,          b.sharpe,          True),
            "Volatilidad":    (a.volatilidad,     b.volatilidad,     False),
            "Max Drawdown":   (a.max_drawdown,    b.max_drawdown,    False),
            "VaR 95%":        (a.var_95,           b.var_95,           False),
            "Alpha":          (a.alpha,            b.alpha,            True),
        }
        puntos_a = puntos_b = 0
        gan_a, gan_b = {}, {}
        for metrica, (va, vb, higher_better) in comparaciones.items():
            if higher_better:
                if va > vb + 0.001:   w_m = "A"; puntos_a += 1
                elif vb > va + 0.001: w_m = "B"; puntos_b += 1
                else:                 w_m = "empate"
            else:
                if va < vb - 0.001:   w_m = "A"; puntos_a += 1
                elif vb < va - 0.001: w_m = "B"; puntos_b += 1
                else:                 w_m = "empate"
            gan_a[metrica] = w_m
            gan_b[metrica] = w_m

        a.ganador_metricas = gan_a
        b.ganador_metricas = gan_b
        veredicto = "A" if puntos_a > puntos_b else "B" if puntos_b > puntos_a else "empate"

        resumen = (
            f"El Portafolio {veredicto} gana con {max(puntos_a, puntos_b)} de 6 métricas. "
            f"{'Mejor retorno ajustado por riesgo (Sharpe superior).' if veredicto == 'A' and a.sharpe > b.sharpe else ''}"
            f"{'Mayor retorno con menor riesgo.' if veredicto == 'B' and b.sharpe > a.sharpe else ''}"
            f"{'Resultado muy ajustado — ambos portafolios son competitivos.' if veredicto == 'empate' else ''}"
        )

        return DueloResponse(
            portafolio_a=a, portafolio_b=b,
            veredicto=veredicto,
            puntos_a=puntos_a, puntos_b=puntos_b,
            resumen=resumen,
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error duelo: {e}")


# POST /maquina-tiempo — portafolio en un período histórico
async def post_maquina_tiempo(
    body: DynamicAnalysisRequest,
    data: DataService = Depends(get_data_service),
) -> MaquinaTiempoResponse:
    """
    Reconstruye el comportamiento del portafolio en un rango de fechas histórico.
    Máx. 10 años hacia atrás.
    """
    try:
        import yfinance as yf
        all_t = body.tickers + ["^GSPC"]
        start = body.start_date or (datetime.today() - timedelta(days=365 * body.years)).strftime("%Y-%m-%d")
        end   = body.end_date   or datetime.today().strftime("%Y-%m-%d")

        raw = yf.download(all_t, start=start, end=end, auto_adjust=True,
                          progress=False, multi_level_index=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw
        prices = prices.dropna(how="all").ffill().dropna()

        if prices.empty:
            raise HTTPException(status_code=404, detail="Sin datos para el período solicitado.")

        fechas = [d.strftime("%Y-%m-%d") for d in prices.index]
        norm = prices / prices.iloc[0] * 100

        retornos_norm = {}
        estadisticas = {}
        for t in body.tickers:
            if t not in norm.columns:
                continue
            serie = norm[t].dropna()
            retornos_norm[t] = [round(float(v), 4) for v in serie]
            log_r = np.log(prices[t] / prices[t].shift(1)).dropna()
            jb_s, jb_p = stats.jarque_bera(log_r.values)
            estadisticas[t] = {
                "retorno_total":   round(float(prices[t].iloc[-1] / prices[t].iloc[0] - 1), 4),
                "volatilidad":     round(float(log_r.std() * np.sqrt(252)), 4),
                "max_drawdown":    round(float(((norm[t] / norm[t].cummax()) - 1).min()), 4),
                "sharpe_aprox":    round(float(log_r.mean() * 252 / (log_r.std() * np.sqrt(252))), 4),
                "jarque_bera_p":   round(float(jb_p), 4),
            }

        bench_norm = [round(float(v), 4) for v in norm["^GSPC"].dropna()] if "^GSPC" in norm.columns else []
        retornos_totales = {t: estadisticas[t]["retorno_total"] for t in estadisticas}
        mejor = max(retornos_totales, key=retornos_totales.get) if retornos_totales else ""
        peor  = min(retornos_totales, key=retornos_totales.get) if retornos_totales else ""

        return MaquinaTiempoResponse(
            tickers=body.tickers,
            start_date=start,
            end_date=end,
            n_dias=len(prices),
            retornos_norm=retornos_norm,
            fechas=fechas,
            estadisticas=estadisticas,
            benchmark_norm=bench_norm,
            mejor_activo=mejor,
            peor_activo=peor,
        )
    except HTTPException:
        raise
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error máquina del tiempo: {e}")