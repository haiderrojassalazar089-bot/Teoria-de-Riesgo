"""
backend/app/services.py
Lógica de negocio encapsulada en clases con type hints completos.
Decorador personalizado para logging de tiempo de ejecución.
"""

from __future__ import annotations
import time
import logging
import functools
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Decorador personalizado: logging de tiempo ───────────────
def timer_log(func):
    """Decorador que registra el tiempo de ejecución de cada servicio."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[{func.__qualname__}] ejecutado en {elapsed:.3f}s")
        return result
    return wrapper


# ── Constantes ────────────────────────────────────────────────
SECTOR_MAP: dict[str, str] = {
    "AAPL": "Tecnología",
    "JPM" : "Financiero",
    "XOM" : "Energía",
    "JNJ" : "Salud",
    "AMZN": "Consumo discrecional",
}

NOMBRE_MAP: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "JPM" : "JPMorgan Chase & Co.",
    "XOM" : "ExxonMobil Corporation",
    "JNJ" : "Johnson & Johnson",
    "AMZN": "Amazon.com Inc.",
}


# ════════════════════════════════════════════════
# DataService — descarga de datos
# ════════════════════════════════════════════════

class DataService:
    """Servicio de conexión a Yahoo Finance (yfinance 1.2.0)."""

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df.columns.name = None
        df.index.name   = "Date"
        return df.dropna()

    @timer_log
    def get_ohlcv(self, ticker: str, years: int = 3) -> pd.DataFrame:
        """Descarga OHLCV completo para un ticker."""
        start = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        try:
            df = yf.download(ticker, start=start, auto_adjust=True,
                             progress=False, multi_level_index=False)
            return self._clean(df)
        except Exception as e:
            raise ConnectionError(f"No se pudo descargar {ticker}: {e}") from e

    @timer_log
    def get_multi_close(self, tickers: list[str], years: int = 3) -> pd.DataFrame:
        """Descarga precios de cierre para múltiples tickers."""
        all_tickers = list(set(tickers))
        start = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        try:
            raw = yf.download(all_tickers, start=start, auto_adjust=True,
                              progress=False, multi_level_index=False)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw
            prices.columns.name = None
            prices.index.name   = "Date"
            return prices.dropna(how="all").ffill()
        except Exception as e:
            raise ConnectionError(f"Error descargando {tickers}: {e}") from e

    @timer_log
    def get_rf(self) -> dict:
        """Tasa libre de riesgo desde ^IRX."""
        try:
            irx = yf.download("^IRX", period="5d", progress=False,
                               auto_adjust=True, multi_level_index=False)
            irx.columns.name = None
            latest = float(irx["Close"].dropna().iloc[-1])
            return {
                "annual" : latest / 100,
                "daily"  : latest / 100 / 252,
                "display": f"{latest:.2f}%",
                "source" : "Yahoo Finance · ^IRX (T-Bill 3M)",
                "date"   : irx.index[-1].strftime("%Y-%m-%d"),
            }
        except Exception:
            return {
                "annual" : 0.0525,
                "daily"  : 0.0525 / 252,
                "display": "5.25%",
                "source" : "Referencia manual (fallback)",
                "date"   : datetime.today().strftime("%Y-%m-%d"),
            }


# ════════════════════════════════════════════════
# TechnicalIndicators — indicadores técnicos
# ════════════════════════════════════════════════

class TechnicalIndicators:
    """Calcula indicadores técnicos sobre series de precios."""

    @staticmethod
    def sma(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(w).mean()

    @staticmethod
    def ema(s: pd.Series, w: int) -> pd.Series:
        return s.ewm(span=w, adjust=False).mean()

    @staticmethod
    def rsi(s: pd.Series, w: int = 14) -> pd.Series:
        d = s.diff()
        g = d.clip(lower=0).rolling(w).mean()
        l = (-d.clip(upper=0)).rolling(w).mean()
        return 100 - 100 / (1 + g / l.replace(0, np.nan))

    @staticmethod
    def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
             ) -> tuple[pd.Series, pd.Series, pd.Series]:
        ml  = TechnicalIndicators.ema(s, fast) - TechnicalIndicators.ema(s, slow)
        sig = TechnicalIndicators.ema(ml, signal)
        return ml, sig, ml - sig

    @staticmethod
    def bollinger(s: pd.Series, w: int = 20, k: float = 2.0
                  ) -> tuple[pd.Series, pd.Series, pd.Series]:
        m   = TechnicalIndicators.sma(s, w)
        std = s.rolling(w).std()
        return m + k * std, m, m - k * std

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
        lo   = low.rolling(k).min()
        hi   = high.rolling(k).max()
        pk   = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
        return pk, pk.rolling(d).mean()

    @timer_log
    def compute_all(self, df: pd.DataFrame) -> dict:
        """Calcula todos los indicadores y retorna dict serializable."""
        c = df["Close"]
        sma20 = self.sma(c, 20)
        sma50 = self.sma(c, 50)
        ema21 = self.ema(c, 21)
        bb_u, _, bb_l = self.bollinger(c)
        rsi14 = self.rsi(c, 14)
        macd_l, macd_s, macd_h = self.macd(c)
        stk, std = self.stochastic(df["High"], df["Low"], c)

        def to_list(s: pd.Series) -> list[Optional[float]]:
            return [None if np.isnan(v) else round(float(v), 4) for v in s]

        return {
            "fechas"      : [d.strftime("%Y-%m-%d") for d in df.index],
            "close"       : [round(float(v), 4) for v in c],
            "sma_20"      : to_list(sma20),
            "sma_50"      : to_list(sma50),
            "ema_21"      : to_list(ema21),
            "bb_upper"    : to_list(bb_u),
            "bb_lower"    : to_list(bb_l),
            "rsi"         : to_list(rsi14),
            "macd"        : to_list(macd_l),
            "macd_signal" : to_list(macd_s),
            "macd_hist"   : to_list(macd_h),
            "stoch_k"     : to_list(stk),
            "stoch_d"     : to_list(std),
        }


# ════════════════════════════════════════════════
# RiskCalculator — VaR y CVaR
# ════════════════════════════════════════════════

class RiskCalculator:
    """Calcula VaR y CVaR con tres métodos."""

    def __init__(self, data_service: DataService) -> None:
        self._ds = data_service

    @timer_log
    def compute_var(
        self,
        tickers: list[str],
        weights: list[float],
        confidence: float = 0.95,
        years: int = 3,
        n_sim: int = 10000,
    ) -> dict:
        prices  = self._ds.get_multi_close(tickers, years)
        prices  = prices[tickers]
        log_ret = np.log(prices / prices.shift(1)).dropna()
        w       = np.array(weights)
        port_r  = log_ret.values @ w

        # Paramétrico
        mu    = port_r.mean()
        sigma = port_r.std()
        var_p95 = float(-stats.norm.ppf(1 - 0.95, mu, sigma))
        var_p99 = float(-stats.norm.ppf(1 - 0.99, mu, sigma))

        # Histórico
        var_h95 = float(-np.percentile(port_r, (1 - 0.95) * 100))
        var_h99 = float(-np.percentile(port_r, (1 - 0.99) * 100))

        # Montecarlo
        np.random.seed(42)
        sim = np.random.normal(mu, sigma, n_sim)
        var_mc95 = float(-np.percentile(sim, (1 - 0.95) * 100))
        var_mc99 = float(-np.percentile(sim, (1 - 0.99) * 100))

        # CVaR (Expected Shortfall)
        cvar_95 = float(-port_r[port_r <= -var_p95].mean()) if (port_r <= -var_p95).any() else var_p95
        cvar_99 = float(-port_r[port_r <= -var_p99].mean()) if (port_r <= -var_p99).any() else var_p99

        return {
            "var_parametrico_95": round(var_p95, 6),
            "var_parametrico_99": round(var_p99, 6),
            "var_historico_95"  : round(var_h95, 6),
            "var_historico_99"  : round(var_h99, 6),
            "var_montecarlo_95" : round(var_mc95, 6),
            "var_montecarlo_99" : round(var_mc99, 6),
            "cvar_95"           : round(cvar_95, 6),
            "cvar_99"           : round(cvar_99, 6),
            "var_anualizado_95" : round(var_p95 * np.sqrt(252), 6),
            "var_anualizado_99" : round(var_p99 * np.sqrt(252), 6),
            "distribucion"      : [round(float(v), 6) for v in port_r.tolist()],
        }


# ════════════════════════════════════════════════
# PortfolioAnalyzer — CAPM y Markowitz
# ════════════════════════════════════════════════

class PortfolioAnalyzer:
    """CAPM, frontera eficiente y métricas de portafolio."""

    def __init__(self, data_service: DataService) -> None:
        self._ds = data_service

    @staticmethod
    def _classify(beta: float) -> str:
        if beta > 1.2: return "Agresivo"
        if beta < 0.8: return "Defensivo"
        return "Neutro"

    @timer_log
    def compute_capm(self, tickers: list[str], benchmark: str = "^GSPC",
                     years: int = 3) -> dict:
        all_t   = list(set(tickers + [benchmark]))
        prices  = self._ds.get_multi_close(all_t, years)
        log_ret = np.log(prices / prices.shift(1)).dropna()
        rf      = self._ds.get_rf()
        rm_d    = log_ret[benchmark].mean()

        activos = []
        for t in tickers:
            r_a  = log_ret[t]
            r_m  = log_ret[benchmark]
            df_r = pd.concat([r_a, r_m], axis=1).dropna()
            slope, intercept, r, _, _ = stats.linregress(df_r.iloc[:, 1], df_r.iloc[:, 0])
            er   = rf["daily"] + slope * (rm_d - rf["daily"])
            vt   = r_a.var()
            vm   = r_m.var()
            vsys = (slope ** 2) * vm
            psys = min(vsys / vt * 100, 100) if vt > 0 else 0
            activos.append({
                "ticker"                    : t,
                "sector"                    : SECTOR_MAP.get(t, "Otro"),
                "beta"                      : round(slope, 4),
                "alpha_anual"               : round(intercept * 252, 4),
                "r_cuadrado"                : round(r ** 2, 4),
                "retorno_esperado"          : round(er * 252, 4),
                "clasificacion"             : self._classify(slope),
                "riesgo_sistematico_pct"    : round(psys, 2),
                "riesgo_idiosincratico_pct" : round(100 - psys, 2),
            })

        return {
            "rf_display" : rf["display"],
            "rf_annual"  : round(rf["annual"], 6),
            "rf_source"  : rf["source"],
            "rf_date"    : rf["date"],
            "benchmark"  : benchmark,
            "rm_annual"  : round(rm_d * 252, 4),
            "activos"    : activos,
        }

    @timer_log
    def compute_frontera(
        self,
        tickers: list[str],
        years: int = 3,
        n_portfolios: int = 10000,
        target_return: Optional[float] = None,
    ) -> dict:
        prices  = self._ds.get_multi_close(tickers, years)
        prices  = prices[tickers]
        log_ret = np.log(prices / prices.shift(1)).dropna()
        rf      = self._ds.get_rf()
        mu      = log_ret.mean().values * 252
        cov     = log_ret.cov().values * 252
        n       = len(tickers)

        np.random.seed(42)
        rets, vols, sharpes, pesos_list = [], [], [], []
        for _ in range(n_portfolios):
            w = np.random.dirichlet(np.ones(n))
            r = float(w @ mu)
            v = float(np.sqrt(w @ cov @ w))
            s = (r - rf["annual"]) / v if v > 0 else 0
            rets.append(round(r, 6))
            vols.append(round(v, 6))
            sharpes.append(round(s, 4))
            pesos_list.append([round(float(x), 4) for x in w])

        arr_r = np.array(rets)
        arr_v = np.array(vols)
        arr_s = np.array(sharpes)

        # Mínima varianza
        idx_mv  = int(np.argmin(arr_v))
        # Máximo Sharpe
        idx_ms  = int(np.argmax(arr_s))

        def make_port(idx: int, nombre: str) -> dict:
            return {
                "nombre"     : nombre,
                "pesos"      : pesos_list[idx],
                "retorno"    : float(arr_r[idx]),
                "volatilidad": float(arr_v[idx]),
                "sharpe"     : float(arr_s[idx]),
            }

        objetivo = None
        if target_return is not None:
            diffs  = np.abs(arr_r - target_return)
            idx_ob = int(np.argmin(diffs))
            objetivo = make_port(idx_ob, f"Objetivo ({target_return:.1%})")

        return {
            "tickers"        : tickers,
            "n_simulaciones" : n_portfolios,
            "retornos"       : rets,
            "volatilidades"  : vols,
            "sharpes"        : sharpes,
            "pesos_simulados": pesos_list,
            "min_varianza"   : make_port(idx_mv, "Mínima Varianza"),
            "max_sharpe"     : make_port(idx_ms, "Máximo Sharpe"),
            "objetivo"       : objetivo,
        }


# ════════════════════════════════════════════════
# AlertasService — señales automáticas
# ════════════════════════════════════════════════

class AlertasService:
    """Genera señales de compra/venta basadas en indicadores técnicos."""

    def __init__(self, data_service: DataService) -> None:
        self._ds = data_service
        self._ti = TechnicalIndicators()

    @timer_log
    def compute_alertas(self, tickers: list[str]) -> list[dict]:
        alertas: list[dict] = []
        for ticker in tickers:
            try:
                df = self._ds.get_ohlcv(ticker, years=1)
                ind = self._ti.compute_all(df)
                alertas.extend(self._evaluar(ticker, ind, df))
            except Exception as e:
                logger.warning(f"Error en alertas para {ticker}: {e}")
        return alertas

    def _evaluar(self, ticker: str, ind: dict, df: pd.DataFrame) -> list[dict]:
        señales: list[dict] = []
        c = ind["close"]

        # RSI
        rsi_vals = [v for v in ind["rsi"] if v is not None]
        if rsi_vals:
            r = rsi_vals[-1]
            if r > 70:
                señales.append({"ticker": ticker, "indicador": "RSI",
                    "señal": "VENTA", "valor": round(r, 2), "umbral": 70.0,
                    "descripcion": f"RSI en {r:.1f} — zona de sobrecompra (>70). Posible corrección."})
            elif r < 30:
                señales.append({"ticker": ticker, "indicador": "RSI",
                    "señal": "COMPRA", "valor": round(r, 2), "umbral": 30.0,
                    "descripcion": f"RSI en {r:.1f} — zona de sobreventa (<30). Posible rebote."})

        # MACD
        macd_vals = [v for v in ind["macd"] if v is not None]
        sig_vals  = [v for v in ind["macd_signal"] if v is not None]
        if len(macd_vals) >= 2 and len(sig_vals) >= 2:
            cross_up   = macd_vals[-2] < sig_vals[-2] and macd_vals[-1] > sig_vals[-1]
            cross_down = macd_vals[-2] > sig_vals[-2] and macd_vals[-1] < sig_vals[-1]
            if cross_up:
                señales.append({"ticker": ticker, "indicador": "MACD",
                    "señal": "COMPRA", "valor": round(macd_vals[-1], 4), "umbral": None,
                    "descripcion": "Cruce alcista del MACD sobre la señal — Golden Cross MACD."})
            elif cross_down:
                señales.append({"ticker": ticker, "indicador": "MACD",
                    "señal": "VENTA", "valor": round(macd_vals[-1], 4), "umbral": None,
                    "descripcion": "Cruce bajista del MACD bajo la señal — Death Cross MACD."})

        # Bandas de Bollinger
        bb_u = [v for v in ind["bb_upper"] if v is not None]
        bb_l = [v for v in ind["bb_lower"] if v is not None]
        if bb_u and bb_l and c:
            precio = c[-1]
            if precio >= bb_u[-1]:
                señales.append({"ticker": ticker, "indicador": "Bollinger",
                    "señal": "VENTA", "valor": round(precio, 2), "umbral": round(bb_u[-1], 2),
                    "descripcion": f"Precio ({precio:.2f}) toca banda superior ({bb_u[-1]:.2f}) — sobrecompra."})
            elif precio <= bb_l[-1]:
                señales.append({"ticker": ticker, "indicador": "Bollinger",
                    "señal": "COMPRA", "valor": round(precio, 2), "umbral": round(bb_l[-1], 2),
                    "descripcion": f"Precio ({precio:.2f}) toca banda inferior ({bb_l[-1]:.2f}) — sobreventa."})

        # SMA Golden/Death Cross
        sma20 = [v for v in ind["sma_20"] if v is not None]
        sma50 = [v for v in ind["sma_50"] if v is not None]
        if len(sma20) >= 2 and len(sma50) >= 2:
            if sma20[-2] < sma50[-2] and sma20[-1] > sma50[-1]:
                señales.append({"ticker": ticker, "indicador": "SMA",
                    "señal": "COMPRA", "valor": round(sma20[-1], 2), "umbral": round(sma50[-1], 2),
                    "descripcion": "Golden Cross: SMA20 cruza SMA50 hacia arriba — tendencia alcista."})
            elif sma20[-2] > sma50[-2] and sma20[-1] < sma50[-1]:
                señales.append({"ticker": ticker, "indicador": "SMA",
                    "señal": "VENTA", "valor": round(sma20[-1], 2), "umbral": round(sma50[-1], 2),
                    "descripcion": "Death Cross: SMA20 cruza SMA50 hacia abajo — tendencia bajista."})

        # Estocástico
        sk = [v for v in ind["stoch_k"] if v is not None]
        sd = [v for v in ind["stoch_d"] if v is not None]
        if len(sk) >= 2 and len(sd) >= 2:
            if sk[-1] < 20 and sk[-1] > sd[-1] and sk[-2] < sd[-2]:
                señales.append({"ticker": ticker, "indicador": "Estocástico",
                    "señal": "COMPRA", "valor": round(sk[-1], 2), "umbral": 20.0,
                    "descripcion": f"%K ({sk[-1]:.1f}) cruza %D en zona sobreventa (<20)."})
            elif sk[-1] > 80 and sk[-1] < sd[-1] and sk[-2] > sd[-2]:
                señales.append({"ticker": ticker, "indicador": "Estocástico",
                    "señal": "VENTA", "valor": round(sk[-1], 2), "umbral": 80.0,
                    "descripcion": f"%K ({sk[-1]:.1f}) cruza %D en zona sobrecompra (>80)."})

        return señales