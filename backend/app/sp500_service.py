"""
backend/app/sp500_service.py
Descarga y cachea la lista del S&P 500 desde Wikipedia vía yfinance/pandas.
Provee validación Pydantic de tickers contra el índice real.
"""
from __future__ import annotations
import logging
import time
import pandas as pd
import yfinance as yf
from functools import lru_cache
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ── Cache en memoria con TTL manual (24h) ─────────────────────
_sp500_cache: dict = {"tickers": [], "timestamp": 0.0}
_CACHE_TTL = 86400  # 24 horas


def get_sp500_tickers() -> list[str]:
    """
    Descarga los tickers del S&P 500 desde Wikipedia.
    Cachea el resultado por 24 horas.
    """
    now = time.time()
    if _sp500_cache["tickers"] and (now - _sp500_cache["timestamp"]) < _CACHE_TTL:
        return _sp500_cache["tickers"]

    try:
        logger.info("Descargando lista S&P 500 desde Wikipedia...")
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            header=0
        )[0]
        # Limpiar tickers (algunos tienen puntos: BRK.B → BRK-B en yfinance)
        tickers = (
            table["Symbol"]
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        _sp500_cache["tickers"] = tickers
        _sp500_cache["timestamp"] = now
        logger.info(f"S&P 500: {len(tickers)} tickers cargados.")
        return tickers
    except Exception as e:
        logger.warning(f"Error descargando S&P 500: {e}. Usando fallback.")
        # Fallback con los más conocidos
        fallback = [
            "AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","BRK-B",
            "JPM","JNJ","V","PG","UNH","HD","MA","XOM","BAC","ABBV",
            "PFE","AVGO","KO","PEP","MRK","CVX","TMO","COST","WMT",
            "ACN","MCD","LIN","DHR","DIS","TXN","ADBE","NFLX","PM",
            "CMCSA","NEE","RTX","INTC","VZ","QCOM","IBM","AMGN","HON",
            "UPS","LOW","SPGI","GS","BLK","CRM","CAT","SBUX","GE",
            "AMD","INTU","ELV","CI","MDT","AXP","SYK","ISRG","GILD",
            "ADI","REGN","VRTX","MMC","ZTS","LRCX","BSX","MO","SO",
            "DUK","CL","CB","AON","ITW","PLD","PSA","AMT","EQIX",
            "JNJ","XOM","JPM","AMZN","AAPL"
        ]
        if not _sp500_cache["tickers"]:
            _sp500_cache["tickers"] = list(set(fallback))
        return _sp500_cache["tickers"]


def get_sp500_info() -> list[dict]:
    """
    Retorna lista con ticker + nombre de empresa del S&P 500.
    """
    try:
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            header=0
        )[0]
        result = []
        for _, row in table.iterrows():
            ticker = str(row["Symbol"]).strip().replace(".", "-")
            name   = str(row.get("Security", row.get("Company", ticker))).strip()
            sector = str(row.get("GICS Sector", "")).strip()
            result.append({
                "ticker": ticker,
                "name":   name,
                "sector": sector,
            })
        return result
    except Exception as e:
        logger.warning(f"Error obteniendo info S&P 500: {e}")
        return [{"ticker": t, "name": t, "sector": ""} for t in get_sp500_tickers()]


def is_sp500_ticker(ticker: str) -> bool:
    """Verifica si un ticker pertenece al S&P 500."""
    valid = get_sp500_tickers()
    return ticker.upper().replace(".", "-") in [v.upper() for v in valid]


def validate_tickers_sp500(tickers: list[str]) -> tuple[bool, list[str]]:
    """
    Valida que todos los tickers estén en el S&P 500.
    Retorna (ok, lista_de_invalidos).
    """
    valid = {v.upper() for v in get_sp500_tickers()}
    invalid = [t for t in tickers if t.upper().replace(".", "-") not in valid]
    return len(invalid) == 0, invalid