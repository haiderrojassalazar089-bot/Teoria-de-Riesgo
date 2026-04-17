"""
data/loader.py
yfinance 1.2.0 + Python 3.12.1 + Streamlit cache
"""
 
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
 
logger = logging.getLogger(__name__)
 
TICKERS   = ["AAPL", "JPM", "XOM", "JNJ", "AMZN"]
BENCHMARK = "^GSPC"
SECTOR_MAP = {
    "AAPL": "Tecnología",
    "JPM" : "Financiero",
    "XOM" : "Energía",
    "JNJ" : "Salud",
    "AMZN": "Consumo discrecional",
}
TICKER_COLORS = {
    "AAPL": "#A89060",
    "JPM" : "#3D8B6E",
    "XOM" : "#3A6B8A",
    "JNJ" : "#5A4E7A",
    "AMZN": "#8B4A4A",
}
 
 
def _dl(tickers, start):
    df = yf.download(tickers, start=start, auto_adjust=True,
                     progress=False, multi_level_index=False)
    df.columns.name = None
    df.index.name   = "Date"
    return df
 
 
@st.cache_data(ttl=1800, show_spinner=False)
def get_prices(years=3):
    tickers = TICKERS + [BENCHMARK]
    start   = (datetime.today() - timedelta(days=365*years)).strftime("%Y-%m-%d")
    for attempt in range(1, 4):
        try:
            raw = _dl(tickers, start)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw
            return prices.dropna(how="all").ffill()
        except Exception as e:
            logger.warning(f"Intento {attempt}/3: {e}")
            time.sleep(2 ** attempt)
    raise ConnectionError("No se pudieron descargar precios.")
 
 
@st.cache_data(ttl=1800, show_spinner=False)
def get_ohlcv(ticker, years=3):
    start = (datetime.today() - timedelta(days=365*years)).strftime("%Y-%m-%d")
    df    = _dl(ticker, start)
    return df.dropna() if not df.empty else pd.DataFrame()
 
 
def get_returns(prices=None, log=True):
    if prices is None:
        prices = get_prices()
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()
 
 
@st.cache_data(ttl=1800, show_spinner=False)
def get_risk_free_rate():
    try:
        irx = yf.download("^IRX", period="5d", progress=False,
                           auto_adjust=True, multi_level_index=False)
        irx.columns.name = None
        latest = float(irx["Close"].dropna().iloc[-1])
        return {
            "annual" : latest / 100,
            "daily"  : latest / 100 / 252,
            "display": f"{latest:.2f}%",
            "source" : "Yahoo Finance · ^IRX",
            "date"   : irx.index[-1].strftime("%Y-%m-%d"),
        }
    except Exception:
        return {
            "annual" : 0.0525,
            "daily"  : 0.0525 / 252,
            "display": "5.25%",
            "source" : "Referencia manual",
            "date"   : datetime.today().strftime("%Y-%m-%d"),
        }