import os
"""
frontend/data/client.py
Cliente HTTP para consumir el backend FastAPI.
Usa tickers dinámicos desde st.session_state.
"""
from __future__ import annotations
import logging
import requests
import streamlit as st
from typing import Optional

logger = logging.getLogger(__name__)
BACKEND_URL = os.environ.get("BACKEND_URL", "https://risklab-backend.onrender.com")


def _tickers_activos() -> list[str]:
    """Retorna los tickers seleccionados por el usuario desde session_state."""
    return st.session_state.get("tickers_seleccionados", ["AAPL", "JPM", "XOM", "JNJ", "AMZN"])


def _tickers_str() -> str:
    return ",".join(_tickers_activos())


def _get(endpoint: str, params: dict = None) -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar al backend. Asegúrate de que está corriendo en el puerto 8002.")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("⏱ El backend tardó demasiado en responder.")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Error del backend: {e.response.json().get('detail', str(e))}")
        st.stop()


def _post(endpoint: str, body: dict) -> dict:
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=body, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar al backend.")
        st.stop()
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        st.error(f"❌ Error del backend: {detail}")
        st.stop()


# ── Endpoints dinámicos ────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_list(sector: str = None, search: str = None) -> dict:
    params = {}
    if sector: params["sector"] = sector
    if search: params["search"] = search
    return _get("/sp500/tickers", params=params)


@st.cache_data(ttl=1800, show_spinner=False)
def get_activos(tickers: str = None) -> dict:
    return _get("/activos", params={"tickers": tickers or _tickers_str()})


@st.cache_data(ttl=1800, show_spinner=False)
def get_precios(ticker: str, years: int = 3) -> dict:
    return _get(f"/precios/{ticker}", params={"years": years})


@st.cache_data(ttl=1800, show_spinner=False)
def get_rendimientos(ticker: str, years: int = 3) -> dict:
    return _get(f"/rendimientos/{ticker}", params={"years": years})


@st.cache_data(ttl=1800, show_spinner=False)
def get_indicadores(ticker: str, years: int = 2) -> dict:
    return _get(f"/indicadores/{ticker}", params={"years": years})


@st.cache_data(ttl=1800, show_spinner=False)
def get_capm(tickers: str = None, years: int = 3) -> dict:
    return _get("/capm", params={"tickers": tickers or _tickers_str(), "years": years})


@st.cache_data(ttl=1800, show_spinner=False)
def get_macro() -> dict:
    return _get("/macro")


@st.cache_data(ttl=300, show_spinner=False)
def get_alertas(tickers: str = None) -> dict:
    return _get("/alertas", params={"tickers": tickers or _tickers_str()})


def post_var(
    tickers: list[str],
    weights: list[float],
    confidence: float = 0.95,
    years: int = 3,
) -> dict:
    return _post("/var", {
        "tickers": tickers,
        "weights": weights,
        "confidence": confidence,
        "years": years,
    })


def post_frontera(
    tickers: list[str],
    years: int = 3,
    n_portfolios: int = 10000,
    target_return: Optional[float] = None,
) -> dict:
    body = {"tickers": tickers, "years": years, "n_portfolios": n_portfolios}
    if target_return is not None:
        body["target_return"] = target_return
    return _post("/frontera-eficiente", body)


def post_montecarlo(
    tickers: list[str],
    weights: list[float],
    horizon_days: int = 252,
    n_simulations: int = 500,
    years_history: int = 3,
) -> dict:
    return _post("/montecarlo", {
        "tickers": tickers,
        "weights": weights,
        "horizon_days": horizon_days,
        "n_simulations": n_simulations,
        "years_history": years_history,
    })


def post_duelo(
    tickers_a: list[str], weights_a: list[float],
    tickers_b: list[str], weights_b: list[float],
    years: int = 3,
) -> dict:
    return _post("/duelo", {
        "portafolio_a": {"tickers": tickers_a, "weights": weights_a, "confidence": 0.95, "years": years},
        "portafolio_b": {"tickers": tickers_b, "weights": weights_b, "confidence": 0.95, "years": years},
        "years": years,
    })


def post_maquina_tiempo(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> dict:
    return _post("/maquina-tiempo", {
        "tickers": tickers,
        "years": 10,
        "start_date": start_date,
        "end_date": end_date,
    })


# ── Compatibilidad con módulos existentes ──────────────────
# Los módulos M1–M8 usan TICKERS como constante.
# Ahora es una propiedad dinámica.
def get_tickers_activos() -> list[str]:
    return _tickers_activos()


# Para retrocompatibilidad
TICKERS = ["AAPL", "JPM", "XOM", "JNJ", "AMZN"]
BENCHMARK = "^GSPC"
SECTOR_MAP = {
    "AAPL": "Tecnología", "JPM": "Financiero", "XOM": "Energía",
    "JNJ": "Salud", "AMZN": "Consumo discrecional",
}
TICKER_COLORS = {
    "AAPL": "#8B6914", "JPM": "#1A6B4A", "XOM": "#1A4F6E",
    "JNJ": "#3D2F6B",  "AMZN": "#8B2A2A",
}