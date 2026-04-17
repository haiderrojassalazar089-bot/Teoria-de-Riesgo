"""
frontend/data/client.py
Cliente HTTP para consumir el backend FastAPI.
Maneja errores, reintentos y caché con st.cache_data.
"""

from __future__ import annotations
import logging
import requests
import streamlit as st
from typing import Optional

logger = logging.getLogger(__name__)

# ── URL del backend ───────────────────────────────────────────
BACKEND_URL = "http://localhost:8002"


def _get(endpoint: str, params: dict = None) -> dict:
    """GET request al backend con manejo de errores."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        r   = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar al backend. Asegúrate de que está corriendo en el puerto 8000.")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("⏱ El backend tardó demasiado en responder.")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Error del backend: {e.response.json().get('detail', str(e))}")
        st.stop()


def _post(endpoint: str, body: dict) -> dict:
    """POST request al backend con manejo de errores."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        r   = requests.post(url, json=body, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar al backend.")
        st.stop()
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        st.error(f"❌ Error del backend: {detail}")
        st.stop()


# ── Endpoints ─────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def get_activos() -> dict:
    return _get("/activos")


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
def get_capm(years: int = 3) -> dict:
    return _get("/capm", params={"years": years})


@st.cache_data(ttl=1800, show_spinner=False)
def get_macro() -> dict:
    return _get("/macro")


@st.cache_data(ttl=300, show_spinner=False)
def get_alertas() -> dict:
    return _get("/alertas")


def post_var(
    tickers: list[str],
    weights: list[float],
    confidence: float = 0.95,
    years: int = 3,
) -> dict:
    return _post("/var", {
        "tickers"   : tickers,
        "weights"   : weights,
        "confidence": confidence,
        "years"     : years,
    })


def post_frontera(
    tickers: list[str],
    years: int = 3,
    n_portfolios: int = 10000,
    target_return: Optional[float] = None,
) -> dict:
    body = {
        "tickers"     : tickers,
        "years"       : years,
        "n_portfolios": n_portfolios,
    }
    if target_return is not None:
        body["target_return"] = target_return
    return _post("/frontera-eficiente", body)


# ── Constantes del portafolio ──────────────────────────────────
TICKERS = ["AAPL", "JPM", "XOM", "JNJ", "AMZN"]
BENCHMARK = "^GSPC"
SECTOR_MAP = {
    "AAPL": "Tecnología",
    "JPM" : "Financiero",
    "XOM" : "Energía",
    "JNJ" : "Salud",
    "AMZN": "Consumo discrecional",
}
TICKER_COLORS = {
    "AAPL": "#8B6914",
    "JPM" : "#1A6B4A",
    "XOM" : "#1A4F6E",
    "JNJ" : "#3D2F6B",
    "AMZN": "#8B2A2A",
}