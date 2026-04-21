"""
frontend/utils/dynamic_tickers.py
Utilidades para que los módulos M1–M8 usen tickers dinámicos
desde st.session_state en lugar de la lista fija.
"""
import streamlit as st
import colorsys

# Paleta de colores extendida para hasta 10 activos
_PALETTE = [
    "#8B6914", "#1A6B4A", "#1A4F6E", "#3D2F6B", "#8B2A2A",
    "#2266A0", "#5A3E7A", "#2A6B4A", "#8B5A14", "#6B2A1A",
]


def get_tickers() -> list[str]:
    """Retorna los tickers activos del portafolio del usuario."""
    tickers = st.session_state.get("tickers_seleccionados", [])
    if not tickers:
        # Fallback a los 5 originales si no hay selección
        return ["AAPL", "JPM", "XOM", "JNJ", "AMZN"]
    return tickers


def get_ticker_colors() -> dict[str, str]:
    """Genera un mapa ticker → color para los tickers activos."""
    tickers = get_tickers()
    fixed = {
        "AAPL": "#8B6914", "JPM": "#1A6B4A", "XOM": "#1A4F6E",
        "JNJ": "#3D2F6B",  "AMZN": "#8B2A2A",
    }
    result = {}
    color_idx = 0
    for t in tickers:
        if t in fixed:
            result[t] = fixed[t]
        else:
            result[t] = _PALETTE[color_idx % len(_PALETTE)]
            color_idx += 1
    return result


def get_ticker_info() -> dict[str, dict]:
    """Retorna info de cada ticker desde session_state (nombre, sector)."""
    return st.session_state.get("tickers_info", {})


def get_sector(ticker: str) -> str:
    info = get_ticker_info()
    if ticker in info:
        return info[ticker].get("sector", "S&P 500")
    fallback = {
        "AAPL": "Tecnología", "JPM": "Financiero", "XOM": "Energía",
        "JNJ": "Salud", "AMZN": "Consumo discrecional",
    }
    return fallback.get(ticker, "S&P 500")


def get_nombre(ticker: str) -> str:
    info = get_ticker_info()
    if ticker in info:
        return info[ticker].get("name", ticker)
    fallback = {
        "AAPL": "Apple Inc.", "JPM": "JPMorgan Chase", "XOM": "ExxonMobil",
        "JNJ": "Johnson & Johnson", "AMZN": "Amazon.com Inc.",
    }
    return fallback.get(ticker, ticker)


def render_portafolio_badge():
    """Muestra un banner con el portafolio activo en cualquier módulo."""
    tickers = get_tickers()
    colors  = get_ticker_colors()
    chips   = " ".join([
        f'<span style="background:{colors.get(t,"#8B6914")}22;color:{colors.get(t,"#8B6914")};'
        f'border:1px solid {colors.get(t,"#8B6914")}44;border-radius:4px;'
        f'padding:1px 7px;font-family:IBM Plex Mono,monospace;font-size:.6rem;'
        f'font-weight:600">{t}</span>'
        for t in tickers
    ])
    st.markdown(f"""
    <div style="background:#F4F6FB;border:1px solid #D8DDE8;border-radius:7px;
    padding:.6rem 1rem;margin-bottom:1.2rem;display:flex;align-items:center;
    gap:.6rem;flex-wrap:wrap">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:.5rem;
        letter-spacing:.16em;text-transform:uppercase;color:#8896A8">Portafolio:</span>
        {chips}
    </div>
    """, unsafe_allow_html=True)


def ticker_selector_widget(key: str = "sel_ticker", label: str = "Activo") -> str:
    """
    Widget de selección de ticker del portafolio activo.
    Úsalo en los módulos M1–M8 para reemplazar el selectbox fijo.
    """
    tickers = get_tickers()
    colors  = get_ticker_colors()
    return st.selectbox(label, tickers, key=key)