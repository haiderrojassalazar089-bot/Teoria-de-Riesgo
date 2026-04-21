"""
frontend/pages/selector.py
Pantalla inicial de selección de activos del S&P 500.
Se muestra antes de cualquier módulo. Guarda en st.session_state.
"""
from __future__ import annotations
import streamlit as st
import requests
import pandas as pd

BACKEND_URL = "http://localhost:8002"

# Paleta
SECTOR_COLORS = {
    "Information Technology": "#8B6914",
    "Financials":             "#1A6B4A",
    "Energy":                 "#1A4F6E",
    "Health Care":            "#3D2F6B",
    "Consumer Discretionary": "#8B2A2A",
    "Communication Services": "#2266A0",
    "Industrials":            "#5A3E7A",
    "Consumer Staples":       "#2A6B4A",
    "Utilities":              "#8B5A14",
    "Real Estate":            "#4A3D8B",
    "Materials":              "#6B2A1A",
}

def get_color(sector: str) -> str:
    for k, v in SECTOR_COLORS.items():
        if k.lower() in sector.lower():
            return v
    return "#8896A8"


@st.cache_data(ttl=86400, show_spinner=False)
def cargar_sp500() -> list[dict]:
    """Descarga la lista del S&P 500 desde el backend (cachea 24h)."""
    try:
        r = requests.get(f"{BACKEND_URL}/sp500/tickers", timeout=30)
        r.raise_for_status()
        return r.json()["tickers"]
    except Exception as e:
        st.error(f"No se pudo cargar la lista del S&P 500: {e}")
        return []


def show():
    # ── CSS adicional para esta página ──
    st.markdown("""
    <style>
    .selector-hero {
        background: linear-gradient(135deg, #1A2035 0%, #2A3550 100%);
        border-radius: 12px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }
    .selector-hero h1 {
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-bottom: 0.4rem !important;
    }
    .selector-hero p {
        color: #8896A8 !important;
        font-size: 0.85rem !important;
    }
    .sector-chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.55rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: 'IBM Plex Mono', monospace;
        margin-right: 4px;
    }
    .ticker-chip {
        background: #F4F6FB;
        border: 1px solid #D8DDE8;
        border-radius: 6px;
        padding: 6px 12px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        margin: 3px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ──
    st.markdown("""
    <div class="selector-hero">
        <h1>Risk<em style="color:#8B6914">Lab</em> · Constructor de Portafolio</h1>
        <p>Selecciona entre 2 y 10 empresas del S&P 500 para comenzar el análisis.
        Todos los módulos M1–M8 y las herramientas avanzadas se adaptarán a tu selección.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Cargando lista del S&P 500..."):
        sp500 = cargar_sp500()

    if not sp500:
        st.error("No se pudo conectar al backend. Verifica que esté corriendo en el puerto 8002.")
        return

    # ── Búsqueda y filtros ──
    col1, col2 = st.columns([2, 1])
    with col1:
        busqueda = st.text_input(
            "🔍 Buscar empresa o ticker",
            placeholder="ej: Apple, MSFT, Amazon, Technology...",
            label_visibility="collapsed",
        )
    with col2:
        sectores_disponibles = sorted(set(i["sector"] for i in sp500 if i["sector"]))
        sector_filtro = st.selectbox(
            "Sector",
            ["Todos los sectores"] + sectores_disponibles,
            label_visibility="collapsed",
        )

    # Filtrar lista
    filtrados = sp500
    if busqueda:
        q = busqueda.lower()
        filtrados = [i for i in filtrados if q in i["ticker"].lower() or q in i["name"].lower()]
    if sector_filtro != "Todos los sectores":
        filtrados = [i for i in filtrados if i["sector"] == sector_filtro]

    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.58rem;
    color:#8896A8;letter-spacing:.12em;text-transform:uppercase;margin-bottom:1rem">
    {len(filtrados)} empresas encontradas · {len(sp500)} en el S&P 500
    </div>
    """, unsafe_allow_html=True)

    # ── Selector multiselect ──
    opciones = {f"{i['ticker']} — {i['name']}": i["ticker"] for i in filtrados}

    # Recuperar selección previa si existe
    seleccion_previa = []
    if "tickers_seleccionados" in st.session_state:
        prev = st.session_state["tickers_seleccionados"]
        seleccion_previa = [k for k, v in opciones.items() if v in prev]

    seleccionados_raw = st.multiselect(
        "Selecciona las empresas",
        options=list(opciones.keys()),
        default=seleccion_previa,
        max_selections=10,
        help="Mínimo 2 empresas, máximo 10. Puedes buscar por nombre o ticker.",
        label_visibility="collapsed",
        placeholder="Haz clic para buscar y seleccionar empresas...",
    )

    tickers_sel = [opciones[s] for s in seleccionados_raw]

    # ── Vista previa de la selección ──
    if tickers_sel:
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
        letter-spacing:.16em;text-transform:uppercase;color:#8896A8;
        margin:1rem 0 .5rem;border-left:2px solid #8B6914;padding-left:8px">
        Tu portafolio — {len(tickers_sel)} activos seleccionados
        </div>
        """, unsafe_allow_html=True)

        info_sel = {i["ticker"]: i for i in sp500 if i["ticker"] in tickers_sel}
        cols = st.columns(min(len(tickers_sel), 5))
        for idx, ticker in enumerate(tickers_sel):
            info = info_sel.get(ticker, {"name": ticker, "sector": ""})
            col = get_color(info.get("sector", ""))
            with cols[idx % 5]:
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                border-top:3px solid {col};border-radius:8px;padding:.9rem;
                text-align:center;margin-bottom:.5rem">
                    <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
                    font-weight:700;color:{col}">{ticker}</div>
                    <div style="font-size:.65rem;color:#4A5568;margin-top:2px;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                    {info.get('name','')[:22]}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.48rem;
                    color:#8896A8;margin-top:4px;letter-spacing:.08em;text-transform:uppercase">
                    {info.get('sector','')[:20]}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Sugerencias rápidas ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
    letter-spacing:.16em;text-transform:uppercase;color:#8896A8;
    margin:1.2rem 0 .5rem;border-left:2px solid #8B6914;padding-left:8px">
    Sugerencias rápidas
    </div>
    """, unsafe_allow_html=True)

    sugerencias = {
        "🏆 Big Tech": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
        "🏦 Financiero": ["JPM", "BAC", "GS", "MS", "V"],
        "⚕️ Salud": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
        "⚡ Energía": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "🛍️ Consumo": ["WMT", "HD", "MCD", "SBUX", "NKE"],
        "📡 Telecomunicaciones": ["T", "VZ", "CMCSA", "NFLX", "DIS"],
    }

    cols_sug = st.columns(len(sugerencias))
    for idx, (nombre, tickers_sug) in enumerate(sugerencias.items()):
        with cols_sug[idx]:
            # Verificar cuáles están en la lista del S&P500 cargada
            disponibles = [t for t in tickers_sug if any(i["ticker"] == t for i in sp500)]
            if st.button(nombre, use_container_width=True, key=f"sug_{idx}"):
                st.session_state["tickers_seleccionados"] = disponibles
                st.session_state["portafolio_confirmado"] = True
                st.rerun()

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Validación y confirmación ──
    if len(tickers_sel) < 2:
        st.info("Selecciona al menos 2 empresas para continuar.")
    else:
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        with col_btn1:
            if st.button(
                f"✓ Analizar portafolio con {len(tickers_sel)} activos",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["tickers_seleccionados"] = tickers_sel
                st.session_state["portafolio_confirmado"] = True
                st.session_state["tickers_info"] = {
                    i["ticker"]: i for i in sp500 if i["ticker"] in tickers_sel
                }
                st.success(f"Portafolio configurado: {', '.join(tickers_sel)}")
                st.rerun()

        with col_btn2:
            if st.button("↺ Limpiar selección", use_container_width=True):
                st.session_state["tickers_seleccionados"] = []
                st.session_state["portafolio_confirmado"] = False
                st.rerun()

    # ── Estado actual ──
    if st.session_state.get("portafolio_confirmado"):
        t_act = st.session_state.get("tickers_seleccionados", [])
        st.markdown(f"""
        <div style="background:rgba(26,107,74,.06);border:1px solid rgba(26,107,74,.2);
        border-radius:8px;padding:.75rem 1rem;margin-top:.5rem;
        font-family:'IBM Plex Mono',monospace;font-size:.68rem;color:#1A6B4A">
        ✓ Portafolio activo: {' · '.join(t_act)} — navega al módulo que desees en el sidebar
        </div>
        """, unsafe_allow_html=True)