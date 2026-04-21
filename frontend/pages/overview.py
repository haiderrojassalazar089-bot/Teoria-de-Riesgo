"""
frontend/pages/overview.py — Vista General
Usa tickers dinámicos desde session_state.
Se actualiza automáticamente al cambiar el portafolio.
"""
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.client import get_activos, get_precios, get_macro
from utils.theme import plotly_base, COLORS
from utils.dynamic_tickers import (
    get_tickers, get_ticker_colors, get_nombre, get_sector,
    render_portafolio_badge,
)
 
 
def show():
    # ── Tickers dinámicos ──────────────────────────────────
    TICKERS = get_tickers()
    TICKER_COLORS = get_ticker_colors()
    tickers_str = ",".join(TICKERS)
 
    render_portafolio_badge()
 
    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">
                ◈ Panel
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                Vista General
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
                    color:#8896A8;letter-spacing:0.08em;">
            Portafolio equi-ponderado · S&P 500 benchmark · Datos en tiempo real
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Cargar datos desde el backend con tickers dinámicos ──
    with st.spinner("Consultando backend..."):
        activos_data = get_activos(tickers=tickers_str)
        macro_data   = get_macro()
 
    rf   = macro_data["tasa_libre_riesgo"]
    bm_r = macro_data["benchmark_retorno"]
    bm_v = macro_data["benchmark_vol"]
 
    with st.spinner("Calculando métricas del portafolio..."):
        all_prices = {}
        for t in TICKERS:
            px_data = get_precios(t, years=3)
            fechas  = [p["fecha"] for p in px_data["precios"]]
            closes  = [p["close"] for p in px_data["precios"]]
            all_prices[t] = pd.Series(closes, index=pd.to_datetime(fechas))
 
    prices_df = pd.DataFrame(all_prices).dropna()
    log_ret   = np.log(prices_df / prices_df.shift(1)).dropna()
    port_r    = log_ret.mean(axis=1)
    cum_ret   = (1 + port_r).cumprod().iloc[-1] - 1
    ann_vol   = port_r.std() * np.sqrt(252)
    rf_ann    = macro_data["tasa_libre_riesgo"]["valor"]
    sharpe    = (port_r.mean() * 252 - rf_ann) / ann_vol
    drawdown  = ((1+port_r).cumprod() / (1+port_r).cumprod().cummax() - 1).min()
 
    # ── KPIs ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Retorno Acumulado",    f"{cum_ret:+.1%}",  "Portafolio equi-ponderado")
    c2.metric("Volatilidad Anual",    f"{ann_vol:.1%}",   "Desv. std × √252")
    c3.metric("Ratio de Sharpe",      f"{sharpe:.2f}",    f"Rf = {rf['display']}")
    c4.metric("Máx. Drawdown",        f"{drawdown:.1%}",  "Peor caída desde máximo")
    c5.metric("Tasa Libre de Riesgo", rf["display"],      rf["fuente"])
 
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
 
    # ── Gráficos ──
    col_l, col_r = st.columns([2, 1])
 
    with col_l:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                    letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                    border-left:2px solid #8B6914;padding-left:8px;">
            Rendimiento normalizado — Base 100
        </div>
        """, unsafe_allow_html=True)
        norm = prices_df / prices_df.iloc[0] * 100
        fig  = go.Figure()
        for t in TICKERS:
            col = TICKER_COLORS.get(t, COLORS["gold"])
            nombre = get_nombre(t)
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm[t].values,
                name=f"{t} — {nombre}",
                line=dict(color=col, width=1.5),
            ))
        fig.update_layout(**plotly_base(320))
        st.plotly_chart(fig, use_container_width=True)
 
    with col_r:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                    letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                    border-left:2px solid #8B6914;padding-left:8px;">
            Correlación de rendimientos
        </div>
        """, unsafe_allow_html=True)
        corr = log_ret.corr()
        pb   = plotly_base(320)
        pb["xaxis"]["type"] = "-"
        pb["yaxis"]["type"] = "-"
        fig2 = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale=[[0.0,"#EEF1F6"],[0.5,"#C4CBD8"],[1.0,"#8B6914"]],
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=10, color="#1A2035"), showscale=False,
        ))
        fig2.update_layout(**pb)
        st.plotly_chart(fig2, use_container_width=True)
 
    # ── Tabla de activos con descripción ──
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    last_date = prices_df.index[-1].strftime("%d %b %Y")
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                border-left:2px solid #8B6914;padding-left:8px;">
        Resumen de activos · {last_date}
    </div>
    """, unsafe_allow_html=True)
 
    rows = []
    tickers_info = st.session_state.get("tickers_info", {})
    for a in activos_data["activos"]:
        t     = a["ticker"]
        ret   = log_ret[t].dropna() if t in log_ret.columns else pd.Series([0])
        ann_v = ret.std() * np.sqrt(252)
        px_s  = prices_df[t].dropna()
        ytd   = px_s.iloc[-1] / px_s[px_s.index.year == px_s.index[-1].year].iloc[0] - 1
        info  = tickers_info.get(t, {})
        nombre = info.get("name", a.get("nombre", t))
        sector = info.get("sector", a.get("sector", "—"))
        rows.append({
            "Ticker"    : t,
            "Empresa"   : nombre,
            "Sector"    : sector,
            "Último"    : f"${a['ultimo']:.2f}",
            "Δ Hoy"     : f"{a['cambio_hoy']:+.2%}",
            "YTD"       : f"{ytd:+.1%}",
            "Vol. Anual": f"{ann_v:.1%}",
        })
 
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
 
    # ── Descripciones de empresas ──
    if tickers_info:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                    letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.8rem;
                    border-left:2px solid #8B6914;padding-left:8px;">
            Empresas del portafolio
        </div>
        """, unsafe_allow_html=True)
        cols_desc = st.columns(min(len(TICKERS), 5))
        for i, t in enumerate(TICKERS):
            info = tickers_info.get(t, {})
            col  = TICKER_COLORS.get(t, COLORS["gold"])
            sector = info.get("sector", get_sector(t))
            nombre = info.get("name", get_nombre(t))
            with cols_desc[i % 5]:
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                border-top:3px solid {col};border-radius:8px;
                padding:.9rem;margin-bottom:.5rem">
                    <div style="font-family:'Playfair Display',serif;font-size:1rem;
                    font-weight:700;color:{col}">{t}</div>
                    <div style="font-size:.72rem;color:#1A2035;margin-top:3px;
                    font-weight:500">{nombre}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.52rem;
                    color:#8896A8;margin-top:4px;letter-spacing:.06em;
                    text-transform:uppercase">{sector}</div>
                </div>
                """, unsafe_allow_html=True)
 
    # ── Macro ──
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                border-left:2px solid #8B6914;padding-left:8px;">
        Indicadores macroeconómicos · Fuente: Yahoo Finance API
    </div>
    """, unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric(rf["nombre"],   rf["display"],   rf["fuente"])
    m2.metric(bm_r["nombre"], bm_r["display"], "S&P 500 anualizado")
    m3.metric(bm_v["nombre"], bm_v["display"], "Volatilidad anualizada")
 
    with st.expander("Metodología y notas técnicas"):
        st.markdown(f"""
        **Arquitectura:** los datos provienen del backend FastAPI (`/activos`, `/precios`, `/macro`).
        El frontend Streamlit consume los endpoints y renderiza las visualizaciones.
 
        **Portafolio activo:** {', '.join(TICKERS)} — equi-ponderado ({round(100/len(TICKERS),1)}% cada uno).
        Retorno y volatilidad calculados sobre log-rendimientos diarios.
 
        **Tasa libre de riesgo:** obtenida automáticamente desde `^IRX` (T-Bill 3M, Yahoo Finance).
        """)