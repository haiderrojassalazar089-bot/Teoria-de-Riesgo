"""
pages/overview.py — Vista General
"""
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
 
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loader import get_prices, get_returns, get_risk_free_rate, TICKERS, BENCHMARK, TICKER_COLORS, SECTOR_MAP
from utils.theme import plotly_base, COLORS
 
 
def show():
    # Header
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
            Portafolio  AAPL · JPM · XOM · JNJ · AMZN  ·  S&P 500 benchmark
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    with st.spinner("Cargando datos del portafolio..."):
        prices  = get_prices(years=3)
        returns = get_returns(prices[TICKERS])
        rf      = get_risk_free_rate()
 
    port_ret = returns.mean(axis=1)
    cum_ret  = (1 + port_ret).cumprod().iloc[-1] - 1
    ann_vol  = port_ret.std() * np.sqrt(252)
    sharpe   = (port_ret.mean() * 252 - rf["annual"]) / ann_vol
    drawdown = ((1+port_ret).cumprod() / (1+port_ret).cumprod().cummax() - 1).min()
    last_date = prices.index[-1].strftime("%d %b %Y")
 
    # ── KPIs con st.metric ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Retorno Acumulado", f"{cum_ret:+.1%}", "Portafolio equi-ponderado")
    with c2:
        st.metric("Volatilidad Anual", f"{ann_vol:.1%}", "Desv. std × √252")
    with c3:
        st.metric("Ratio de Sharpe", f"{sharpe:.2f}", f"Rf = {rf['display']}")
    with c4:
        st.metric("Máx. Drawdown", f"{drawdown:.1%}", "Peor caída desde máximo")
    with c5:
        st.metric("Tasa Libre de Riesgo", rf["display"], rf["source"])
 
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
        norm = prices[TICKERS] / prices[TICKERS].iloc[0] * 100
        fig  = go.Figure()
        for ticker in TICKERS:
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm[ticker].values, name=ticker,
                line=dict(color=TICKER_COLORS[ticker], width=1.5),
                hovertemplate=f"<b>{ticker}</b>: %{{y:.1f}}<extra></extra>",
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
        corr = returns[TICKERS].corr()
        fig2 = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0.0,"#FFFFFF"],[0.5,"#2A3A4A"],[1.0,"#8B6914"]],
            zmin=-1, zmax=1,
            text=np.round(corr.values,2),
            texttemplate="%{text}",
            textfont=dict(size=10, color="#1A2035"),
            showscale=False,
        ))
        pb = plotly_base(320)
        pb["xaxis"] = dict(gridcolor="#E6EAF2", showline=False,
                           tickfont=dict(color="#8896A8", size=9))
        pb["yaxis"] = dict(gridcolor="#E6EAF2", showline=False,
                           tickfont=dict(color="#8896A8", size=9))
        fig2.update_layout(**pb)
        st.plotly_chart(fig2, use_container_width=True)
 
    # ── Tabla ──
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                border-left:2px solid #8B6914;padding-left:8px;">
        Resumen de activos · {last_date}
    </div>
    """, unsafe_allow_html=True)
 
    rows = []
    for ticker in TICKERS:
        px   = prices[ticker].dropna()
        ret  = returns[ticker].dropna()
        ytd  = px.iloc[-1] / px[px.index.year == px.index[-1].year].iloc[0] - 1
        ann_v = ret.std() * np.sqrt(252)
        rows.append({
            "Ticker"    : ticker,
            "Sector"    : SECTOR_MAP[ticker],
            "Último"    : f"${px.iloc[-1]:.2f}",
            "Δ Hoy"     : f"{ret.iloc[-1]:+.2%}",
            "YTD"       : f"{ytd:+.1%}",
            "Vol. Anual": f"{ann_v:.1%}",
        })
 
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
 
    with st.expander("Metodología y notas técnicas"):
        st.markdown("""
        **Portafolio de referencia:** cinco acciones del S&P 500 de sectores distintos,
        equi-ponderadas (20% cada una). Datos descargados dinámicamente desde Yahoo Finance,
        horizonte de 3 años.
 
        **Métricas:** retorno y volatilidad calculados sobre log-rendimientos diarios.
        El Ratio de Sharpe usa `^IRX` (T-Bill 3M) como tasa libre de riesgo actualizada.
 
        **Correlaciones:** sobre log-rendimientos diarios del período completo.
        Valores cercanos a 0 benefician la diversificación del portafolio.
        """)