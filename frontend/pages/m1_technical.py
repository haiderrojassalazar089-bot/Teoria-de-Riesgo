"""
pages/m1_technical.py — Módulo 1: Análisis Técnico
Streamlit + Plotly | yfinance 1.2.0
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.client import get_indicadores, get_precios, TICKERS, TICKER_COLORS
from utils.theme import plotly_base, COLORS


# ── Indicadores ───────────────────────────────────────────────

def sma(s, w):   return s.rolling(w).mean()
def ema(s, w):   return s.ewm(span=w, adjust=False).mean()

def rsi(s, w=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(w).mean()
    l = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def macd(s, f=12, sl=26, sg=9):
    ml  = ema(s, f) - ema(s, sl)
    sig = ema(ml, sg)
    return ml, sig, ml - sig

def bollinger(s, w=20, k=2):
    m   = sma(s, w)
    std = s.rolling(w).std()
    return m + k*std, m, m - k*std

def stochastic(h, l, c, k=14, d=3):
    lo   = l.rolling(k).min()
    hi   = h.rolling(k).max()
    pk   = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    return pk, pk.rolling(d).mean()


# ── Gráficos ──────────────────────────────────────────────────

def fig_price(df, ticker, s1, s2, e1, chart_type):
    c   = df["Close"]
    col = TICKER_COLORS.get(ticker, COLORS["gold"])
    fig = go.Figure()

    if chart_type == "Velas japonesas":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"].values, high=df["High"].values,
            low=df["Low"].values,   close=c.values,
            name=ticker,
            increasing_line_color=COLORS["emerald"],
            decreasing_line_color=COLORS["rose"],
            increasing_fillcolor=COLORS["emerald"],
            decreasing_fillcolor=COLORS["rose"],
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=c.values, name=ticker,
            line=dict(color=col, width=1.8),
        ))

    fig.add_trace(go.Scatter(x=df.index, y=sma(c, s1).values, name=f"SMA {s1}",
        line=dict(color=COLORS["sky2"], width=1.1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=sma(c, s2).values, name=f"SMA {s2}",
        line=dict(color=COLORS["violet2"], width=1.1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=ema(c, e1).values, name=f"EMA {e1}",
        line=dict(color=COLORS["emerald2"], width=1.3)))

    u, m, l = bollinger(c)
    fig.add_trace(go.Scatter(x=df.index, y=u.values, name="BB+",
        line=dict(color=COLORS["gold"], width=0.8, dash="dot"), opacity=0.5))
    fig.add_trace(go.Scatter(x=df.index, y=l.values, name="BB−",
        line=dict(color=COLORS["gold"], width=0.8, dash="dot"), opacity=0.5,
        fill="tonexty", fillcolor="rgba(168,144,96,0.04)"))

    base = plotly_base(420)
    base["xaxis_rangeslider_visible"] = chart_type == "Velas japonesas"
    fig.update_layout(**base,
        title=dict(text=f"{ticker}  ·  Precio & Medias Móviles",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")))
    return fig


def fig_rsi(df, w):
    r   = rsi(df["Close"], w)
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(139,74,74,0.06)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(61,139,110,0.06)", line_width=0)
    fig.add_hline(y=70, line=dict(color=COLORS["rose"],    width=0.8, dash="dash"))
    fig.add_hline(y=30, line=dict(color=COLORS["emerald"], width=0.8, dash="dash"))
    fig.add_hline(y=50, line=dict(color=COLORS["border2"], width=0.5))
    fig.add_trace(go.Scatter(x=df.index, y=r.values, name=f"RSI({w})",
        line=dict(color=COLORS["gold"], width=1.4),
        hovertemplate="RSI: %{y:.1f}<extra></extra>"))
    base = {**plotly_base(200), "yaxis": dict(range=[0,100], gridcolor=COLORS["border"],
        tickfont=dict(color=COLORS["text3"], size=9))}
    fig.update_layout(**base,
        title=dict(text=f"RSI  ·  {w} períodos",
                   font=dict(size=11, color=COLORS["text2"], family="IBM Plex Mono")))
    return fig


def fig_macd(df):
    ml, sl, hist = macd(df["Close"])
    colors = [COLORS["emerald"] if v >= 0 else COLORS["rose"] for v in hist.fillna(0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=hist.values, name="Histograma",
        marker_color=colors, marker_line_width=0, opacity=0.9))
    fig.add_trace(go.Scatter(x=df.index, y=ml.values, name="MACD",
        line=dict(color=COLORS["gold"], width=1.4)))
    fig.add_trace(go.Scatter(x=df.index, y=sl.values, name="Señal",
        line=dict(color=COLORS["sky2"], width=1.1, dash="dot")))
    fig.add_hline(y=0, line=dict(color=COLORS["border2"], width=0.5))
    fig.update_layout(**plotly_base(200),
        title=dict(text="MACD  ·  (12, 26, 9)",
                   font=dict(size=11, color=COLORS["text2"], family="IBM Plex Mono")))
    return fig


def fig_stoch(df):
    k, d = stochastic(df["High"], df["Low"], df["Close"])
    fig  = go.Figure()
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(139,74,74,0.06)", line_width=0)
    fig.add_hrect(y0=0,  y1=20,  fillcolor="rgba(61,139,110,0.06)", line_width=0)
    fig.add_hline(y=80, line=dict(color=COLORS["rose"],    width=0.8, dash="dash"))
    fig.add_hline(y=20, line=dict(color=COLORS["emerald"], width=0.8, dash="dash"))
    fig.add_trace(go.Scatter(x=df.index, y=k.values, name="%K",
        line=dict(color=COLORS["gold"], width=1.4)))
    fig.add_trace(go.Scatter(x=df.index, y=d.values, name="%D",
        line=dict(color=COLORS["sky2"], width=1.1, dash="dot")))
    base = {**plotly_base(200), "yaxis": dict(range=[0,100], gridcolor=COLORS["border"],
        tickfont=dict(color=COLORS["text3"], size=9))}
    fig.update_layout(**base,
        title=dict(text="Oscilador Estocástico  ·  %K / %D",
                   font=dict(size=11, color=COLORS["text2"], family="IBM Plex Mono")))
    return fig


# ── Layout ────────────────────────────────────────────────────

def show():
    # Header
    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:4px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">
                Módulo 01
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.6rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                Análisis Técnico
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                    color:#8896A8;letter-spacing:0.08em;">
            Indicadores técnicos interactivos  ·  Datos en tiempo real vía Yahoo Finance
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Controles
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        ticker = st.selectbox("Activo", TICKERS, index=0)
    with col2:
        chart_type = st.selectbox("Tipo de gráfico", ["Línea", "Velas japonesas"])
    with col3:
        years = st.selectbox("Período", ["1 año", "2 años", "3 años"], index=1)

    years_map = {"1 año": 1, "2 años": 2, "3 años": 3}

    col4, col5, col6, col7 = st.columns(4)
    with col4:
        s1 = st.slider("SMA 1", 5, 200, 20, 5)
    with col5:
        s2 = st.slider("SMA 2", 5, 200, 50, 5)
    with col6:
        e1 = st.slider("EMA", 5, 100, 21, 5)
    with col7:
        rsi_w = st.slider("RSI", 7, 30, 14, 1)

    st.markdown("<div style='margin:0.5rem 0;'></div>", unsafe_allow_html=True)

    # Cargar datos
    with st.spinner(f"Cargando {ticker}..."):
        data = get_precios(ticker, years=years_map[years])
        import pandas as pd
        df = pd.DataFrame(data["precios"])
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.set_index("fecha")
        df.columns = [c.capitalize() for c in df.columns]
        df.index.name = "Date" 

    if df.empty:
        st.error("No se pudieron cargar los datos. Intenta nuevamente.")
        return

    # Gráfico principal
    st.plotly_chart(fig_price(df, ticker, s1, s2, e1, chart_type),
                    use_container_width=True)

    with st.expander("SMA / EMA  ·  Interpretación"):
        st.markdown("""
        La **SMA** (Media Móvil Simple) promedia los últimos N precios de cierre con igual peso.
        La **EMA** (Media Móvil Exponencial) da mayor ponderación a los datos recientes,
        reaccionando más rápidamente a cambios de tendencia.

        **Golden Cross:** cruce de la SMA corta sobre la SMA larga → señal alcista.
        **Death Cross:** cruce inverso → señal bajista.

        Las **Bandas de Bollinger** se construyen como SMA(20) ± 2σ. El estrechamiento
        de las bandas (*squeeze*) anticipa un movimiento brusco inminente.
        """)

    # RSI y MACD
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(fig_rsi(df, rsi_w), use_container_width=True)
        with st.expander("RSI  ·  Interpretación"):
            st.markdown("""
            Oscilador entre 0 y 100. **RSI > 70** → zona de sobrecompra (posible corrección).
            **RSI < 30** → zona de sobreventa (posible rebote). Período estándar: 14 días.
            """)
    with col_r:
        st.plotly_chart(fig_macd(df), use_container_width=True)
        with st.expander("MACD  ·  Interpretación"):
            st.markdown("""
            Diferencia entre EMA(12) y EMA(26). La **línea de señal** es EMA(9) del MACD.
            Cruce del MACD sobre la señal → entrada alcista.
            El histograma muestra la distancia entre ambas líneas.
            """)

    # Estocástico
    st.plotly_chart(fig_stoch(df), use_container_width=True)
    with st.expander("Oscilador Estocástico  ·  Interpretación"):
        st.markdown("""
        **%K** mide la posición del precio de cierre dentro del rango High-Low de K períodos.
        **%D** es la media móvil de %K. Valores > 80 → sobrecompra. < 20 → sobreventa.
        El cruce de %K sobre %D en zonas extremas genera señales de entrada.
        """)