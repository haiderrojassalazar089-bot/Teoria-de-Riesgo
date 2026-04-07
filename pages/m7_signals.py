"""
pages/m7_signals.py — Módulo 7: Señales & Alertas
Streamlit + Plotly | yfinance 1.2.0
Panel de señales técnicas tipo semáforo con umbrales configurables
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loader import get_ohlcv, TICKERS, TICKER_COLORS, SECTOR_MAP
from utils.theme import plotly_base, COLORS


# ── Indicadores ───────────────────────────────────────────────

def sma(s, w):  return s.rolling(w).mean()
def ema(s, w):  return s.ewm(span=w, adjust=False).mean()

def rsi(s, w=14):
    d  = s.diff()
    g  = d.clip(lower=0).rolling(w).mean()
    l  = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def macd(s, f=12, sl=26, sg=9):
    ml  = ema(s, f) - ema(s, sl)
    sig = ema(ml, sg)
    return ml, sig, ml - sig

def bollinger(s, w=20, k=2):
    m   = sma(s, w)
    std = s.rolling(w).std()
    return m + k * std, m, m - k * std

def stochastic(h, l, c, k=14, d=3):
    lo = l.rolling(k).min()
    hi = h.rolling(k).max()
    pk = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    return pk, pk.rolling(d).mean()


# ── Señales ───────────────────────────────────────────────────

def eval_macd(df):
    ml, sl, _ = macd(df["Close"])
    cross_up   = (ml.iloc[-1] > sl.iloc[-1]) and (ml.iloc[-2] <= sl.iloc[-2])
    cross_down = (ml.iloc[-1] < sl.iloc[-1]) and (ml.iloc[-2] >= sl.iloc[-2])
    above      = ml.iloc[-1] > sl.iloc[-1]
    if cross_up:
        return "COMPRA", "Cruce alcista del MACD sobre señal"
    if cross_down:
        return "VENTA", "Cruce bajista del MACD bajo señal"
    if above:
        return "NEUTRAL+", "MACD sobre señal — tendencia alcista sin cruce reciente"
    return "NEUTRAL-", "MACD bajo señal — tendencia bajista sin cruce reciente"

def eval_rsi(df, rsi_w, sob_compra, sob_venta):
    r = rsi(df["Close"], rsi_w).iloc[-1]
    if r >= sob_compra:
        return "VENTA", f"RSI = {r:.1f} → sobrecompra (> {sob_compra})"
    if r <= sob_venta:
        return "COMPRA", f"RSI = {r:.1f} → sobreventa (< {sob_venta})"
    return "NEUTRAL", f"RSI = {r:.1f} — zona neutral ({sob_venta}–{sob_compra})"

def eval_bollinger(df, bb_w, bb_k):
    c = df["Close"]
    u, _, l = bollinger(c, bb_w, bb_k)
    p = c.iloc[-1]
    if p >= u.iloc[-1]:
        return "VENTA", f"Precio toca banda superior BB ({p:.2f} ≥ {u.iloc[-1]:.2f})"
    if p <= l.iloc[-1]:
        return "COMPRA", f"Precio toca banda inferior BB ({p:.2f} ≤ {l.iloc[-1]:.2f})"
    return "NEUTRAL", f"Precio dentro de bandas BB ({l.iloc[-1]:.2f} – {u.iloc[-1]:.2f})"

def eval_golden_cross(df, s1, s2):
    c   = df["Close"]
    f   = sma(c, s1)
    sl  = sma(c, s2)
    gc  = (f.iloc[-1] > sl.iloc[-1]) and (f.iloc[-2] <= sl.iloc[-2])
    dc  = (f.iloc[-1] < sl.iloc[-1]) and (f.iloc[-2] >= sl.iloc[-2])
    above = f.iloc[-1] > sl.iloc[-1]
    if gc:
        return "COMPRA", f"Golden Cross: SMA({s1}) cruzó SMA({s2}) al alza"
    if dc:
        return "VENTA",  f"Death Cross: SMA({s1}) cruzó SMA({s2}) a la baja"
    if above:
        return "NEUTRAL+", f"SMA({s1}) > SMA({s2}) — tendencia alcista vigente"
    return "NEUTRAL-", f"SMA({s1}) < SMA({s2}) — tendencia bajista vigente"

def eval_stoch(df, k_p, d_p, zona_alta, zona_baja):
    h, l, c = df["High"], df["Low"], df["Close"]
    pk, pd_ = stochastic(h, l, c, k_p, d_p)
    k_now, d_now = pk.iloc[-1], pd_.iloc[-1]
    cross_up   = (pk.iloc[-1] > pd_.iloc[-1]) and (pk.iloc[-2] <= pd_.iloc[-2])
    cross_down = (pk.iloc[-1] < pd_.iloc[-1]) and (pk.iloc[-2] >= pd_.iloc[-2])
    in_low  = k_now <= zona_baja
    in_high = k_now >= zona_alta
    if cross_up and in_low:
        return "COMPRA", f"%K = {k_now:.1f} cruza %D al alza en sobreventa (< {zona_baja})"
    if cross_down and in_high:
        return "VENTA",  f"%K = {k_now:.1f} cruza %D a la baja en sobrecompra (> {zona_alta})"
    if in_high:
        return "PRECAUCIÓN", f"Estocástico en sobrecompra: %K = {k_now:.1f}"
    if in_low:
        return "ATENCIÓN", f"Estocástico en sobreventa: %K = {k_now:.1f}"
    return "NEUTRAL", f"%K = {k_now:.1f}, %D = {d_now:.1f} — zona neutral"


# ── Semáforo ──────────────────────────────────────────────────

BADGE_COLORS = {
    "COMPRA"     : ("#1A6B4A", "#E8F5EE"),
    "VENTA"      : ("#8B2A2A", "#FAF0F0"),
    "NEUTRAL"    : ("#8896A8", "#F4F6FB"),
    "NEUTRAL+"   : ("#2266A0", "#EEF4FB"),
    "NEUTRAL-"   : ("#5A4E7A", "#F2F0FB"),
    "PRECAUCIÓN" : ("#8B6914", "#FBF5E8"),
    "ATENCIÓN"   : ("#3A6B8A", "#EEF4F8"),
}

def badge_html(label):
    fg, bg = BADGE_COLORS.get(label, ("#8896A8", "#F4F6FB"))
    return (f'<span style="background:{bg};color:{fg};border:1px solid {fg}44;'
            f'border-radius:4px;padding:2px 8px;font-family:\'IBM Plex Mono\',monospace;'
            f'font-size:0.62rem;font-weight:600;letter-spacing:0.08em;">{label}</span>')

def semaforo_dot(label):
    colores = {
        "COMPRA"    : "#1A6B4A",
        "VENTA"     : "#8B2A2A",
        "NEUTRAL"   : "#8896A8",
        "NEUTRAL+"  : "#2266A0",
        "NEUTRAL-"  : "#5A4E7A",
        "PRECAUCIÓN": "#8B6914",
        "ATENCIÓN"  : "#3A6B8A",
    }
    c = colores.get(label, "#8896A8")
    return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{c};margin-right:6px;"></span>'


def score_label(senales):
    """Convierte señales a score numérico para resumen."""
    score_map = {"COMPRA": 2, "NEUTRAL+": 1, "ATENCIÓN": 1,
                 "NEUTRAL": 0, "NEUTRAL-": -1, "PRECAUCIÓN": -1,
                 "VENTA": -2}
    total = sum(score_map.get(s, 0) for s in senales)
    if total >= 3:   return "COMPRA FUERTE", COLORS["emerald"]
    if total >= 1:   return "SESGO ALCISTA", "#2266A0"
    if total <= -3:  return "VENTA FUERTE",  COLORS["rose"]
    if total <= -1:  return "SESGO BAJISTA", "#5A4E7A"
    return "NEUTRAL", COLORS["text3"]


# ── Gráfico de señales en el tiempo ──────────────────────────

def fig_precio_senales(df, ticker, s1, s2, rsi_w, bb_w, bb_k):
    c = df["Close"]
    col = TICKER_COLORS.get(ticker, COLORS["gold"])
    u, m_bb, l_bb = bollinger(c, bb_w, bb_k)
    ma_s1 = sma(c, s1)
    ma_s2 = sma(c, s2)
    ml, sig, _ = macd(c)
    r = rsi(c, rsi_w)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.03,
    )

    # Panel precio
    fig.add_trace(go.Scatter(x=c.index, y=c.values, name=ticker,
        line=dict(color=col, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=c.index, y=ma_s1.values, name=f"SMA {s1}",
        line=dict(color=COLORS["sky2"], width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=c.index, y=ma_s2.values, name=f"SMA {s2}",
        line=dict(color=COLORS["violet2"], width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=c.index, y=u.values, name="BB+",
        line=dict(color=COLORS["gold"], width=0.8, dash="dot"), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=c.index, y=l_bb.values, name="BB−",
        line=dict(color=COLORS["gold"], width=0.8, dash="dot"), opacity=0.5,
        fill="tonexty", fillcolor="rgba(168,144,96,0.04)"), row=1, col=1)

    # Panel RSI
    fig.add_hline(y=70, line=dict(color=COLORS["rose"], width=0.8, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color=COLORS["emerald"], width=0.8, dash="dash"), row=2, col=1)
    fig.add_trace(go.Scatter(x=r.index, y=r.values, name=f"RSI({rsi_w})",
        line=dict(color=COLORS["gold"], width=1.3)), row=2, col=1)

    # Panel MACD
    colors_hist = [COLORS["emerald"] if v >= 0 else COLORS["rose"] for v in (ml - sig).fillna(0)]
    fig.add_trace(go.Bar(x=ml.index, y=(ml - sig).values, name="Histograma",
        marker_color=colors_hist, marker_line_width=0, opacity=0.8), row=3, col=1)
    fig.add_trace(go.Scatter(x=ml.index, y=ml.values, name="MACD",
        line=dict(color=COLORS["gold"], width=1.3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=sig.index, y=sig.values, name="Señal",
        line=dict(color=COLORS["sky2"], width=1, dash="dot")), row=3, col=1)

    pb = plotly_base(500)
    fig.update_layout(**pb,
        title=dict(text=f"{ticker}  ·  Panel de Señales Técnicas",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")))
    for r_i in [1, 2, 3]:
        fig.update_yaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9),
                         row=r_i, col=1)
        fig.update_xaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9),
                         row=r_i, col=1)
    return fig


# ── Layout ────────────────────────────────────────────────────

def show():
    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">
                Módulo 07
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                Señales & Alertas
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
                    color:#8896A8;letter-spacing:0.08em;">
            MACD · RSI · Bollinger · Golden/Death Cross · Estocástico · Semáforo por activo
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Controles de umbrales ──
    with st.expander("⚙️  Configurar umbrales", expanded=False):
        u1, u2, u3, u4, u5, u6 = st.columns(6)
        with u1: rsi_w      = st.slider("RSI período", 7, 30, 14)
        with u2: sob_compra = st.slider("RSI sobrecompra", 60, 85, 70)
        with u3: sob_venta  = st.slider("RSI sobreventa",  15, 40, 30)
        with u4: s1_cross   = st.slider("SMA corta (cruce)", 5, 50,  20)
        with u5: s2_cross   = st.slider("SMA larga (cruce)", 20, 200, 50)
        with u6: bb_w       = st.slider("BB período", 10, 50, 20)
        u7, u8 = st.columns(2)
        with u7: bb_k = st.slider("BB desviaciones estándar", 1.0, 3.0, 2.0, 0.5)
        with u8: stoch_alta = st.slider("Estocástico sobrecompra", 70, 90, 80)
        stoch_baja = st.slider("Estocástico sobreventa", 10, 30, 20)
    # defaults si no se cambia el expander
    if "rsi_w" not in dir():
        rsi_w = 14; sob_compra = 70; sob_venta = 30
        s1_cross = 20; s2_cross = 50; bb_w = 20; bb_k = 2.0
        stoch_alta = 80; stoch_baja = 20

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Carga de datos ──
    with st.spinner("Cargando datos de todos los activos..."):
        data_all = {}
        for t in TICKERS:
            df = get_ohlcv(t, years=1)
            if not df.empty:
                data_all[t] = df

    # ── Panel de semáforos — todos los activos ──
    sec_labels = ["MACD", "RSI", "Bollinger", "Golden Cross", "Estocástico"]

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                letter-spacing:0.16em;text-transform:uppercase;margin-bottom:1rem;
                border-left:2px solid #8B6914;padding-left:8px;">
        Panel de alertas — Semáforo por activo
    </div>
    """, unsafe_allow_html=True)

    cols_grid = st.columns(len(TICKERS))

    for i, ticker in enumerate(TICKERS):
        if ticker not in data_all:
            continue
        df = data_all[ticker]
        col = TICKER_COLORS.get(ticker, COLORS["gold"])

        s_macd  = eval_macd(df)
        s_rsi   = eval_rsi(df, rsi_w, sob_compra, sob_venta)
        s_bb    = eval_bollinger(df, bb_w, bb_k)
        s_gc    = eval_golden_cross(df, s1_cross, s2_cross)
        s_stoch = eval_stoch(df, 14, 3, stoch_alta, stoch_baja)

        senales = [s_macd[0], s_rsi[0], s_bb[0], s_gc[0], s_stoch[0]]
        resumen, res_color = score_label(senales)

        ultimo = df["Close"].iloc[-1]
        cambio = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100

        with cols_grid[i]:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                        border-top:3px solid {col};border-radius:8px;
                        padding:1.1rem;margin-bottom:0.5rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;
                            margin-bottom:10px;">
                    <div>
                        <span style="font-family:'Playfair Display',serif;font-size:1.05rem;
                                     font-weight:700;color:#1A2035;">{ticker}</span>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.52rem;
                                    color:#8896A8;margin-top:1px;">{SECTOR_MAP[ticker]}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;
                                    font-weight:600;color:#1A2035;">${ultimo:.2f}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                                    color:{'#1A6B4A' if cambio >= 0 else '#8B2A2A'};">
                            {'+' if cambio >= 0 else ''}{cambio:.2f}%</div>
                    </div>
                </div>

                <div style="background:{res_color}22;border:1px solid {res_color}44;
                            border-radius:5px;padding:5px 8px;text-align:center;
                            margin-bottom:10px;">
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                                 font-weight:700;color:{res_color};letter-spacing:0.08em;">
                        {resumen}
                    </span>
                </div>

                <div style="display:flex;flex-direction:column;gap:5px;">
                    {"".join([
                        f'''<div style="display:flex;justify-content:space-between;
                                align-items:center;padding:3px 0;
                                border-bottom:1px solid #F4F6FB;">
                            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.57rem;
                                         color:#8896A8;">{sec_labels[j]}</span>
                            {badge_html(s[0])}
                        </div>'''
                        for j, s in enumerate([s_macd, s_rsi, s_bb, s_gc, s_stoch])
                    ])}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Detalle por activo ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                border-left:2px solid #3A6B8A;padding-left:8px;">
        Análisis detallado por activo
    </div>
    """, unsafe_allow_html=True)

    ticker_det = st.selectbox("Activo para análisis detallado", TICKERS)

    if ticker_det in data_all:
        df_det = data_all[ticker_det]

        # Señales detalle
        sd_macd  = eval_macd(df_det)
        sd_rsi   = eval_rsi(df_det, rsi_w, sob_compra, sob_venta)
        sd_bb    = eval_bollinger(df_det, bb_w, bb_k)
        sd_gc    = eval_golden_cross(df_det, s1_cross, s2_cross)
        sd_stoch = eval_stoch(df_det, 14, 3, stoch_alta, stoch_baja)

        senales_det = [sd_macd, sd_rsi, sd_bb, sd_gc, sd_stoch]

        for j, (label, desc) in enumerate(senales_det):
            fg, bg = BADGE_COLORS.get(label, ("#8896A8", "#F4F6FB"))
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                        border-left:3px solid {fg};border-radius:6px;
                        padding:0.7rem 1rem;margin-bottom:0.5rem;
                        display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap;">
                <div style="min-width:90px;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
                                color:#8896A8;letter-spacing:0.1em;text-transform:uppercase;">
                        {sec_labels[j]}</div>
                    {badge_html(label)}
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:0.8rem;color:#4A5568;
                            flex:1;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.plotly_chart(
            fig_precio_senales(df_det, ticker_det, s1_cross, s2_cross, rsi_w, bb_w, bb_k),
            use_container_width=True
        )

        with st.expander("Metodología — Generación de señales"):
            st.markdown(f"""
            **MACD:** se detecta cruce cuando la línea MACD cambia de posición respecto a la señal
            entre la penúltima y última observación. Cruce al alza → COMPRA; cruce a la baja → VENTA.

            **RSI:** zona de sobrecompra configurada en **{sob_compra}** (señal VENTA),
            sobreventa en **{sob_venta}** (señal COMPRA). Período: **{rsi_w}**.

            **Bandas de Bollinger:** precio tocando o superando la banda superior → VENTA;
            tocando o bajando de la banda inferior → COMPRA. Período: **{bb_w}**, k = **{bb_k}σ**.

            **Golden/Death Cross:** SMA({s1_cross}) cruza SMA({s2_cross}). Cruce al alza → COMPRA
            (*Golden Cross*). Cruce a la baja → VENTA (*Death Cross*).

            **Estocástico:** cruce de %K sobre %D en zona de sobreventa (< {stoch_baja}) → COMPRA;
            cruce a la baja en sobrecompra (> {stoch_alta}) → VENTA.

            > **Nota académica:** ninguna señal técnica tiene valor predictivo garantizado.
            > Se recomienda combinar múltiples indicadores y confirmar con análisis fundamental.
            > El rendimiento pasado de las señales no garantiza resultados futuros.
            """)
