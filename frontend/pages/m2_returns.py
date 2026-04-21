"""
pages/m2_returns.py — Módulo 2: Rendimientos
Streamlit + Plotly | yfinance 1.2.0
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.client import get_rendimientos, get_precios, SECTOR_MAP
from utils.theme import plotly_base, COLORS
from utils.dynamic_tickers import get_tickers, get_ticker_colors, render_portafolio_badge


def sec_title(text, color=None):
    col = color or COLORS["gold"]
    return st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                border-left:2px solid {col};padding-left:8px;">
        {text}
    </div>
    """, unsafe_allow_html=True)


def fig_histogram(r, ticker):
    TICKER_COLORS = get_ticker_colors()
    mu, sigma = r.mean(), r.std()
    x = np.linspace(r.min(), r.max(), 300)
    pdf_normal = stats.norm.pdf(x, mu, sigma)
    pdf_normal = pdf_normal * len(r) * (r.max() - r.min()) / 60

    col = TICKER_COLORS.get(ticker, COLORS["gold"])
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r.values, nbinsx=60, name="Rendimientos",
        marker_color=col, opacity=0.7, marker_line_width=0,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pdf_normal, name="Normal teórica",
        line=dict(color=COLORS["rose"], width=1.8, dash="dash"),
    ))
    fig.add_vline(x=mu, line=dict(color=COLORS["gold"], width=1.2, dash="dot"),
                  annotation_text=f"μ={mu:.4f}",
                  annotation_font=dict(color=COLORS["gold"], size=9))
    pb = plotly_base(300)
    pb["xaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text=f"{ticker}  ·  Distribución de Log-Rendimientos",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        bargap=0.05, showlegend=True)
    return fig


def fig_qq(r, ticker):
    TICKER_COLORS = get_ticker_colors()
    (osm, osr), (slope, intercept, _) = stats.probplot(r.values, dist="norm")
    line_y = slope * np.array(osm) + intercept
    col = TICKER_COLORS.get(ticker, COLORS["gold"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers", name="Cuantiles empíricos",
        marker=dict(color=col, size=3.5, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=osm, y=line_y, mode="lines", name="Normal teórica",
        line=dict(color=COLORS["rose"], width=1.8, dash="dash"),
    ))
    pb = plotly_base(300)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text=f"{ticker}  ·  Q-Q Plot vs Normal",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Cuantiles teóricos", yaxis_title="Cuantiles empíricos")
    return fig


def fig_volatility(r, ticker):
    TICKER_COLORS = get_ticker_colors()
    r2 = r ** 2
    col = TICKER_COLORS.get(ticker, COLORS["gold"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=r2.index, y=r2.values, name="r²",
        line=dict(color=col, width=0.9), opacity=0.85,
    ))
    fig.update_layout(**plotly_base(240),
        title=dict(text=f"{ticker}  ·  Cuadrado de rendimientos (r²)  —  Agrupamiento de Volatilidad",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        showlegend=False)
    return fig


def fig_boxplot(returns_df, TICKERS):
    TICKER_COLORS = get_ticker_colors()
    fig = go.Figure()
    for ticker in TICKERS:
        if ticker not in returns_df.columns:
            continue
        col = TICKER_COLORS.get(ticker, COLORS["gold"])
        fig.add_trace(go.Box(
            y=returns_df[ticker].values, name=ticker,
            marker_color=col, line_color=col,
            fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.15)",
            boxmean="sd", line_width=1.3,
        ))
    pb = plotly_base(340)
    pb["xaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text="Boxplot comparativo — Log-Rendimientos diarios",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        showlegend=False)
    return fig


def descriptive_stats(r):
    return {
        "Media diaria"      : r.mean(),
        "Media anualizada"  : r.mean() * 252,
        "Desv. Std. diaria" : r.std(),
        "Desv. Std. anual"  : r.std() * np.sqrt(252),
        "Asimetría"         : float(stats.skew(r)),
        "Curtosis (exceso)" : float(stats.kurtosis(r)),
        "Mínimo"            : r.min(),
        "Máximo"            : r.max(),
        "Observaciones"     : len(r),
    }


def normality_tests(r):
    jb_s, jb_p = stats.jarque_bera(r.values)
    sw_s, sw_p = stats.shapiro(r.values[:min(len(r), 5000)])
    return {"jb_stat": jb_s, "jb_p": jb_p, "sw_stat": sw_s, "sw_p": sw_p}


def kpi_card(label, value, color="#8B6914", bg="#FFFFFF", border_style="top"):
    border_css = (
        f"border-top:3px solid {color};"
        if border_style == "top"
        else f"border-left:3px solid {color};"
    )
    return f"""
    <div style="background:{bg};border:1px solid #D8DDE8;{border_css}
    border-radius:8px;padding:1.1rem 1rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:.52rem;
        color:#8896A8;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.5rem">
        {label}</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.35rem;
        font-weight:700;color:#1A2035">{value}</div>
    </div>
    """


def show():
    render_portafolio_badge()

    TICKERS = get_tickers()
    TICKER_COLORS = get_ticker_colors()

    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">
                Módulo 02
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                Rendimientos
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
                    color:#8896A8;letter-spacing:0.08em;">
            Caracterización estadística · Pruebas de normalidad · Hechos estilizados
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Cargando datos..."):
        all_log, all_sim = {}, {}
        for t in TICKERS:
            d = get_rendimientos(t, years=3)
            idx = pd.to_datetime(d["fechas"])
            all_log[t] = pd.Series(d["log_returns"], index=idx)
            all_sim[t] = pd.Series(d["simple_returns"], index=idx)
        log_ret = pd.DataFrame(all_log).dropna()
        sim_ret = pd.DataFrame(all_sim).dropna()

    c1, c2 = st.columns([1, 2])
    with c1:
        ticker = st.selectbox("Activo", TICKERS, index=0)
    with c2:
        ret_type = st.radio("Tipo de rendimiento",
                            ["Log-rendimiento (recomendado)", "Rendimiento simple"],
                            horizontal=True)

    r = log_ret[ticker] if "Log" in ret_type else sim_ret[ticker]
    label = "Log-rendimiento" if "Log" in ret_type else "Rendimiento simple"

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── KPIs — fila 1: métricas principales ──
    sec_title("Estadísticas Descriptivas")

    fmt_map = {
        "Media diaria"      : "{:.5f}",
        "Media anualizada"  : "{:.2%}",
        "Desv. Std. diaria" : "{:.5f}",
        "Desv. Std. anual"  : "{:.2%}",
        "Asimetría"         : "{:.4f}",
        "Curtosis (exceso)" : "{:.4f}",
        "Mínimo"            : "{:.4f}",
        "Máximo"            : "{:.4f}",
        "Observaciones"     : "{:.0f}",
    }

    st_vals = descriptive_stats(r)

    # Fila 1 — 4 métricas de rentabilidad/riesgo
    row1 = st.columns(4)
    kpis1 = [
        ("Media diaria",      COLORS["gold"],    "#FFFFFF", "top"),
        ("Media anualizada",  COLORS["gold"],    "#FFFFFF", "top"),
        ("Desv. Std. diaria", COLORS["sky"],     "#FFFFFF", "top"),
        ("Desv. Std. anual",  COLORS["sky"],     "#FFFFFF", "top"),
    ]
    for col, (k, color, bg, bstyle) in zip(row1, kpis1):
        with col:
            st.markdown(
                kpi_card(k, fmt_map[k].format(st_vals[k]), color, bg, bstyle),
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    # Fila 2 — 5 métricas de forma/distribución
    row2 = st.columns(5)
    kpis2 = [
        ("Asimetría",         COLORS["rose"],    "#FAF4F4", "left"),
        ("Curtosis (exceso)", COLORS["rose"],    "#FAF4F4", "left"),
        ("Mínimo",            COLORS["text"],    "#F4F6FB", "left"),
        ("Máximo",            COLORS["emerald"], "#F4F6FB", "left"),
        ("Observaciones",     COLORS["text"],    "#F4F6FB", "left"),
    ]
    for col, (k, color, bg, bstyle) in zip(row2, kpis2):
        with col:
            st.markdown(
                kpi_card(k, fmt_map[k].format(st_vals[k]), color, bg, bstyle),
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Histograma y Q-Q ──
    col_l, col_r = st.columns(2)
    with col_l:
        sec_title(f"Distribución de {label}")
        st.plotly_chart(fig_histogram(r, ticker), use_container_width=True)
    with col_r:
        sec_title("Q-Q Plot vs Distribución Normal")
        st.plotly_chart(fig_qq(r, ticker), use_container_width=True)

    with st.expander("Interpretación — Distribución y Q-Q Plot"):
        st.markdown("""
        El **histograma** compara la distribución empírica de los rendimientos con una
        distribución normal de igual media y desviación estándar.

        El **Q-Q Plot** enfrenta los cuantiles empíricos contra los teóricos normales.
        Las desviaciones en los extremos evidencian **colas pesadas** (*fat tails*), uno de
        los hechos estilizados más robustos de los mercados financieros.
        """)

    # ── Volatilidad y Boxplot ──
    sec_title("Agrupamiento de Volatilidad (Volatility Clustering)", COLORS["sky"])
    st.plotly_chart(fig_volatility(r, ticker), use_container_width=True)

    with st.expander("Interpretación — Agrupamiento de Volatilidad"):
        st.markdown("""
        El gráfico de **r²** evidencia que períodos de alta volatilidad tienden a agruparse.
        Este fenómeno justifica el uso de modelos **ARCH/GARCH** (Módulo 3).
        """)

    sec_title("Boxplot Comparativo — Todos los Activos", COLORS["violet"])
    st.plotly_chart(fig_boxplot(log_ret, TICKERS), use_container_width=True)

    # ── Pruebas de normalidad ──
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    sec_title("Pruebas de Normalidad", COLORS["rose"])

    tests = normality_tests(r)
    rechaza_jb = tests["jb_p"] < 0.05
    rechaza_sw = tests["sw_p"] < 0.05

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                    border-left:3px solid {'#8B2A2A' if rechaza_jb else '#1A6B4A'};
                    border-radius:6px;padding:1.2rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                        color:#8896A8;letter-spacing:0.12em;text-transform:uppercase;
                        margin-bottom:8px;">Jarque-Bera</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
                        font-weight:600;color:#1A2035;">stat = {tests['jb_stat']:.4f}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                        color:#4A5568;margin-top:4px;">p-valor = {tests['jb_p']:.4f}</div>
            <div style="font-size:0.78rem;margin-top:8px;
                        color:{'#8B2A2A' if rechaza_jb else '#1A6B4A'};font-weight:600;">
                {'⛔ Rechaza normalidad' if rechaza_jb else '✅ No rechaza normalidad'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                    border-left:3px solid {'#8B2A2A' if rechaza_sw else '#1A6B4A'};
                    border-radius:6px;padding:1.2rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                        color:#8896A8;letter-spacing:0.12em;text-transform:uppercase;
                        margin-bottom:8px;">Shapiro-Wilk</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
                        font-weight:600;color:#1A2035;">stat = {tests['sw_stat']:.4f}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                        color:#4A5568;margin-top:4px;">p-valor = {tests['sw_p']:.4f}</div>
            <div style="font-size:0.78rem;margin-top:8px;
                        color:{'#8B2A2A' if rechaza_sw else '#1A6B4A'};font-weight:600;">
                {'⛔ Rechaza normalidad' if rechaza_sw else '✅ No rechaza normalidad'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                    border-left:3px solid #8B6914;border-radius:6px;padding:1.2rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                        color:#8896A8;letter-spacing:0.12em;text-transform:uppercase;
                        margin-bottom:8px;">Interpretación</div>
            <div style="font-size:0.78rem;color:#4A5568;line-height:1.65;">
                Ambas pruebas evalúan H₀: distribución normal.<br><br>
                En series financieras H₀ se rechaza casi siempre por
                <b>colas pesadas</b> y <b>asimetría</b>. Esto impacta el
                VaR paramétrico y justifica usar distribución <i>t</i> en GARCH.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Hechos estilizados ──
    sec_title("Hechos Estilizados de los Mercados Financieros", COLORS["gold"])

    h1, h2, h3, h4 = st.columns(4)
    hechos = [
        (h1, "Colas Pesadas", "Curtosis > 3. Los eventos extremos son más frecuentes que bajo normalidad. El Q-Q plot lo evidencia en los extremos.", COLORS["gold"]),
        (h2, "Agrupamiento de Volatilidad", "Alta volatilidad sigue a alta volatilidad. La dependencia temporal en r² motiva los modelos GARCH.", COLORS["sky"]),
        (h3, "Asimetría Negativa", "Los mercados caen más rápido de lo que suben. Pérdidas extremas son más probables que ganancias equivalentes.", COLORS["rose"]),
        (h4, "Efecto Apalancamiento", "Caídas del precio aumentan la volatilidad más que subidas equivalentes. Correlación negativa entre r y σ futura.", COLORS["emerald"]),
    ]

    for col, titulo, texto, color in hechos:
        with col:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                        border-top:2px solid {color};border-radius:6px;
                        padding:1.1rem;height:100%;">
                <div style="font-family:'Playfair Display',serif;font-size:0.9rem;
                            font-weight:600;color:#1A2035;margin-bottom:8px;">
                    {titulo}
                </div>
                <div style="font-size:0.78rem;color:#4A5568;line-height:1.65;">
                    {texto}
                </div>
            </div>
            """, unsafe_allow_html=True)