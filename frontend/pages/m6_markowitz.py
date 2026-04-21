"""
pages/m6_markowitz.py — Módulo 6: Optimización de Portafolio (Markowitz)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.client import get_rendimientos, post_frontera, get_macro, SECTOR_MAP
from utils.theme import plotly_base, COLORS
from utils.dynamic_tickers import get_tickers, get_ticker_colors, render_portafolio_badge


def sec_title(text, color=None):
    col = color or COLORS["gold"]
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                border-left:2px solid {col};padding-left:8px;">
        {text}
    </div>
    """, unsafe_allow_html=True)


def portfolio_metrics(weights, mu_daily, cov_daily, rf_daily):
    w      = np.array(weights)
    ret_a  = np.dot(w, mu_daily) * 252
    vol_a  = np.sqrt(w @ cov_daily @ w) * np.sqrt(252)
    sharpe = (ret_a - rf_daily * 252) / vol_a if vol_a > 0 else 0
    return ret_a, vol_a, sharpe


def simulate_portfolios(mu_daily, cov_daily, rf_daily, n=10_000):
    n_assets = len(mu_daily)
    rets, vols, sharpes, weights_all = [], [], [], []
    np.random.seed(0)
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        r, v, s = portfolio_metrics(w, mu_daily, cov_daily, rf_daily)
        rets.append(r)
        vols.append(v)
        sharpes.append(s)
        weights_all.append(w)
    return np.array(rets), np.array(vols), np.array(sharpes), np.array(weights_all)


def min_variance_portfolio(mu_daily, cov_daily, rf_daily):
    n = len(mu_daily)
    def neg_sharpe(w):
        r, v, s = portfolio_metrics(w, mu_daily, cov_daily, rf_daily)
        return -s
    def portfolio_vol(w):
        return np.sqrt(w @ cov_daily @ w) * np.sqrt(252)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n
    res_mv = minimize(portfolio_vol, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    res_ms = minimize(neg_sharpe,    w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res_mv.x, res_ms.x


def efficient_frontier(mu_daily, cov_daily, rf_daily, n_points=80):
    n = len(mu_daily)
    min_ret = np.dot(np.ones(n) / n, mu_daily) * 252 * 0.5
    max_ret = mu_daily.max() * 252 * 1.05
    targets = np.linspace(min_ret, max_ret, n_points)
    ef_vols, ef_rets = [], []
    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mu_daily) * 252 - t},
        ]
        res = minimize(lambda w: np.sqrt(w @ cov_daily @ w) * np.sqrt(252),
                       np.ones(n) / n, method="SLSQP",
                       bounds=[(0, 1)] * n, constraints=constraints)
        if res.success:
            ef_vols.append(res.fun)
            ef_rets.append(target)
    return np.array(ef_vols), np.array(ef_rets)


def fig_heatmap_corr(returns_df, TICKERS):
    """Matriz con misma paleta que Vista General: blanco → gris → dorado."""
    corr = returns_df[TICKERS].corr()
    fig  = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0.0, "#EEF1F6"],
            [0.5, "#C4CBD8"],
            [1.0, "#8B6914"],
        ],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, color="#1A2035"),
        showscale=False,
    ))
    pb = plotly_base(340)
    pb["xaxis"] = dict(gridcolor=COLORS["border2"], showline=False,
                       tickfont=dict(color=COLORS["text3"], size=10))
    pb["yaxis"] = dict(gridcolor=COLORS["border2"], showline=False,
                       tickfont=dict(color=COLORS["text3"], size=10))
    fig.update_layout(**pb,
        title=dict(text="Matriz de Correlación — Log-Rendimientos Diarios",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")))
    return fig


def fig_frontera(sim_rets, sim_vols, sim_sharpes,
                 ef_vols, ef_rets,
                 w_mv, w_ms, mu_daily, cov_daily, rf_daily, tickers):
    TICKER_COLORS = get_ticker_colors()
    ret_mv, vol_mv, shr_mv = portfolio_metrics(w_mv, mu_daily, cov_daily, rf_daily)
    ret_ms, vol_ms, shr_ms = portfolio_metrics(w_ms, mu_daily, cov_daily, rf_daily)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_vols, y=sim_rets, mode="markers",
        marker=dict(color=sim_sharpes, colorscale="YlOrBr",
                    size=3, opacity=0.45,
                    colorbar=dict(title="Sharpe", thickness=10,
                                  tickfont=dict(color=COLORS["text3"], size=8),
                                  outlinewidth=0)),
        name="Portafolios simulados",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    ))
    if len(ef_vols) > 0:
        fig.add_trace(go.Scatter(
            x=ef_vols, y=ef_rets, mode="lines",
            name="Frontera Eficiente",
            line=dict(color=COLORS["gold"], width=2.5),
        ))
    fig.add_trace(go.Scatter(
        x=[vol_mv], y=[ret_mv], mode="markers+text",
        name=f"Mín. Varianza (Sharpe={shr_mv:.2f})",
        text=["MV"], textposition="top right",
        textfont=dict(color=COLORS["sky"], size=11, family="IBM Plex Mono"),
        marker=dict(color=COLORS["sky"], size=16, symbol="diamond",
                    line=dict(color="white", width=2)),
    ))
    fig.add_trace(go.Scatter(
        x=[vol_ms], y=[ret_ms], mode="markers+text",
        name=f"Máx. Sharpe ({shr_ms:.2f})",
        text=["MS"], textposition="top right",
        textfont=dict(color=COLORS["emerald"], size=11, family="IBM Plex Mono"),
        marker=dict(color=COLORS["emerald"], size=16, symbol="star",
                    line=dict(color="white", width=2)),
    ))
    for t in tickers:
        idx = tickers.index(t)
        r_t = mu_daily[idx] * 252
        v_t = np.sqrt(cov_daily[idx, idx]) * np.sqrt(252)
        col = TICKER_COLORS.get(t, COLORS["gold"])
        fig.add_trace(go.Scatter(
            x=[v_t], y=[r_t], mode="markers+text",
            name=t, text=[t], textposition="top center",
            textfont=dict(color=col, size=9, family="IBM Plex Mono"),
            marker=dict(color=col, size=10, line=dict(color="white", width=1.5)),
        ))
    pb = plotly_base(480)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    pb.pop("hovermode", None)
    fig.update_layout(**pb,
        title=dict(text="Conjunto Factible y Frontera Eficiente — Markowitz",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Volatilidad Anual", xaxis_tickformat=".1%",
        yaxis_title="Rendimiento Esperado Anual", yaxis_tickformat=".1%",
        hovermode="closest")
    return fig


def fig_composicion(w_mv, w_ms, tickers):
    TICKER_COLORS = get_ticker_colors()
    colors = [TICKER_COLORS.get(t, COLORS["gold"]) for t in tickers]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Mínima Varianza", "Máximo Sharpe"],
                        specs=[[{"type": "pie"}, {"type": "pie"}]])
    fig.add_trace(go.Pie(
        labels=tickers, values=w_mv * 100, marker_colors=colors,
        textinfo="label+percent", textfont=dict(size=10, family="IBM Plex Mono"),
        hole=0.35, name="MV",
    ), row=1, col=1)
    fig.add_trace(go.Pie(
        labels=tickers, values=w_ms * 100, marker_colors=colors,
        textinfo="label+percent", textfont=dict(size=10, family="IBM Plex Mono"),
        hole=0.35, name="MS",
    ), row=1, col=2)
    pb = plotly_base(320)
    pb.pop("xaxis", None); pb.pop("yaxis", None); pb.pop("hovermode", None)
    fig.update_layout(**pb,
        title=dict(text="Composición de Portafolios Óptimos",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        showlegend=False)
    return fig


def render_portfolio_kpis(label, color, ret, vol, sharpe, weights, tickers):
    """Tarjeta completa con KPIs y composición de pesos."""
    color_text2 = COLORS["text2"]
    filas = []
    for t, w in zip(tickers, weights):
        pct = f"{w*100:.2f}%"
        fila = (
            '<div style="display:flex;justify-content:space-between;padding:4px 0;'
            'border-bottom:1px solid #F4F6FB">'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:.62rem;'
            f'font-weight:600;color:{color_text2}">{t}</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:.62rem;'
            f'color:{color};font-weight:700">{pct}</span>'
            '</div>'
        )
        filas.append(fila)
    pesos_html = "".join(filas)
    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #D8DDE8;
    border-top:3px solid {color};border-radius:8px;padding:1.2rem">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
        color:#8896A8;letter-spacing:.18em;text-transform:uppercase;margin-bottom:1rem">
        {label}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.5rem;margin-bottom:1rem">
            <div style="background:#F4F6FB;border-radius:6px;padding:.7rem">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.48rem;
                color:#8896A8;letter-spacing:.1em;text-transform:uppercase;margin-bottom:3px">
                Retorno anual</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
                font-weight:700;color:{color}">{ret:.2%}</div>
            </div>
            <div style="background:#F4F6FB;border-radius:6px;padding:.7rem">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.48rem;
                color:#8896A8;letter-spacing:.1em;text-transform:uppercase;margin-bottom:3px">
                Volatilidad</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
                font-weight:700;color:#1A2035">{vol:.2%}</div>
            </div>
            <div style="background:#F4F6FB;border-radius:6px;padding:.7rem">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.48rem;
                color:#8896A8;letter-spacing:.1em;text-transform:uppercase;margin-bottom:3px">
                Sharpe</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
                font-weight:700;color:#1A2035">{sharpe:.3f}</div>
            </div>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:.48rem;
        color:#8896A8;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem">
        Composición</div>
        {pesos_html}
    </div>
    """, unsafe_allow_html=True)


def show():
    render_portafolio_badge()
    TICKERS = get_tickers()

    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">
                Módulo 06
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                Markowitz
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
                    color:#8896A8;letter-spacing:0.08em;">
            Frontera Eficiente · Mínima Varianza · Máximo Sharpe · Portafolio Óptimo
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Cargando datos y optimizando portafolio..."):
        all_log = {}
        for t in TICKERS:
            d = get_rendimientos(t, years=3)
            idx = pd.to_datetime(d["fechas"])
            all_log[t] = pd.Series(d["log_returns"], index=idx)
        log_ret = pd.DataFrame(all_log).dropna()
        macro   = get_macro()
        rf = {"annual":  macro["tasa_libre_riesgo"]["valor"],
              "daily":   macro["tasa_libre_riesgo"]["valor"] / 252,
              "display": macro["tasa_libre_riesgo"]["display"]}

    mu_daily  = log_ret[TICKERS].mean().values
    cov_daily = log_ret[TICKERS].cov().values
    rf_daily  = rf["daily"]

    c1, c2 = st.columns([1, 2])
    with c1:
        n_sim = st.selectbox("Portafolios a simular", [10_000, 30_000, 50_000],
                             index=0, format_func=lambda x: f"{x:,}")
    with c2:
        ret_obj_pct = st.slider("Rendimiento objetivo anual (portafolio personalizado)",
                                min_value=5.0, max_value=40.0, value=15.0,
                                step=0.5, format="%.1f%%")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── 1. Correlación ──
    sec_title("① Matriz de Correlación entre Activos")
    st.plotly_chart(fig_heatmap_corr(log_ret, TICKERS), use_container_width=True)

    with st.expander("Interpretación — Correlaciones y diversificación"):
        st.markdown("""
        La **correlación** mide el co-movimiento entre activos. Valores cercanos a 0 o negativos
        benefician la diversificación — combinar activos poco correlacionados reduce la volatilidad
        del portafolio sin sacrificar retorno esperado (Markowitz, 1952).
        """)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── 2. Frontera ──
    sec_title("② Conjunto Factible y Frontera Eficiente", COLORS["sky"])

    with st.spinner(f"Simulando {n_sim:,} portafolios..."):
        sim_rets, sim_vols, sim_sharpes, _ = simulate_portfolios(mu_daily, cov_daily, rf_daily, n=n_sim)
        w_mv, w_ms = min_variance_portfolio(mu_daily, cov_daily, rf_daily)
        ef_vols, ef_rets = efficient_frontier(mu_daily, cov_daily, rf_daily)

    st.plotly_chart(
        fig_frontera(sim_rets, sim_vols, sim_sharpes, ef_vols, ef_rets,
                     w_mv, w_ms, mu_daily, cov_daily, rf_daily, TICKERS),
        use_container_width=True
    )

    with st.expander("Interpretación — Frontera eficiente"):
        st.markdown("""
        **★ MV (Mínima Varianza):** portafolio de menor volatilidad posible.
        **★ MS (Máximo Sharpe):** maximiza el retorno por unidad de riesgo.
        Ningún portafolio racional debería ubicarse por debajo de la frontera.
        """)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── 3. KPIs + composición en tarjetas ──
    sec_title("③ Portafolios Óptimos — Métricas y Composición", COLORS["gold"])

    ret_mv, vol_mv, shr_mv = portfolio_metrics(w_mv, mu_daily, cov_daily, rf_daily)
    ret_ms, vol_ms, shr_ms = portfolio_metrics(w_ms, mu_daily, cov_daily, rf_daily)

    col_mv, col_ms = st.columns(2)
    with col_mv:
        render_portfolio_kpis("◆ Mínima Varianza", COLORS["sky"],
                              ret_mv, vol_mv, shr_mv, w_mv, TICKERS)
    with col_ms:
        render_portfolio_kpis("★ Máximo Sharpe", COLORS["emerald"],
                              ret_ms, vol_ms, shr_ms, w_ms, TICKERS)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── 4. Gráfico de tortas ──
    sec_title("④ Composición Visual de Portafolios Óptimos", COLORS["emerald"])
    st.plotly_chart(fig_composicion(w_mv, w_ms, TICKERS), use_container_width=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── 5. Portafolio con objetivo ──
    sec_title("⑤ Portafolio Eficiente con Rendimiento Objetivo", COLORS["violet"])
    ret_obj = ret_obj_pct / 100
    n = len(mu_daily)
    constraints_obj = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: np.dot(w, mu_daily) * 252 - ret_obj},
    ]
    res_obj = minimize(
        lambda w: np.sqrt(w @ cov_daily @ w) * np.sqrt(252),
        np.ones(n) / n, method="SLSQP",
        bounds=[(0, 1)] * n, constraints=constraints_obj
    )
    if res_obj.success:
        w_obj = res_obj.x
        ret_o, vol_o, shr_o = portfolio_metrics(w_obj, mu_daily, cov_daily, rf_daily)
        o1, o2, o3 = st.columns(3)
        o1.metric("Retorno Anual Objetivo", f"{ret_o:.2%}")
        o2.metric("Volatilidad Mínima",     f"{vol_o:.2%}")
        o3.metric("Ratio de Sharpe",        f"{shr_o:.3f}")
        rows_obj = [{"Ticker": TICKERS[i], "Sector": SECTOR_MAP.get(TICKERS[i], "—"),
                     "Peso": f"{w_obj[i]*100:.2f}%"} for i in range(n)]
        st.dataframe(pd.DataFrame(rows_obj), use_container_width=True, hide_index=True)
    else:
        st.warning(f"No existe portafolio factible con retorno {ret_obj_pct:.1f}%. Ajusta el objetivo.")