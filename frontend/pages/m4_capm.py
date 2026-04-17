"""
pages/m4_capm.py — Módulo 4: CAPM y Beta
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
from data.client import get_capm, get_precios, TICKERS, TICKER_COLORS, SECTOR_MAP
BENCHMARK = '^GSPC'
from utils.theme import plotly_base, COLORS


# ── Cálculos CAPM ─────────────────────────────────────────────

def compute_beta(ret_asset, ret_mkt):
    df = pd.concat([ret_asset, ret_mkt], axis=1).dropna()
    df.columns = ["asset", "mkt"]
    slope, intercept, r, p, se = stats.linregress(df["mkt"], df["asset"])
    return {
        "beta"     : slope,
        "alpha"    : intercept,
        "r2"       : r ** 2,
        "p_beta"   : p,
        "se"       : se,
        "r_asset"  : df["asset"],
        "r_mkt"    : df["mkt"],
    }

def classify_beta(beta):
    if beta > 1.2:   return "Agresivo",  COLORS["rose"]
    if beta < 0.8:   return "Defensivo", COLORS["emerald"]
    return "Neutro", COLORS["gold"]

def expected_return_capm(beta, rf_daily, rm_daily):
    return rf_daily + beta * (rm_daily - rf_daily)

def decompose_risk(beta, ret_asset, ret_mkt):
    var_total = ret_asset.var()
    var_mkt   = ret_mkt.var()
    var_sist  = (beta ** 2) * var_mkt
    var_idio  = max(var_total - var_sist, 0)
    pct_sist  = var_sist / var_total * 100 if var_total > 0 else 0
    pct_idio  = 100 - pct_sist
    return pct_sist, pct_idio


# ── Gráficos ──────────────────────────────────────────────────

def fig_scatter_single(reg, ticker):
    col   = TICKER_COLORS.get(ticker, COLORS["gold"])
    x_rng = np.linspace(reg["r_mkt"].min(), reg["r_mkt"].max(), 200)
    y_fit = reg["alpha"] + reg["beta"] * x_rng
    ci    = 1.96 * reg["se"] * np.sqrt(
                1 / len(reg["r_mkt"]) +
                (x_rng - reg["r_mkt"].mean()) ** 2 /
                ((reg["r_mkt"] - reg["r_mkt"].mean()) ** 2).sum()
            )
    fig = go.Figure()
    # IC 95%
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_rng, x_rng[::-1]]),
        y=np.concatenate([y_fit + ci, (y_fit - ci)[::-1]]),
        fill="toself", fillcolor="rgba(139,105,20,0.06)",
        line=dict(width=0), name="IC 95%", hoverinfo="skip",
    ))
    # Puntos
    fig.add_trace(go.Scatter(
        x=reg["r_mkt"].values, y=reg["r_asset"].values,
        mode="markers", name=ticker,
        marker=dict(color=col, size=4, opacity=0.35),
        hovertemplate="Mkt: %{x:.4f}<br>" + ticker + ": %{y:.4f}<extra></extra>",
    ))
    # Línea de regresión
    fig.add_trace(go.Scatter(
        x=x_rng, y=y_fit, mode="lines",
        name=f"β = {reg['beta']:.4f}",
        line=dict(color=col, width=2),
    ))
    # Línea de mercado (β=1)
    fig.add_trace(go.Scatter(
        x=x_rng, y=x_rng, mode="lines",
        name="β = 1 (mercado)",
        line=dict(color=COLORS["text3"], width=1, dash="dot"),
    ))
    pb = plotly_base(360)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    pb.pop("hovermode", None)
    fig.update_layout(**pb,
        title=dict(
            text=f"{ticker} vs S&P 500  ·  β={reg['beta']:.4f}  ·  R²={reg['r2']:.4f}",
            font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Log-rend. S&P 500",
        yaxis_title=f"Log-rend. {ticker}",
        hovermode="closest")
    return fig


def fig_scatter_all(log_ret):
    fig = go.Figure()
    ret_mkt = log_ret[BENCHMARK]
    for ticker in TICKERS:
        reg = compute_beta(log_ret[ticker], ret_mkt)
        col = TICKER_COLORS.get(ticker, COLORS["gold"])
        fig.add_trace(go.Scatter(
            x=reg["r_mkt"].values, y=reg["r_asset"].values,
            mode="markers", name=ticker,
            marker=dict(color=col, size=3.5, opacity=0.25),
        ))
        x_rng = np.linspace(ret_mkt.min(), ret_mkt.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_rng, y=reg["alpha"] + reg["beta"] * x_rng,
            mode="lines", name=f"{ticker} β={reg['beta']:.2f}",
            line=dict(color=col, width=1.6),
        ))
    pb = plotly_base(380)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    pb.pop("hovermode", None)
    fig.update_layout(**pb,
        title=dict(text="Todos los activos vs S&P 500",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Log-rend. S&P 500",
        yaxis_title="Log-rend. Activo",
        hovermode="closest")
    return fig


def fig_sml(betas, exp_rets, rf_daily, rm_daily):
    b_range = np.linspace(-0.2, 2.0, 200)
    sml_y   = (rf_daily + b_range * (rm_daily - rf_daily)) * 252
    fig = go.Figure()
    # SML
    fig.add_trace(go.Scatter(
        x=b_range, y=sml_y, mode="lines", name="SML",
        line=dict(color=COLORS["text3"], width=1.2, dash="dot"),
    ))
    # Rf
    fig.add_trace(go.Scatter(
        x=[0], y=[rf_daily * 252], mode="markers+text",
        name="Rf", text=["Rf"],
        textposition="top right",
        textfont=dict(color=COLORS["emerald"], size=10, family="IBM Plex Mono"),
        marker=dict(color=COLORS["emerald"], size=10, symbol="diamond"),
    ))
    # Activos
    for ticker in TICKERS:
        col   = TICKER_COLORS.get(ticker, COLORS["gold"])
        label, _ = classify_beta(betas[ticker])
        fig.add_trace(go.Scatter(
            x=[betas[ticker]], y=[exp_rets[ticker] * 252],
            mode="markers+text", name=ticker,
            text=[ticker], textposition="top center",
            textfont=dict(color=col, size=10, family="IBM Plex Mono"),
            marker=dict(color=col, size=13,
                        line=dict(color="white", width=1.5)),
            hovertemplate=(f"<b>{ticker}</b><br>"
                           f"β: {betas[ticker]:.3f}<br>"
                           f"E[R]: {exp_rets[ticker]*252:.2%}<br>"
                           f"Clasif.: {label}<extra></extra>"),
        ))
    pb = plotly_base(380)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    pb.pop("hovermode", None)
    fig.update_layout(**pb,
        title=dict(text="Security Market Line (SML)",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Beta (β)",
        yaxis_title="Rendimiento esperado anual",
        yaxis_tickformat=".1%",
        hovermode="closest")
    return fig


def fig_risk_decomposition(betas, log_ret):
    ret_mkt  = log_ret[BENCHMARK]
    sist_pct = []
    idio_pct = []
    for ticker in TICKERS:
        ps, pi = decompose_risk(betas[ticker], log_ret[ticker], ret_mkt)
        sist_pct.append(ps)
        idio_pct.append(pi)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=TICKERS, y=sist_pct, name="Sistemático",
        marker_color=COLORS["rose"], opacity=0.85, marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        x=TICKERS, y=idio_pct, name="No Sistemático (Idiosincrático)",
        marker_color=COLORS["emerald"], opacity=0.85, marker_line_width=0,
    ))
    pb = plotly_base(300)
    pb["xaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text="Descomposición del Riesgo Total",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        barmode="stack", yaxis_title="% del Riesgo Total",
        yaxis_tickformat=".1f")
    return fig


# ── Layout ────────────────────────────────────────────────────

def sec_title(text, color=None):
    col = color or COLORS["gold"]
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                border-left:2px solid {col};padding-left:8px;">
        {text}
    </div>
    """, unsafe_allow_html=True)


def show():
    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">
                Módulo 04
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                CAPM & Beta
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
                    color:#8896A8;letter-spacing:0.08em;">
            Riesgo sistemático · Beta por MCO · Rendimiento esperado · Security Market Line
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Cargando datos y tasa libre de riesgo..."):
        import pandas as pd, numpy as np
        capm_data = get_capm(years=3)
        rf = {"display": capm_data["rf_display"], "annual": capm_data["rf_annual"],
              "daily": capm_data["rf_annual"]/252, "source": capm_data["rf_source"],
              "date": capm_data["rf_date"]}
        all_log = {}
        for t in TICKERS + [BENCHMARK]:
            d = get_precios(t, years=3)
            idx = pd.to_datetime([p["fecha"] for p in d["precios"]])
            closes = [p["close"] for p in d["precios"]]
            s = pd.Series(closes, index=idx)
            all_log[t] = np.log(s/s.shift(1)).dropna()
        log_ret = pd.DataFrame(all_log).dropna()

    ret_mkt  = log_ret[BENCHMARK]
    rm_daily = ret_mkt.mean()

    # Banner tasa libre de riesgo
    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                border-left:3px solid {COLORS['gold']};border-radius:6px;
                padding:0.9rem 1.2rem;margin-bottom:1.5rem;
                display:flex;align-items:center;gap:2rem;flex-wrap:wrap;">
        <div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.52rem;
                        color:#8896A8;letter-spacing:0.14em;text-transform:uppercase;">
                Tasa libre de riesgo (API)</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                        font-weight:700;color:#8B6914;">{rf['display']}</div>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#4A5568;">
            {rf['source']}
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#8896A8;">
            Actualizado: {rf['date']}
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#4A5568;">
            E[Rᵢ] = Rf + βᵢ · (E[Rm] − Rf)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Selector activo
    ticker = st.selectbox("Activo para análisis detallado", TICKERS, index=0)
    reg    = compute_beta(log_ret[ticker], ret_mkt)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── KPIs del activo seleccionado ──
    sec_title(f"① Beta y Métricas CAPM — {ticker}")
    label, col_beta = classify_beta(reg["beta"])
    er_ann = expected_return_capm(reg["beta"], rf["daily"], rm_daily) * 252
    ps, pi = decompose_risk(reg["beta"], log_ret[ticker], ret_mkt)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Beta (β)", f"{reg['beta']:.4f}", label)
    k2.metric("Alpha (α) anual", f"{reg['alpha']*252:.4f}")
    k3.metric("R²", f"{reg['r2']:.4f}")
    k4.metric("E[R] anual CAPM", f"{er_ann:.2%}", f"Rf={rf['display']}")
    k5.metric("Riesgo sistemático", f"{ps:.1f}%", f"Idiosincrático: {pi:.1f}%")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Dispersión individual + todos ──
    col_l, col_r = st.columns(2)
    with col_l:
        sec_title(f"② Dispersión {ticker} vs S&P 500 — Línea de Regresión", COLORS["sky"])
        st.plotly_chart(fig_scatter_single(reg, ticker), use_container_width=True)
    with col_r:
        sec_title("Todos los Activos vs S&P 500", COLORS["sky"])
        st.plotly_chart(fig_scatter_all(log_ret), use_container_width=True)

    with st.expander("Interpretación — Beta y regresión MCO"):
        st.markdown(f"""
        **β = {reg['beta']:.4f}** indica que cuando el S&P 500 sube 1%, {ticker} sube en promedio
        {reg['beta']:.2f}%. {'Activo **agresivo** — amplifica movimientos del mercado.' if reg['beta'] > 1.2
        else 'Activo **defensivo** — amortigua movimientos del mercado.' if reg['beta'] < 0.8
        else 'Activo **neutro** — se mueve en línea con el mercado.'}

        **R² = {reg['r2']:.4f}**: el {reg['r2']*100:.1f}% de la varianza de {ticker} es explicada
        por el mercado (riesgo sistemático). El restante {(1-reg['r2'])*100:.1f}% es riesgo
        idiosincrático, eliminable mediante diversificación.

        **Alpha (α) = {reg['alpha']*252:.4f}** anualizado: retorno no explicado por el CAPM.
        Alpha positivo → el activo supera la predicción del modelo.
        """)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Tabla resumen CAPM ──
    sec_title("③ Tabla Resumen CAPM — Todos los Activos", COLORS["emerald"])

    betas    = {}
    exp_rets = {}
    rows     = []
    for t in TICKERS:
        r   = compute_beta(log_ret[t], ret_mkt)
        betas[t]    = r["beta"]
        exp_rets[t] = expected_return_capm(r["beta"], rf["daily"], rm_daily)
        label_t, _  = classify_beta(r["beta"])
        ps_t, pi_t  = decompose_risk(r["beta"], log_ret[t], ret_mkt)
        rows.append({
            "Ticker"         : t,
            "Sector"         : SECTOR_MAP[t],
            "Beta (β)"       : round(r["beta"], 4),
            "Alpha anual"    : round(r["alpha"] * 252, 4),
            "R²"             : round(r["r2"], 4),
            "E[R] anual"     : f"{exp_rets[t]*252:.2%}",
            "Clasificación"  : label_t,
            "Riesgo Sist. %" : f"{ps_t:.1f}%",
            "Riesgo Idio. %" : f"{pi_t:.1f}%",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── SML + Descomposición ──
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        sec_title("④ Security Market Line (SML)", COLORS["violet"])
        st.plotly_chart(fig_sml(betas, exp_rets, rf["daily"], rm_daily),
                        use_container_width=True)
    with col_r2:
        sec_title("⑤ Descomposición del Riesgo Total", COLORS["rose"])
        st.plotly_chart(fig_risk_decomposition(betas, log_ret),
                        use_container_width=True)

    with st.expander("Interpretación — SML y riesgo sistemático vs idiosincrático"):
        st.markdown("""
        **Security Market Line (SML):** representa el rendimiento esperado según el CAPM
        para cada nivel de β. Activos **sobre** la SML están subvalorados (retorno mayor
        al predicho); activos **bajo** la SML están sobrevalorados.

        **Riesgo sistemático:** covarianza con el mercado, no diversificable. Es el único
        riesgo compensado con mayor retorno esperado según el CAPM.

        **Riesgo idiosincrático:** específico de la empresa. Se elimina al combinar activos
        con baja correlación — fundamento teórico de la diversificación de portafolios
        (Markowitz, Módulo 6).

        **Limitaciones del CAPM:** asume mercados eficientes, un solo factor de riesgo y
        distribuciones normales. En la práctica, el alfa de Jensen distinto de cero evidencia
        rendimientos no explicados. Modelos multifactor (Fama-French 3 factores) extienden
        este marco.
        """)