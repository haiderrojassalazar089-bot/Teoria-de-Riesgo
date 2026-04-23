"""
pages/m5_var.py — Módulo 5: VaR & CVaR
Streamlit + Plotly | yfinance 1.2.0
Métodos: Paramétrico · Histórico · Montecarlo (Normal + t-Student) · CVaR · Backtesting Kupiec
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.client import get_rendimientos, post_var
from utils.theme import plotly_base, COLORS
from utils.dynamic_tickers import get_tickers, get_ticker_colors, render_portafolio_badge


# ── Helpers ───────────────────────────────────────────────────

def sec_title(text, color=None):
    col = color or COLORS["gold"]
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                border-left:2px solid {col};padding-left:8px;">
        {text}
    </div>
    """, unsafe_allow_html=True)


# ── Estimación de grados de libertad t-Student ────────────────

def estimar_gl_tstudent(r):
    """
    Estima los grados de libertad (ν) de la distribución t-Student
    por máxima verosimilitud usando scipy.stats.t.fit().
    Retorna ν, loc y scale calibrados a los datos reales.
    ν pequeño (3-6): colas muy pesadas.
    ν grande (>30): aproxima a la normal.
    """
    # fit retorna (df, loc, scale)
    df_fit, loc_fit, scale_fit = stats.t.fit(r.values, floc=r.mean())
    # Clamp: mínimo 2.1 para que la varianza exista, máximo 50
    df_fit = np.clip(df_fit, 2.1, 50.0)
    return df_fit, loc_fit, scale_fit


# ── Cálculos VaR ──────────────────────────────────────────────

def var_parametrico(r, confianza=0.95):
    mu    = r.mean()
    sigma = r.std()
    z     = stats.norm.ppf(1 - confianza)
    var_d = -(mu + z * sigma)
    var_a = var_d * np.sqrt(252)
    cvar_d = -(mu - sigma * stats.norm.pdf(stats.norm.ppf(1 - confianza)) / (1 - confianza))
    return {"var_d": var_d, "var_a": var_a, "cvar_d": cvar_d, "mu": mu, "sigma": sigma}


def var_historico(r, confianza=0.95):
    var_d     = -np.percentile(r, (1 - confianza) * 100)
    var_a     = var_d * np.sqrt(252)
    threshold = -var_d
    cvar_d    = -r[r <= threshold].mean()
    return {"var_d": var_d, "var_a": var_a, "cvar_d": cvar_d}


def var_montecarlo(r, confianza=0.95, n_sim=10_000, horizonte=1, distribucion="Normal"):
    """
    Calcula VaR y CVaR por simulación Montecarlo.

    distribucion="Normal"   → N(μ, σ²) — supuesto clásico
    distribucion="t-Student" → t(ν, μ, σ) con ν estimado por MLE
                               Captura colas pesadas reales del activo.
    """
    mu    = r.mean()
    sigma = r.std()
    np.random.seed(42)

    if distribucion == "t-Student":
        nu, loc_t, scale_t = estimar_gl_tstudent(r)
        # Genera muestras de t escalada/desplazada al horizonte
        sims = stats.t.rvs(df=nu, loc=loc_t * horizonte,
                           scale=scale_t * np.sqrt(horizonte),
                           size=n_sim)
        meta = {"nu": nu, "loc": loc_t, "scale": scale_t}
    else:
        sims  = np.random.normal(mu * horizonte, sigma * np.sqrt(horizonte), n_sim)
        nu    = None
        meta  = {"nu": None}

    var_d  = -np.percentile(sims, (1 - confianza) * 100)
    var_a  = var_d * np.sqrt(252 / horizonte)
    cvar_d = -sims[sims <= -var_d].mean()

    return {
        "var_d" : var_d,
        "var_a" : var_a,
        "cvar_d": cvar_d,
        "sims"  : sims,
        "meta"  : meta,
    }


def kupiec_test(r, var_d, confianza):
    T      = len(r)
    violac = (r < -var_d).sum()
    p_obs  = violac / T
    p_esp  = 1 - confianza

    if violac == 0:
        return {"violaciones": 0, "p_obs": 0.0, "LR": 0.0, "p_valor": 1.0, "T": T}

    lr = -2 * (
        violac * np.log(p_esp / p_obs) +
        (T - violac) * np.log((1 - p_esp) / (1 - p_obs))
    )
    p_valor = 1 - stats.chi2.cdf(lr, df=1)
    return {"violaciones": violac, "p_obs": p_obs, "LR": lr, "p_valor": p_valor, "T": T}


# ── Gráficos ──────────────────────────────────────────────────

def fig_distribucion(r, var_95_hist, var_99_hist, cvar_95_hist, ticker):
    TICKER_COLORS = get_ticker_colors()
    col = TICKER_COLORS.get(ticker, COLORS["gold"])
    mu, sigma = r.mean(), r.std()
    x = np.linspace(r.min() - 0.005, r.max() + 0.005, 400)
    pdf = stats.norm.pdf(x, mu, sigma)
    pdf_scaled = pdf * len(r) * (r.max() - r.min()) / 60

    fig = go.Figure()

    mask_cvar = x <= -cvar_95_hist
    fig.add_trace(go.Scatter(
        x=x[mask_cvar], y=pdf_scaled[mask_cvar],
        fill="tozeroy", fillcolor="rgba(139,42,42,0.18)",
        line=dict(width=0), name="CVaR 95%", hoverinfo="skip",
    ))
    mask_var95 = (x <= -var_95_hist) & (x > -cvar_95_hist)
    fig.add_trace(go.Scatter(
        x=np.append(x[mask_var95], x[mask_var95][::-1]),
        y=np.append(pdf_scaled[mask_var95], np.zeros(mask_var95.sum())),
        fill="toself", fillcolor="rgba(139,42,42,0.10)",
        line=dict(width=0), name="Zona VaR 95%", hoverinfo="skip",
    ))
    fig.add_trace(go.Histogram(
        x=r.values, nbinsx=60, name="Rendimientos",
        marker_color=col, opacity=0.60, marker_line_width=0,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pdf_scaled, name="Normal teórica",
        line=dict(color=COLORS["text3"], width=1.4, dash="dash"),
    ))
    fig.add_vline(x=-var_95_hist,  line=dict(color=COLORS["rose"],   width=1.8, dash="dash"),
                  annotation_text="VaR 95%",
                  annotation_font=dict(color=COLORS["rose"], size=9))
    fig.add_vline(x=-var_99_hist,  line=dict(color="#5A1A1A",        width=1.8, dash="dot"),
                  annotation_text="VaR 99%",
                  annotation_font=dict(color="#5A1A1A", size=9))
    fig.add_vline(x=-cvar_95_hist, line=dict(color=COLORS["emerald"], width=1.6, dash="longdash"),
                  annotation_text="CVaR 95%",
                  annotation_font=dict(color=COLORS["emerald"], size=9))

    pb = plotly_base(380)
    pb["xaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text=f"{ticker}  ·  Distribución de Rendimientos con VaR y CVaR",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        bargap=0.04, showlegend=True)
    return fig


def fig_montecarlo_comparado(sims_norm, sims_t, var_norm, var_t,
                              cvar_norm, cvar_t, confianza, ticker, nu):
    """
    Muestra en dos paneles lado a lado la distribución Normal vs t-Student,
    con sus líneas de VaR y CVaR para comparación directa.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Normal  ·  VaR={var_norm*100:.3f}%  CVaR={cvar_norm*100:.3f}%",
            f"t-Student (ν={nu:.1f})  ·  VaR={var_t*100:.3f}%  CVaR={cvar_t*100:.3f}%",
        ],
        horizontal_spacing=0.08,
    )

    # ── Panel Normal ──
    colors_n = [COLORS["rose"] if s < -var_norm else COLORS["gold"] for s in sims_norm]
    fig.add_trace(go.Histogram(
        x=sims_norm, nbinsx=80,
        marker_color=colors_n, opacity=0.70,
        marker_line_width=0, name="Normal",
    ), row=1, col=1)
    fig.add_vline(x=-var_norm,
                  line=dict(color=COLORS["rose"], width=2, dash="dash"),
                  annotation_text=f"VaR {int(confianza*100)}%",
                  annotation_font=dict(color=COLORS["rose"], size=9),
                  row=1, col=1)
    fig.add_vline(x=-cvar_norm,
                  line=dict(color=COLORS["emerald"], width=1.8, dash="longdash"),
                  annotation_text=f"CVaR {int(confianza*100)}%",
                  annotation_font=dict(color=COLORS["emerald"], size=9),
                  row=1, col=1)

    # ── Panel t-Student ──
    colors_t = [COLORS["rose"] if s < -var_t else COLORS["sky"] for s in sims_t]
    fig.add_trace(go.Histogram(
        x=sims_t, nbinsx=80,
        marker_color=colors_t, opacity=0.70,
        marker_line_width=0, name="t-Student",
    ), row=1, col=2)
    fig.add_vline(x=-var_t,
                  line=dict(color=COLORS["rose"], width=2, dash="dash"),
                  annotation_text=f"VaR {int(confianza*100)}%",
                  annotation_font=dict(color=COLORS["rose"], size=9),
                  row=1, col=2)
    fig.add_vline(x=-cvar_t,
                  line=dict(color=COLORS["emerald"], width=1.8, dash="longdash"),
                  annotation_text=f"CVaR {int(confianza*100)}%",
                  annotation_font=dict(color=COLORS["emerald"], size=9),
                  row=1, col=2)

    pb = plotly_base(360)
    fig.update_layout(**pb,
        title=dict(
            text=f"{ticker}  ·  Montecarlo: Normal vs t-Student ({int(len(sims_norm)):,} simulaciones)",
            font=dict(size=12, color=COLORS["text"], family="Playfair Display"),
        ),
        showlegend=False,
    )
    for c in [1, 2]:
        fig.update_xaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9),
                         type="-", row=1, col=c)
        fig.update_yaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9),
                         row=1, col=c)
    return fig


def fig_colas_comparadas(r, mc_norm, mc_t, confianza, ticker):
    """
    Zoom en la cola izquierda: superpone los histogramas de Normal y t-Student
    junto con los datos reales, para ver cuál distribución aproxima mejor
    los eventos extremos del activo.
    """
    # Rango de la cola: solo rendimientos menores al VaR normal
    cola_lim = -mc_norm["var_d"] * 3.5
    x_range  = [cola_lim, -mc_norm["var_d"] * 0.5]

    fig = go.Figure()

    # Datos reales (cola)
    r_cola = r[r < -mc_norm["var_d"] * 0.8]
    fig.add_trace(go.Histogram(
        x=r_cola.values, nbinsx=25,
        name="Datos reales (cola)",
        marker_color=COLORS["gold"], opacity=0.65,
        marker_line_width=0,
    ))

    # Curva Normal teórica escalada
    mu, sigma = r.mean(), r.std()
    x_plot = np.linspace(cola_lim, 0, 300)
    pdf_n  = stats.norm.pdf(x_plot, mu, sigma)
    pdf_n  = pdf_n * len(r) * (r.max() - r.min()) / 60
    fig.add_trace(go.Scatter(
        x=x_plot, y=pdf_n, name="Normal teórica",
        line=dict(color=COLORS["text3"], width=2, dash="dash"),
    ))

    # Curva t-Student escalada
    nu, loc_t, scale_t = mc_t["meta"]["nu"], mc_t["meta"]["loc"], mc_t["meta"]["scale"]
    pdf_t = stats.t.pdf(x_plot, df=nu, loc=loc_t, scale=scale_t)
    pdf_t = pdf_t * len(r) * (r.max() - r.min()) / 60
    fig.add_trace(go.Scatter(
        x=x_plot, y=pdf_t, name=f"t-Student (ν={nu:.1f})",
        line=dict(color=COLORS["sky"], width=2),
    ))

    # Líneas VaR
    fig.add_vline(x=-mc_norm["var_d"],
                  line=dict(color=COLORS["gold"], width=1.4, dash="dot"),
                  annotation_text="VaR Normal",
                  annotation_font=dict(color=COLORS["gold"], size=9))
    fig.add_vline(x=-mc_t["var_d"],
                  line=dict(color=COLORS["sky"], width=1.4, dash="dot"),
                  annotation_text="VaR t-Student",
                  annotation_font=dict(color=COLORS["sky"], size=9))

    pb = plotly_base(300)
    pb["xaxis"]["type"]  = "-"
    pb["xaxis"]["range"] = x_range
    fig.update_layout(**pb,
        title=dict(
            text=f"{ticker}  ·  Zoom Cola Izquierda — Normal vs t-Student vs Datos Reales",
            font=dict(size=12, color=COLORS["text"], family="Playfair Display"),
        ),
    )
    return fig


def fig_backtesting(r, var_d, confianza, ticker):
    TICKER_COLORS = get_ticker_colors()
    violaciones = r < -var_d
    colors_bar  = [COLORS["rose"] if v else TICKER_COLORS.get(ticker, COLORS["gold"])
                   for v in violaciones]

    fig = go.Figure()
    fig.add_hline(y=-var_d,
                  line=dict(color=COLORS["rose"], width=1.5, dash="dash"),
                  annotation_text=f"−VaR {int(confianza*100)}%",
                  annotation_font=dict(color=COLORS["rose"], size=9))
    fig.add_trace(go.Bar(
        x=r.index, y=r.values,
        marker_color=colors_bar, marker_line_width=0,
        opacity=0.7, name="Rendimiento diario",
    ))
    fig.update_layout(**plotly_base(300),
        title=dict(
            text=f"{ticker}  ·  Backtesting — Violaciones del VaR {int(confianza*100)}% (rojo)",
            font=dict(size=12, color=COLORS["text"], family="Playfair Display"),
        ))
    return fig


# ── Layout ────────────────────────────────────────────────────

def show():
    render_portafolio_badge()

    TICKERS = get_tickers()

    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">
                Módulo 05
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                VaR & CVaR
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
                    color:#8896A8;letter-spacing:0.08em;">
            Valor en Riesgo · Paramétrico · Histórico · Montecarlo (Normal & t-Student) · Expected Shortfall · Backtesting
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Carga de datos ──
    with st.spinner("Cargando datos..."):
        all_log = {}
        for t in TICKERS:
            d   = get_rendimientos(t, years=3)
            idx = pd.to_datetime(d["fechas"])
            all_log[t] = pd.Series(d["log_returns"], index=idx)
        log_ret = pd.DataFrame(all_log).dropna()

    # ── Controles ──
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        ticker = st.selectbox("Activo", TICKERS, index=0)
    with c2:
        confianza = st.selectbox("Nivel de confianza", [0.95, 0.99], index=0,
                                  format_func=lambda x: f"{int(x*100)}%")
    with c3:
        n_sim = st.selectbox("Simulaciones MC", [10_000, 50_000, 100_000], index=0,
                              format_func=lambda x: f"{x:,}")
    with c4:
        monto = st.number_input("Monto del portafolio (USD)", value=1_000_000,
                                 step=100_000, format="%d")

    r = log_ret[ticker]

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Cálculos base ──
    p95  = var_parametrico(r, 0.95)
    p99  = var_parametrico(r, 0.99)
    h95  = var_historico(r, 0.95)
    h99  = var_historico(r, 0.99)

    # Montecarlo Normal (para comparación)
    mc_norm     = var_montecarlo(r, confianza, n_sim, distribucion="Normal")
    mc_norm_99  = var_montecarlo(r, 0.99,      n_sim, distribucion="Normal")

    # Montecarlo t-Student — estimación MLE de ν
    mc_t        = var_montecarlo(r, confianza, n_sim, distribucion="t-Student")
    mc_t_99     = var_montecarlo(r, 0.99,      n_sim, distribucion="t-Student")

    nu_estimado = mc_t["meta"]["nu"]

    var_sel  = p95  if confianza == 0.95 else p99
    hist_sel = h95  if confianza == 0.95 else h99

    # ── ① KPIs ──
    sec_title(f"① Resumen VaR — {ticker} · Confianza {int(confianza*100)}%")

    # Banner informativo sobre ν
    st.markdown(
        f'<div style="background:#EEF4FB;border:1px solid #2266A044;'
        f'border-left:3px solid #2266A0;border-radius:6px;'
        f'padding:0.6rem 1.1rem;margin-bottom:1rem;'
        f"font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#1A2035;\">"
        f"Grados de libertad t-Student estimados por MLE para <b>{ticker}</b>: "
        f"<b>ν = {nu_estimado:.2f}</b> &nbsp;·&nbsp; "
        f"{'Colas muy pesadas — distribución normal inadecuada' if nu_estimado < 8 else 'Colas moderadas — distribución normal es aproximación aceptable' if nu_estimado < 20 else 'Colas ligeras — distribución converge a normal'}"
        f"</div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("VaR Paramétrico (diario)", f"{var_sel['var_d']*100:.3f}%",
              f"USD {var_sel['var_d']*monto:,.0f}")
    k2.metric("VaR Histórico (diario)",   f"{hist_sel['var_d']*100:.3f}%",
              f"USD {hist_sel['var_d']*monto:,.0f}")
    k3.metric("VaR MC Normal (diario)",   f"{mc_norm['var_d']*100:.3f}%",
              f"USD {mc_norm['var_d']*monto:,.0f}")
    k4.metric(f"VaR MC t-Student ν={nu_estimado:.1f}", f"{mc_t['var_d']*100:.3f}%",
              f"USD {mc_t['var_d']*monto:,.0f}")
    k5.metric("CVaR / ES (histórico)",    f"{hist_sel['cvar_d']*100:.3f}%",
              f"USD {hist_sel['cvar_d']*monto:,.0f}")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── ② Distribución histórica ──
    sec_title("② Distribución de Rendimientos con Líneas de VaR y CVaR", COLORS["sky"])
    st.plotly_chart(
        fig_distribucion(r, h95["var_d"], h99["var_d"], h95["cvar_d"], ticker),
        use_container_width=True,
    )

    with st.expander("Interpretación — Distribución y VaR"):
        st.markdown(f"""
El **VaR al {int(confianza*100)}%** indica que, con {int(confianza*100)}% de probabilidad,
la pérdida diaria no superará **{var_sel['var_d']*100:.3f}%** del portafolio.
Para un monto de USD {monto:,}, equivale a **USD {var_sel['var_d']*monto:,.0f}**.

El **CVaR (Expected Shortfall)** promedia las pérdidas del peor {100-int(confianza*100)}% de días.
Es una medida coherente de riesgo (Artzner et al., 1999) y el estándar de Basilea III (FRTB).
La zona roja en el histograma representa esa cola de pérdidas extremas.
        """)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── ③ Montecarlo comparado ──
    sec_title("③ Simulación Montecarlo — Normal vs t-Student", COLORS["gold"])

    st.plotly_chart(
        fig_montecarlo_comparado(
            mc_norm["sims"], mc_t["sims"],
            mc_norm["var_d"], mc_t["var_d"],
            mc_norm["cvar_d"], mc_t["cvar_d"],
            confianza, ticker, nu_estimado,
        ),
        use_container_width=True,
    )

    # ── Diferencia entre métodos ──
    diff_var  = (mc_t["var_d"]  - mc_norm["var_d"])  * 100
    diff_cvar = (mc_t["cvar_d"] - mc_norm["cvar_d"]) * 100
    diff_col  = COLORS["rose"] if diff_var > 0 else COLORS["emerald"]

    st.markdown(
        f'<div style="background:#FFFFFF;border:1px solid #D8DDE8;'
        f'border-left:3px solid {diff_col};border-radius:6px;'
        f'padding:0.85rem 1.2rem;margin-top:0.4rem;">'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#8896A8;">'
        f'Impacto de usar t-Student vs Normal &nbsp;·&nbsp; '
        f'VaR: <b style="color:{diff_col};">{diff_var:+.3f}%</b> &nbsp;|&nbsp; '
        f'CVaR: <b style="color:{diff_col};">{diff_cvar:+.3f}%</b> &nbsp;|&nbsp; '
        f'En USD: <b style="color:{diff_col};">USD {diff_var/100*monto:+,.0f}</b>'
        f'</span>'
        f'<div style="font-family:\'Inter\',sans-serif;font-size:0.77rem;color:#4A5568;margin-top:6px;">'
        f'{"La t-Student captura colas más pesadas que la normal → VaR y CVaR mayores → estimación más conservadora y realista." if diff_var > 0 else "En este período la normal y la t-Student convergen → ν alto → colas no son significativamente más pesadas que la normal."}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── ④ Zoom cola izquierda ──
    sec_title("④ Zoom Cola Izquierda — ¿Qué distribución ajusta mejor los extremos?", COLORS["violet"])
    st.plotly_chart(
        fig_colas_comparadas(r, mc_norm, mc_t, confianza, ticker),
        use_container_width=True,
    )

    with st.expander("¿Por qué la t-Student es más apropiada para datos financieros?"):
        st.markdown(f"""
**El problema de la distribución Normal:**
La distribución normal asume que los eventos extremos son prácticamente imposibles.
Bajo normalidad, un movimiento de 4σ debería ocurrir una vez cada 32 años.
En realidad, en los mercados financieros ocurre varias veces por año.

**La distribución t-Student:**
La t-Student tiene el mismo centro y simetría que la normal, pero sus colas son más gruesas.
El parámetro ν (grados de libertad) controla qué tan pesadas son esas colas:

| ν | Interpretación |
|---|---------------|
| 2–4 | Colas extremadamente pesadas — mercados en crisis |
| 4–8 | Colas pesadas — típico de acciones individuales |
| 8–15 | Colas moderadas — índices diversificados |
| >30 | Converge a la normal — poco beneficio sobre supuesto normal |

**Para {ticker}, ν estimado = {nu_estimado:.2f}:**
{'Las colas son significativamente más pesadas que la normal. El VaR t-Student es más conservador y por tanto más honesto sobre el riesgo real del activo.' if nu_estimado < 10 else 'Las colas son moderadamente pesadas. La t-Student sigue siendo preferible a la normal, aunque la diferencia en VaR no es dramática para este período de muestra.'}

**Estimación de ν:**
Se usa Máxima Verosimilitud (MLE) mediante `scipy.stats.t.fit()`, que encuentra los valores
de ν, μ y σ que maximizan la probabilidad de observar los rendimientos históricos reales.
Este método es el estándar acadénmico y regulatorio para calibrar distribuciones de cola.
        """)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── ⑤ Tabla comparativa ──
    sec_title("⑤ Tabla Comparativa de Métodos VaR", COLORS["emerald"])

    rows_comp = []
    for conf_label, conf_val in [("95%", 0.95), ("99%", 0.99)]:
        pm  = var_parametrico(r, conf_val)
        hm  = var_historico(r, conf_val)
        mn  = var_montecarlo(r, conf_val, n_sim, distribucion="Normal")
        mt  = var_montecarlo(r, conf_val, n_sim, distribucion="t-Student")
        rows_comp.append({
            "Confianza"          : conf_label,
            "VaR Param. Diario"  : f"{pm['var_d']*100:.4f}%",
            "VaR Param. Anual"   : f"{pm['var_a']*100:.2f}%",
            "VaR Histór. Diario" : f"{hm['var_d']*100:.4f}%",
            "VaR MC Normal"      : f"{mn['var_d']*100:.4f}%",
            f"VaR MC t(ν={nu_estimado:.1f})": f"{mt['var_d']*100:.4f}%",
            "CVaR Histórico"     : f"{hm['cvar_d']*100:.4f}%",
            "CVaR t-Student"     : f"{mt['cvar_d']*100:.4f}%",
            f"CVaR USD"          : f"${mt['cvar_d']*monto:,.0f}",
        })

    st.dataframe(pd.DataFrame(rows_comp), use_container_width=True, hide_index=True)

    with st.expander("Comparación de métodos — Ventajas y limitaciones"):
        st.markdown("""
| Método | Ventajas | Limitaciones |
|--------|----------|--------------|
| **Paramétrico** | Analítico, rápido, extensible a portafolios | Asume normalidad → subestima colas pesadas |
| **Histórico** | Sin supuestos distribucionales, captura eventos reales | Limitado al período de muestra; no extrapola |
| **MC Normal** | Flexible, repetible, ampliable | Hereda el supuesto de normalidad |
| **MC t-Student** | Captura colas pesadas; ν calibrado a datos reales | Más costoso; asume estacionariedad de ν |
| **CVaR/ES** | Coherente (sub-aditivo), captura severidad de cola | Menos intuitivo; sensible a outliers extremos |

**Nota regulatoria:** Basilea III (FRTB) migró del VaR 99% al ES 97.5%.
La t-Student con ν calibrado es el estándar en modelos internos de bancos (IMM approach).
        """)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── ⑥ Backtesting Kupiec ──
    sec_title("⑥ Backtesting — Test de Kupiec (POF)", COLORS["rose"])
    st.plotly_chart(
        fig_backtesting(r, var_sel["var_d"], confianza, ticker),
        use_container_width=True,
    )

    kup      = kupiec_test(r, var_sel["var_d"], confianza)
    rechaza  = kup["p_valor"] < 0.05
    col_kup  = COLORS["rose"] if rechaza else COLORS["emerald"]
    tasa_esp = (1 - confianza) * 100

    c1_k, c2_k, c3_k = st.columns(3)
    with c1_k:
        st.metric("Violaciones observadas", kup["violaciones"],
                  f"de {kup['T']} días")
    with c2_k:
        st.metric("Tasa de fallo observada", f"{kup['p_obs']*100:.2f}%",
                  f"Esperada: {tasa_esp:.1f}%")
    with c3_k:
        st.metric("p-valor Kupiec", f"{kup['p_valor']:.4f}",
                  "Rechaza H₀" if rechaza else "No rechaza H₀")

    st.markdown(
        f'<div style="background:#FFFFFF;border:1px solid {col_kup}44;'
        f'border-left:3px solid {col_kup};border-radius:6px;'
        f'padding:0.9rem 1.2rem;margin-top:0.8rem;">'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#8896A8;">'
        f'Test de Kupiec &nbsp;·&nbsp; LR = {kup["LR"]:.4f} &nbsp;·&nbsp; p = {kup["p_valor"]:.4f}'
        f'</span>'
        f'<span style="font-family:\'Inter\',sans-serif;font-size:0.8rem;font-weight:600;'
        f'color:{col_kup};margin-left:1.5rem;">'
        f'{"⛔ El modelo VaR subestima el riesgo real" if rechaza else "✅ El modelo VaR es estadísticamente válido"}'
        f'</span>'
        f'<div style="font-family:\'Inter\',sans-serif;font-size:0.78rem;color:#4A5568;'
        f'margin-top:0.5rem;line-height:1.6;">'
        f'H₀: la tasa de violaciones = {tasa_esp:.1f}% (nivel {int(confianza*100)}%). '
        f'{"Se rechaza H₀ a α=5% → el VaR no captura adecuadamente el riesgo de cola." if rechaza else "No se rechaza H₀ a α=5% → el modelo está correctamente calibrado."}'
        f'</div></div>',
        unsafe_allow_html=True,
    )