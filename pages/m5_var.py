"""
pages/m5_var.py — Módulo 5: VaR & CVaR
Streamlit + Plotly | yfinance 1.2.0
Métodos: Paramétrico · Histórico · Montecarlo · CVaR · Backtesting Kupiec
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loader import get_prices, get_returns, TICKERS, TICKER_COLORS
from utils.theme import plotly_base, COLORS


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
    var_d = -np.percentile(r, (1 - confianza) * 100)
    var_a = var_d * np.sqrt(252)
    threshold = -var_d
    cvar_d = -r[r <= threshold].mean()
    return {"var_d": var_d, "var_a": var_a, "cvar_d": cvar_d}


def var_montecarlo(r, confianza=0.95, n_sim=10_000, horizonte=1):
    mu    = r.mean()
    sigma = r.std()
    np.random.seed(42)
    sims  = np.random.normal(mu * horizonte, sigma * np.sqrt(horizonte), n_sim)
    var_d = -np.percentile(sims, (1 - confianza) * 100)
    var_a = var_d * np.sqrt(252 / horizonte)
    cvar_d = -sims[sims <= -var_d].mean()
    return {"var_d": var_d, "var_a": var_a, "cvar_d": cvar_d, "sims": sims}


def kupiec_test(r, var_d, confianza):
    """
    Test de Kupiec (POF - Proportion of Failures).
    H0: la tasa de violaciones observada = (1 - confianza).
    """
    T         = len(r)
    violac    = (r < -var_d).sum()
    p_obs     = violac / T
    p_esp     = 1 - confianza

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
    col = TICKER_COLORS.get(ticker, COLORS["gold"])
    mu, sigma = r.mean(), r.std()
    x = np.linspace(r.min() - 0.005, r.max() + 0.005, 400)
    pdf = stats.norm.pdf(x, mu, sigma)
    pdf_scaled = pdf * len(r) * (r.max() - r.min()) / 60

    fig = go.Figure()

    # Área pérdidas extremas (cola izq CVaR)
    mask_cvar = x <= -cvar_95_hist
    fig.add_trace(go.Scatter(
        x=x[mask_cvar], y=pdf_scaled[mask_cvar],
        fill="tozeroy", fillcolor="rgba(139,42,42,0.18)",
        line=dict(width=0), name="CVaR 95%", hoverinfo="skip",
    ))
    # Área VaR 95%
    mask_var95 = (x <= -var_95_hist) & (x > -cvar_95_hist)
    fig.add_trace(go.Scatter(
        x=np.append(x[mask_var95], x[mask_var95][::-1]),
        y=np.append(pdf_scaled[mask_var95], np.zeros(mask_var95.sum())),
        fill="toself", fillcolor="rgba(139,42,42,0.10)",
        line=dict(width=0), name="Zona VaR 95%", hoverinfo="skip",
    ))

    # Histograma
    fig.add_trace(go.Histogram(
        x=r.values, nbinsx=60, name="Rendimientos",
        marker_color=col, opacity=0.60, marker_line_width=0,
    ))
    # Normal
    fig.add_trace(go.Scatter(
        x=x, y=pdf_scaled, name="Normal teórica",
        line=dict(color=COLORS["text3"], width=1.4, dash="dash"),
    ))
    # Líneas VaR
    fig.add_vline(x=-var_95_hist, line=dict(color=COLORS["rose"], width=1.8, dash="dash"),
                  annotation_text="VaR 95%",
                  annotation_font=dict(color=COLORS["rose"], size=9))
    fig.add_vline(x=-var_99_hist, line=dict(color="#5A1A1A", width=1.8, dash="dot"),
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


def fig_montecarlo(sims_95, var_mc, cvar_mc, confianza, ticker):
    fig = go.Figure()
    colors = [COLORS["rose"] if s < -var_mc else COLORS["gold"] for s in sims_95]
    fig.add_trace(go.Histogram(
        x=sims_95, nbinsx=80, name="Simulaciones MC",
        marker_color=colors, opacity=0.70, marker_line_width=0,
    ))
    fig.add_vline(x=-var_mc, line=dict(color=COLORS["rose"], width=2, dash="dash"),
                  annotation_text=f"VaR {int(confianza*100)}%",
                  annotation_font=dict(color=COLORS["rose"], size=9))
    fig.add_vline(x=-cvar_mc, line=dict(color=COLORS["emerald"], width=1.8, dash="longdash"),
                  annotation_text=f"CVaR {int(confianza*100)}%",
                  annotation_font=dict(color=COLORS["emerald"], size=9))
    pb = plotly_base(320)
    pb["xaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text=f"{ticker}  ·  Distribución Montecarlo (10,000 simulaciones)",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        bargap=0.03, showlegend=False)
    return fig


def fig_backtesting(r, var_d, confianza, ticker):
    violaciones = r < -var_d
    colors_bar  = [COLORS["rose"] if v else TICKER_COLORS.get(ticker, COLORS["gold"])
                   for v in violaciones]
    opacities   = [0.9 if v else 0.45 for v in violaciones]

    fig = go.Figure()
    fig.add_hline(y=-var_d, line=dict(color=COLORS["rose"], width=1.5, dash="dash"),
                  annotation_text=f"−VaR {int(confianza*100)}%",
                  annotation_font=dict(color=COLORS["rose"], size=9))
    fig.add_trace(go.Bar(
        x=r.index, y=r.values,
        marker_color=colors_bar, marker_line_width=0,
        opacity=0.7, name="Rendimiento diario",
    ))
    fig.update_layout(**plotly_base(300),
        title=dict(text=f"{ticker}  ·  Backtesting — Violaciones del VaR {int(confianza*100)}% (rojo)",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")))
    return fig


# ── Layout ────────────────────────────────────────────────────

def show():
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
            Valor en Riesgo · Paramétrico · Histórico · Montecarlo · Expected Shortfall · Backtesting
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Cargando datos..."):
        prices  = get_prices(years=3)
        log_ret = get_returns(prices[TICKERS], log=True)

    # Controles
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

    # ── Cálculos ──
    p95 = var_parametrico(r, 0.95)
    p99 = var_parametrico(r, 0.99)
    h95 = var_historico(r, 0.95)
    h99 = var_historico(r, 0.99)
    mc  = var_montecarlo(r, confianza, n_sim)
    mc99 = var_montecarlo(r, 0.99, n_sim)

    var_sel  = var_parametrico(r, confianza) if confianza == 0.95 else p99
    hist_sel = h95 if confianza == 0.95 else h99

    # ── KPIs ──
    sec_title(f"① Resumen VaR — {ticker} · Confianza {int(confianza*100)}%")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("VaR Paramétrico (diario)", f"{var_sel['var_d']*100:.3f}%",
              f"USD {var_sel['var_d']*monto:,.0f}")
    k2.metric("VaR Histórico (diario)",   f"{hist_sel['var_d']*100:.3f}%",
              f"USD {hist_sel['var_d']*monto:,.0f}")
    k3.metric("VaR Montecarlo (diario)",  f"{mc['var_d']*100:.3f}%",
              f"USD {mc['var_d']*monto:,.0f}")
    k4.metric("CVaR / ES (histórico)",    f"{hist_sel['cvar_d']*100:.3f}%",
              f"USD {hist_sel['cvar_d']*monto:,.0f}")
    k5.metric("VaR Anualizado (param.)",  f"{var_sel['var_a']*100:.2f}%",
              f"×√252")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Distribución ──
    sec_title("② Distribución de Rendimientos con Líneas de VaR y CVaR", COLORS["sky"])
    st.plotly_chart(
        fig_distribucion(r, h95["var_d"], h99["var_d"], h95["cvar_d"], ticker),
        use_container_width=True
    )

    with st.expander("Interpretación — Distribución y VaR"):
        st.markdown(f"""
        El **VaR al {int(confianza*100)}%** indica que, con {int(confianza*100)}% de probabilidad,
        la pérdida diaria no superará **{var_sel['var_d']*100:.3f}%** del portafolio.
        Para un monto de USD {monto:,}, esto equivale a **USD {var_sel['var_d']*monto:,.0f}**.

        El **CVaR (Expected Shortfall)** va más allá: mide la pérdida *promedio esperada*
        dado que se supera el VaR. Es una medida coherente de riesgo (sub-aditividad) y
        preferida en Basilea III y FRTB. La zona roja en el histograma representa esta cola.
        """)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Montecarlo ──
    sec_title("③ Simulación Montecarlo", COLORS["gold"])
    st.plotly_chart(fig_montecarlo(mc["sims"], mc["var_d"], mc["cvar_d"], confianza, ticker),
                    use_container_width=True)

    with st.expander("Metodología — Simulación Montecarlo"):
        st.markdown(f"""
        Se generaron **{n_sim:,} escenarios** a partir de una distribución normal con
        μ = {r.mean():.6f} y σ = {r.std():.6f} (parámetros históricos del activo).

        **Ventajas frente al paramétrico:** captura mejor la forma de la distribución
        cuando se utilizan distribuciones no normales (t-Student, skewed-t).
        **Limitación:** asume i.i.d.; no modela directamente la heterocedasticidad.
        Para VaR dinámico, combinar con GARCH (Módulo 3) mejora significativamente.
        """)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Tabla Comparativa ──
    sec_title("④ Tabla Comparativa de Métodos VaR", COLORS["emerald"])

    rows_comp = []
    for conf_label, conf_val in [("95%", 0.95), ("99%", 0.99)]:
        pm = var_parametrico(r, conf_val)
        hm = var_historico(r, conf_val)
        mm = var_montecarlo(r, conf_val, n_sim)
        rows_comp.append({
            "Confianza"           : conf_label,
            "VaR Param. Diario"   : f"{pm['var_d']*100:.4f}%",
            "VaR Param. Anual"    : f"{pm['var_a']*100:.2f}%",
            "VaR Histór. Diario"  : f"{hm['var_d']*100:.4f}%",
            "VaR Histór. Anual"   : f"{hm['var_a']*100:.2f}%",
            "VaR MC Diario"       : f"{mm['var_d']*100:.4f}%",
            "CVaR (Histórico)"    : f"{hm['cvar_d']*100:.4f}%",
            f"CVaR USD ({conf_label})": f"${hm['cvar_d']*monto:,.0f}",
        })

    st.dataframe(pd.DataFrame(rows_comp), use_container_width=True, hide_index=True)

    with st.expander("Comparación de métodos — Ventajas y limitaciones"):
        st.markdown("""
        | Método | Ventajas | Limitaciones |
        |--------|----------|--------------|
        | **Paramétrico** | Simple, analítico, rápido | Asume normalidad → subestima colas |
        | **Histórico** | Sin supuestos distribucionales | Depende del período de muestra |
        | **Montecarlo** | Flexible, permite cualquier distribución | Costoso, sensible a parámetros |
        | **CVaR/ES** | Coherente, captura cola completa | Menos intuitivo que VaR |

        **Nota regulatoria:** Basilea III (FRTB) migró del VaR al ES al 97.5%, equivalente
        al CVaR al 97.5%. El ES es preferido por ser una medida de riesgo coherente.
        """)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Backtesting Kupiec ──
    sec_title("⑤ Backtesting — Test de Kupiec (POF)", COLORS["rose"])
    st.plotly_chart(fig_backtesting(r, var_sel["var_d"], confianza, ticker),
                    use_container_width=True)

    kup = kupiec_test(r, var_sel["var_d"], confianza)
    rechaza = kup["p_valor"] < 0.05
    col_kup = COLORS["rose"] if rechaza else COLORS["emerald"]
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

    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid {col_kup}44;
                border-left:3px solid {col_kup};border-radius:6px;
                padding:0.9rem 1.2rem;margin-top:0.8rem;">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#8896A8;">
            Test de Kupiec &nbsp;·&nbsp; LR = {kup['LR']:.4f} &nbsp;·&nbsp; p = {kup['p_valor']:.4f}
        </span>
        <span style="font-family:'Inter',sans-serif;font-size:0.8rem;font-weight:600;
                     color:{col_kup};margin-left:1.5rem;">
            {'⛔ El modelo VaR subestima el riesgo real' if rechaza
             else '✅ El modelo VaR es estadísticamente válido'}
        </span>
        <div style="font-family:'Inter',sans-serif;font-size:0.78rem;color:#4A5568;
                    margin-top:0.5rem;line-height:1.6;">
            H₀: la tasa de violaciones = {tasa_esp:.1f}% (nivel {int(confianza*100)}%).
            {'Se rechaza H₀ a α=5% → el VaR no captura adecuadamente el riesgo de cola.'
             if rechaza else 'No se rechaza H₀ a α=5% → el modelo es calibrado correctamente.'}
        </div>
    </div>
    """, unsafe_allow_html=True)
