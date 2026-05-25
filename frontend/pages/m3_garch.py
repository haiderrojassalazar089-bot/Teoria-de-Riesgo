"""
pages/m3_garch.py — Módulo 3: ARCH / GARCH / EWMA
Streamlit + arch library | yfinance 1.2.0
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.client import get_rendimientos
from utils.theme import plotly_base, COLORS
from utils.dynamic_tickers import get_tickers, get_ticker_colors, render_portafolio_badge

MODELS = {
    "ARCH(1)"    : dict(vol="ARCH",  p=1, q=0, o=0),
    "GARCH(1,1)" : dict(vol="GARCH", p=1, q=1, o=0),
    "EGARCH(1,1)": dict(vol="EGARCH",p=1, q=1, o=0),
}

MODEL_COLORS = {
    "ARCH(1)"    : COLORS["sky"],
    "GARCH(1,1)" : COLORS["gold"],
    "EGARCH(1,1)": COLORS["violet"],
}


# ── Helpers de UI ─────────────────────────────────────────────

def sec_title(text, color=None):
    col = color or COLORS["gold"]
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                border-left:2px solid {col};padding-left:8px;">
        {text}
    </div>
    """, unsafe_allow_html=True)


# ── EWMA ─────────────────────────────────────────────────────

def ewma_volatility(returns: pd.Series, lam: float) -> pd.Series:
    """
    Calcula la volatilidad EWMA (RiskMetrics) de forma recursiva.
    σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
    Inicialización: varianza muestral de los primeros 30 días.
    """
    r = returns.values
    n = len(r)
    var = np.empty(n)
    var[0] = np.var(r[:30]) if n >= 30 else np.var(r)
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return pd.Series(np.sqrt(var), index=returns.index)


def fig_ewma(returns_pct: pd.Series, lam1: float, lam2: float,
             roll_w: int, ticker: str) -> go.Figure:
    """Gráfico comparativo: EWMA (dos lambdas) vs volatilidad rodante."""
    ewma1  = ewma_volatility(returns_pct, lam1)
    ewma2  = ewma_volatility(returns_pct, lam2)
    roll   = returns_pct.rolling(roll_w).std()

    fig = go.Figure()

    # Área de volatilidad rodante (fondo)
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll.values,
        name=f"Rodante {roll_w}d",
        line=dict(color=COLORS["text3"], width=1, dash="dot"),
        opacity=0.6,
    ))
    # EWMA λ₁ (RiskMetrics si λ=0.94)
    fig.add_trace(go.Scatter(
        x=ewma1.index, y=ewma1.values,
        name=f"EWMA λ={lam1}",
        line=dict(color=COLORS["gold"], width=2),
    ))
    # EWMA λ₂ (configurable)
    fig.add_trace(go.Scatter(
        x=ewma2.index, y=ewma2.values,
        name=f"EWMA λ={lam2}",
        line=dict(color=COLORS["sky"], width=2, dash="dash"),
    ))

    pb = plotly_base(380)
    pb.pop("legend", None)
    fig.update_layout(
        **pb,
        title=dict(
            text=f"{ticker}  ·  EWMA vs Volatilidad Muestral Rodante",
            font=dict(size=12, color=COLORS["text"], family="Playfair Display"),
        ),
        yaxis_title="σ (% diario)",
        legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
    )
    fig.update_yaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9))
    fig.update_xaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9))
    return fig


def fig_ewma_decay(lam_values: list) -> go.Figure:
    """Visualiza cómo decae el peso de observaciones pasadas para distintos λ."""
    fig = go.Figure()
    days = np.arange(0, 60)
    palette = [COLORS["gold"], COLORS["sky"], COLORS["rose"], COLORS["emerald"]]
    for i, lam in enumerate(lam_values):
        weights = (1 - lam) * (lam ** days)
        fig.add_trace(go.Scatter(
            x=days, y=weights,
            name=f"λ = {lam}",
            line=dict(color=palette[i % len(palette)], width=2),
        ))
    pb = plotly_base(280)
    pb.pop("legend", None)
    pb["xaxis"]["type"] = "-"
    fig.update_layout(
        **pb,
        title=dict(
            text="Pesos de Observaciones Pasadas — (1−λ)·λᵏ",
            font=dict(size=12, color=COLORS["text"], family="Playfair Display"),
        ),
        xaxis_title="Días de rezago (k)",
        yaxis_title="Peso relativo",
        legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
    )
    fig.update_yaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9))
    fig.update_xaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9))
    return fig


# ── ARCH/GARCH helpers (sin cambios) ─────────────────────────

def fit_model(returns_pct, vol, p, q, o, dist):
    try:
        am  = arch_model(returns_pct, vol=vol, p=p, q=q, o=o,
                         dist=dist, mean="Constant", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        return res
    except Exception:
        return None


def fig_returns_vol(returns_pct, res, ticker):
    cond_vol = res.conditional_volatility
    colors   = [COLORS["emerald"] if v >= 0 else COLORS["rose"]
                for v in returns_pct.values]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.55], vertical_spacing=0.04)
    fig.add_trace(go.Bar(x=returns_pct.index, y=returns_pct.values,
        name="Log-rend. (%)", marker_color=colors,
        marker_line_width=0, opacity=0.85), row=1, col=1)
    fig.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol.values,
        name="σₜ condicional",
        line=dict(color=COLORS["gold"], width=1.6)), row=2, col=1)
    pb = plotly_base(460)
    fig.update_layout(**pb,
        title=dict(text=f"{ticker}  ·  Rendimientos y Volatilidad Condicional",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")))
    fig.update_yaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9))
    fig.update_xaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9))
    return fig


def fig_residuals(res):
    std_r = res.std_resid.dropna()
    (osm, osr), (slope, intercept, _) = stats.probplot(std_r.values, dist="norm")
    line_y = slope * np.array(osm) + intercept
    colors = [COLORS["emerald"] if v >= 0 else COLORS["rose"] for v in std_r.values]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Residuos Estandarizados", "Q-Q Residuos"])
    fig.add_trace(go.Bar(x=std_r.index, y=std_r.values,
        marker_color=colors, marker_line_width=0, opacity=0.85, name="εₜ/σₜ"),
        row=1, col=1)
    fig.add_hline(y=2,  line=dict(color=COLORS["rose"], width=1, dash="dash"), row=1, col=1)
    fig.add_hline(y=-2, line=dict(color=COLORS["rose"], width=1, dash="dash"), row=1, col=1)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Cuantiles",
        marker=dict(color=COLORS["gold"], size=3.5, opacity=0.55)), row=1, col=2)
    fig.add_trace(go.Scatter(x=osm, y=line_y, mode="lines", name="Normal",
        line=dict(color=COLORS["rose"], width=1.8, dash="dash")), row=1, col=2)
    pb = plotly_base(300)
    fig.update_layout(**pb,
        title=dict(text="Diagnóstico de Residuos Estandarizados",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        showlegend=False)
    for c in [1, 2]:
        fig.update_xaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9), row=1, col=c)
        fig.update_yaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text3"], size=9), row=1, col=c)
    return fig


def fig_forecast(res, horizon, ticker) -> go.Figure:
    # EGARCH no soporta método analítico para horizon > 1 → usar simulación
    method = "simulation" if horizon > 1 and hasattr(res.model, "volatility") \
                and res.model.volatility.__class__.__name__ == "EGARCH" \
                else "analytic"
    try: 
        fc = res.forecast(horizon=horizon, reindex=False, method=method)
    except ValueError:
        # Fallback: si aún falla, forzar simulación
        fc = res.forecast(horizon=horizon, reindex=False, method="simulation", simulations=1000)

    fc_vol   = np.sqrt(fc.variance.iloc[-1].values) 
    days     = np.arange(1, horizon + 1)
    hist_vol = float(res.conditional_volatility.mean())      

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([fc_vol * 1.1, (fc_vol * 0.9)[::-1]]),
        fill="toself", fillcolor="rgba(139,105,20,0.06)",
        line=dict(width=0), name="IC ±10%", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=fc_vol, mode="lines+markers",
        name="σₜ₊ₕ pronóstico",
        line=dict(color=COLORS["gold"], width=2),
        marker=dict(size=5, color=COLORS["gold"]),
    ))
    fig.add_hline(y=hist_vol,
        line=dict(color=COLORS["emerald"], width=1.2, dash="dot"),
        annotation_text=f"Vol. media hist.: {hist_vol:.3f}%",
        annotation_font=dict(color=COLORS["emerald"], size=9))
    pb = plotly_base(300)
    pb["xaxis"]["type"] = "-"
    pb.pop("hovermode", None)
    fig.update_layout(**pb,
        title=dict(text=f"{ticker}  ·  Pronóstico de Volatilidad — {horizon} pasos",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Días hacia adelante", yaxis_title="σ (%)")
    return fig


def render_model_cards(results, best_k):
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        color = MODEL_COLORS.get(name, COLORS["gold"])
        is_best = name == best_k
        with cols[i]:
            if res is None:
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                border-top:3px solid {color};border-radius:8px;padding:1.2rem;
                opacity:0.5;min-height:160px">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;
                    font-weight:700;color:{color};margin-bottom:.8rem">{name}</div>
                    <div style="font-size:.75rem;color:#8896A8">No convergió</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                best_badge = """<span style="background:#1A6B4A22;color:#1A6B4A;
                    border:1px solid #1A6B4A44;border-radius:3px;
                    font-size:.48rem;padding:1px 6px;margin-left:6px;
                    letter-spacing:.06em">★ MEJOR AIC</span>""" if is_best else ""

                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                border-top:3px solid {color};border-radius:8px;padding:1.2rem;
                min-height:160px">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;
                    font-weight:700;color:{color};margin-bottom:.8rem;
                    display:flex;align-items:center">
                        {name}{best_badge}
                    </div>
                    <div style="display:grid;grid-template-columns:1fr;gap:.5rem">
                        <div style="background:#F4F6FB;border-radius:5px;padding:.5rem .7rem">
                            <div style="font-family:'IBM Plex Mono',monospace;font-size:.48rem;
                            color:#8896A8;letter-spacing:.1em;text-transform:uppercase">Log-Likelihood</div>
                            <div style="font-family:'Playfair Display',serif;font-size:1rem;
                            font-weight:700;color:#1A2035">{res.loglikelihood:.2f}</div>
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem">
                            <div style="background:#F4F6FB;border-radius:5px;padding:.5rem .7rem">
                                <div style="font-family:'IBM Plex Mono',monospace;font-size:.48rem;
                                color:#8896A8;letter-spacing:.1em;text-transform:uppercase">AIC</div>
                                <div style="font-family:'Playfair Display',serif;font-size:.95rem;
                                font-weight:700;color:{'#1A6B4A' if is_best else '#1A2035'}">{res.aic:.2f}</div>
                            </div>
                            <div style="background:#F4F6FB;border-radius:5px;padding:.5rem .7rem">
                                <div style="font-family:'IBM Plex Mono',monospace;font-size:.48rem;
                                color:#8896A8;letter-spacing:.1em;text-transform:uppercase">BIC</div>
                                <div style="font-family:'Playfair Display',serif;font-size:.95rem;
                                font-weight:700;color:#1A2035">{res.bic:.2f}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SHOW — entrada principal del módulo
# ══════════════════════════════════════════════

def show():
    render_portafolio_badge()

    TICKERS = get_tickers()

    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">
                Módulo 03
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                ARCH / GARCH / EWMA
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
                    color:#8896A8;letter-spacing:0.08em;">
            Volatilidad condicional · ARCH(1) · GARCH(1,1) · EGARCH · EWMA RiskMetrics · Pronóstico
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Cargando datos..."):
        all_log = {}
        for t in TICKERS:
            d = get_rendimientos(t, years=3)
            idx = pd.to_datetime(d["fechas"])
            all_log[t] = pd.Series(d["log_returns"], index=idx)
        log_ret = pd.DataFrame(all_log).dropna()

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        ticker = st.selectbox("Activo", TICKERS, index=0)
    with c2:
        dist   = st.selectbox("Distribución", ["normal", "t", "skewt"], index=1)
    with c3:
        horizon = st.slider("Horizonte de pronóstico (días)", 5, 60, 20)

    returns_pct = log_ret[ticker] * 100

    # ════════════════════════════════════════
    # TAB layout: ARCH/GARCH  |  EWMA
    # ════════════════════════════════════════
    tab_garch, tab_ewma = st.tabs(["📊 ARCH / GARCH", "〰️ EWMA — Suavizamiento Exponencial"])

    # ════════════════════════════════════════
    # TAB 1 — ARCH / GARCH (sin cambios)
    # ════════════════════════════════════════
    with tab_garch:

        # ── 1. Test ARCH-LM ──
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        sec_title("① Justificación — Prueba de Efecto ARCH (ARCH-LM)")

        try:
            lm_stat, lm_p, _, _ = het_arch(returns_pct.values, nlags=5)
            detected = lm_p < 0.05
            color_lm = COLORS["rose"] if detected else COLORS["emerald"]

            col_lm1, col_lm2, col_lm3 = st.columns([1, 1, 2])
            with col_lm1:
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                border-top:3px solid {color_lm};border-radius:8px;padding:1.1rem">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.52rem;
                    color:#8896A8;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem">
                    Estadístico LM</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                    font-weight:700;color:#1A2035">{lm_stat:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_lm2:
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                border-top:3px solid {color_lm};border-radius:8px;padding:1.1rem">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.52rem;
                    color:#8896A8;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem">
                    p-valor</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                    font-weight:700;color:{color_lm}">{lm_p:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_lm3:
                st.markdown(f"""
                <div style="background:{'#FAF4F4' if detected else '#F4FAF6'};
                border:1px solid {color_lm}44;border-left:3px solid {color_lm};
                border-radius:8px;padding:1.1rem;height:100%">
                    <div style="font-size:.82rem;font-weight:600;color:{color_lm};margin-bottom:.3rem">
                    {'⚠️ Efecto ARCH detectado → justifica modelo GARCH' if detected else '✅ No se detecta efecto ARCH'}
                    </div>
                    <div style="font-size:.75rem;color:#4A5568;line-height:1.5">
                    H₀: no hay heterocedasticidad condicional.
                    {'Rechazamos H₀ — la varianza no es constante.' if detected else 'No rechazamos H₀ a α=5%.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            st.warning("No se pudo calcular la prueba ARCH-LM.")

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── 2. Comparación de modelos ──
        sec_title("② Ajuste de Modelos — ARCH(1) · GARCH(1,1) · EGARCH(1,1)", COLORS["sky"])

        with st.spinner("Ajustando modelos ARCH/GARCH..."):
            results = {}
            for name, cfg in MODELS.items():
                results[name] = fit_model(returns_pct, cfg["vol"], cfg["p"],
                                          cfg["q"], cfg["o"], dist)

        valid  = {k: v for k, v in results.items() if v is not None}
        best_k = min(valid, key=lambda k: valid[k].aic) if valid else None

        render_model_cards(results, best_k)

        with st.expander("Criterios de selección — AIC / BIC / Log-Likelihood"):
            st.markdown("""
            **AIC** (Akaike): −2·ℓ + 2k. Penaliza complejidad moderadamente.
            **BIC** (Bayesian): −2·ℓ + k·ln(n). Penaliza más severamente.
            **Log-Likelihood**: mayor es mejor. Mide ajuste al dato.

            El **EGARCH** modela ln(σ²), garantizando varianza positiva sin restricciones
            y captura el efecto apalancamiento de forma asimétrica.
            """)

        sel_name = st.selectbox("Modelo para diagnóstico y pronóstico",
                                list(valid.keys()) if valid else ["—"],
                                index=list(valid.keys()).index(best_k) if best_k in valid else 0)
        sel_res = valid.get(sel_name)

        if sel_res is None:
            st.error("No se pudo ajustar ningún modelo.")
        else:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            # ── 3. Rendimientos y Vol Condicional ──
            sec_title("③ Rendimientos y Volatilidad Condicional Estimada", COLORS["gold"])
            st.plotly_chart(fig_returns_vol(returns_pct, sel_res, ticker),
                            use_container_width=True)

            # ── Parámetros ──
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            sec_title("Parámetros Estimados", COLORS["violet"])

            params  = sel_res.params
            persist = sum(v for k, v in params.items()
                          if k.startswith("alpha") or k.startswith("beta"))

            param_cols = st.columns(min(len(params) + 1, 8))
            for i, (k, v) in enumerate(params.items()):
                if i < len(param_cols):
                    with param_cols[i]:
                        st.metric(k, f"{v:.6f}")
            if len(param_cols) > len(params):
                with param_cols[len(params)]:
                    st.metric("Persistencia (α+β)", f"{persist:.4f}",
                              "→ 1: memoria larga" if persist > 0.95 else "Estable")

            # ── 4. Diagnóstico ──
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            sec_title("④ Diagnóstico de Residuos Estandarizados", COLORS["rose"])
            st.plotly_chart(fig_residuals(sel_res), use_container_width=True)

            std_r = sel_res.std_resid.dropna()
            jb_s, jb_p = stats.jarque_bera(std_r.values)
            rechaza  = jb_p < 0.05
            color_jb = COLORS["rose"] if rechaza else COLORS["emerald"]

            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid {color_jb}44;
                        border-left:3px solid {color_jb};border-radius:6px;
                        padding:0.9rem 1.2rem;margin-top:0.5rem;">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#8896A8;">
                    Jarque-Bera sobre residuos: &nbsp;
                </span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                             color:#1A2035;font-weight:500;">
                    stat={jb_s:.2f} &nbsp; p={jb_p:.4f}
                </span>
                <span style="font-size:0.78rem;font-weight:600;color:{color_jb};margin-left:1rem;">
                    {'→ Residuos no normales — considerar distribución t-Student'
                      if rechaza else '→ Residuos aproximadamente normales'}
                </span>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Interpretación — Diagnóstico de residuos"):
                st.markdown("""
                Los **residuos estandarizados** εₜ/σₜ deben comportarse como ruido blanco con
                varianza unitaria si el modelo es correcto. Si Jarque-Bera rechaza normalidad,
                re-estima con distribución **t-Student** o **t asimétrica**.
                """)

            # ── 5. Pronóstico ──
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            sec_title(f"⑤ Pronóstico de Volatilidad — {horizon} Pasos", COLORS["emerald"])
            st.plotly_chart(fig_forecast(sel_res, horizon, ticker),
                            use_container_width=True)

            with st.expander("Interpretación — Pronóstico de volatilidad"):
                st.markdown("""
                El pronóstico converge a la **volatilidad incondicional de largo plazo** a medida
                que el horizonte aumenta. Este pronóstico alimenta el **VaR dinámico** (Módulo 5)
                y la construcción de portafolios eficientes (Módulo 6).
                """)

    # ════════════════════════════════════════
    # TAB 2 — EWMA
    # ════════════════════════════════════════
    with tab_ewma:

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Fórmula ──
        sec_title("① Fórmula Recursiva — EWMA (RiskMetrics)")

        st.markdown("""
        <div style="background:#F4F6FB;border:1px solid #D8DDE8;border-radius:8px;
                    padding:1.2rem 1.5rem;margin-bottom:1.2rem;">
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
                        color:#1A2035;font-weight:700;margin-bottom:0.6rem;text-align:center;">
                σ²ₜ = λ · σ²ₜ₋₁ &nbsp;+&nbsp; (1 − λ) · r²ₜ₋₁
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                        color:#4A5568;text-align:center;margin-top:0.4rem;line-height:2;">
                σ²ₜ = varianza estimada hoy &nbsp;|&nbsp;
                λ ∈ (0,1) = factor de decaimiento &nbsp;|&nbsp;
                r²ₜ₋₁ = cuadrado del retorno de ayer
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#4A5568;
                    line-height:1.9;margin-bottom:1rem;">
            La fórmula EWMA es <b style="color:#1A2035;">recursiva</b>: la varianza de hoy
            es un promedio ponderado de la varianza de ayer y el cuadrado del retorno de ayer.
            El parámetro λ controla cuánto peso recibe la historia reciente vs la observación más antigua.
            Un λ = 0.94 es el estándar de <b style="color:#8B6914;">RiskMetrics (J.P. Morgan, 1994)</b>,
            calibrado para capturar la dinámica de la volatilidad en mercados financieros diarios.
        </div>
        """, unsafe_allow_html=True)

        # ── Controles λ ──
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("② Configuración de Parámetros", COLORS["sky"])

        col_l1, col_l2, col_rw = st.columns(3)
        with col_l1:
            lam1 = st.slider(
                "λ₁ — RiskMetrics estándar",
                min_value=0.80, max_value=0.99, value=0.94, step=0.01,
                help="λ = 0.94 es el valor canónico de J.P. Morgan RiskMetrics para datos diarios."
            )
        with col_l2:
            lam2 = st.slider(
                "λ₂ — Valor alternativo (configurable)",
                min_value=0.80, max_value=0.99, value=0.85, step=0.01,
                help="Un λ menor da más peso a observaciones recientes — más reactivo pero más ruidoso."
            )
        with col_rw:
            roll_w = st.slider(
                "Ventana volatilidad rodante (días)",
                min_value=10, max_value=60, value=20, step=5,
                help="Ventana de la volatilidad muestral rodante usada como referencia."
            )

        # ── KPIs EWMA ──
        ewma1_last = float(ewma_volatility(returns_pct, lam1).iloc[-1])
        ewma2_last = float(ewma_volatility(returns_pct, lam2).iloc[-1])
        roll_last  = float(returns_pct.rolling(roll_w).std().iloc[-1])

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #D8DDE8;border-top:3px solid {COLORS['gold']};
            border-radius:8px;padding:1rem;text-align:center">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.5rem;color:#8896A8;
                letter-spacing:.15em;text-transform:uppercase;margin-bottom:4px">EWMA λ={lam1} (hoy)</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                font-weight:700;color:#8B6914">{ewma1_last:.4f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #D8DDE8;border-top:3px solid {COLORS['sky']};
            border-radius:8px;padding:1rem;text-align:center">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.5rem;color:#8896A8;
                letter-spacing:.15em;text-transform:uppercase;margin-bottom:4px">EWMA λ={lam2} (hoy)</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                font-weight:700;color:{COLORS['sky']}">{ewma2_last:.4f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #D8DDE8;border-top:3px solid #8896A8;
            border-radius:8px;padding:1rem;text-align:center">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.5rem;color:#8896A8;
                letter-spacing:.15em;text-transform:uppercase;margin-bottom:4px">Rodante {roll_w}d (hoy)</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                font-weight:700;color:#4A5568">{roll_last:.4f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            diff = ewma1_last - roll_last
            col_diff = COLORS["rose"] if diff > 0 else COLORS["emerald"]
            sign = "+" if diff > 0 else ""
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #D8DDE8;border-top:3px solid {col_diff};
            border-radius:8px;padding:1rem;text-align:center">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.5rem;color:#8896A8;
                letter-spacing:.15em;text-transform:uppercase;margin-bottom:4px">EWMA − Rodante</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                font-weight:700;color:{col_diff}">{sign}{diff:.4f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Gráfico comparativo ──
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        sec_title("③ Visualización — EWMA vs Volatilidad Muestral Rodante", COLORS["gold"])
        st.plotly_chart(fig_ewma(returns_pct, lam1, lam2, roll_w, ticker),
                        use_container_width=True)

        # ── Gráfico de decaimiento ──
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("④ Decaimiento de Pesos — (1−λ)·λᵏ", COLORS["violet"])

        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#4A5568;
                    line-height:1.8;margin-bottom:0.8rem;">
            El peso que el EWMA asigna a una observación de hace <i>k</i> días es
            <b style="color:#1A2035;">(1−λ)·λᵏ</b>.
            Con λ = 0.94, una observación de hace 30 días tiene un peso de
            {:.4f} — casi irrelevante. Con λ = 0.85 ese decay ocurre mucho más rápido.
        </div>
        """.format((1 - 0.94) * (0.94 ** 30)), unsafe_allow_html=True)

        st.plotly_chart(fig_ewma_decay([lam1, lam2, 0.97, 0.90]),
                        use_container_width=True)

        # ── Discusión ventajas y limitaciones ──
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("⑤ Ventajas y Limitaciones del EWMA", COLORS["rose"])

        col_v, col_l = st.columns(2)
        with col_v:
            st.markdown(f"""
            <div style="background:rgba(26,107,74,0.05);border:1px solid rgba(26,107,74,0.2);
                        border-top:3px solid {COLORS['emerald']};border-radius:8px;padding:1.1rem;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;font-weight:700;
                            color:{COLORS['emerald']};letter-spacing:.1em;text-transform:uppercase;
                            margin-bottom:.8rem;">✓ Ventajas</div>
                <div style="font-size:.78rem;color:#1A2035;line-height:2;">
                    <b>Parsimonia:</b> un solo parámetro λ — sin estimación por MLE.<br>
                    <b>Sin estimación:</b> λ = 0.94 es universal para datos diarios.<br>
                    <b>Decay constante:</b> pesos decrecen geométricamente de forma predecible.<br>
                    <b>Recursividad:</b> σ²ₜ solo requiere σ²ₜ₋₁ y rₜ₋₁ — O(1) en memoria.<br>
                    <b>Reactivo:</b> captura cambios rápidos de volatilidad en crisis.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_l:
            st.markdown(f"""
            <div style="background:rgba(139,42,42,0.05);border:1px solid rgba(139,42,42,0.2);
                        border-top:3px solid {COLORS['rose']};border-radius:8px;padding:1.1rem;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;font-weight:700;
                            color:{COLORS['rose']};letter-spacing:.1em;text-transform:uppercase;
                            margin-bottom:.8rem;">✗ Limitaciones</div>
                <div style="font-size:.78rem;color:#1A2035;line-height:2;">
                    <b>Sin asimetría:</b> no captura el efecto leverage — caídas y subidas
                    del mismo tamaño tienen el mismo impacto en σ².<br>
                    <b>Sin reversión a la media:</b> la varianza incondicional de largo plazo
                    no está definida — el proceso no es covarianza-estacionario.<br>
                    <b>λ fijo:</b> no se adapta al régimen del mercado — en crisis
                    podría necesitarse un λ distinto.<br>
                    <b>Sin pronóstico estructural:</b> no genera curva de volatilidad futura
                    como GARCH(1,1).
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Comparación EWMA vs GARCH ──
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        with st.expander("Comparación EWMA vs GARCH(1,1) — ¿Cuándo usar cada uno?"):
            st.markdown("""
            | Criterio | EWMA | GARCH(1,1) |
            |---|---|---|
            | Parámetros | 1 (λ fijo) | 3 (ω, α, β) estimados |
            | Estacionariedad | No garantizada | Sí si α+β < 1 |
            | Reversión a la media | No | Sí |
            | Captura de asimetría | No | No (requiere GJR/EGARCH) |
            | Pronóstico multi-paso | No estructural | Sí, converge a σ² incondicional |
            | Costo computacional | Mínimo O(1) | Optimización numérica |
            | Uso típico | VaR diario, sistemas en tiempo real | Investigación, opciones, pricing |

            **Regla práctica:** para VaR operacional diario, EWMA con λ=0.94 es suficiente
            y robusto. Para modelar la estructura temporal de la volatilidad o calibrar
            modelos de opciones, GARCH(1,1) o EGARCH son necesarios.
            """)