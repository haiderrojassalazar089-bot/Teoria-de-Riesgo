"""
pages/m8_macro.py — Módulo 8: Macro & Benchmark
Streamlit + Plotly | yfinance 1.2.0
Indicadores macro · Alpha de Jensen · Tracking Error · Information Ratio
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st
import requests

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.client import (get_rendimientos, get_precios, get_macro, get_capm,
                          SECTOR_MAP)
from utils.theme import plotly_base, COLORS
from utils.dynamic_tickers import get_tickers, get_ticker_colors, render_portafolio_badge

BENCHMARK = "^GSPC"


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


# ── Indicadores macro ─────────────────────────────────────────

def get_macro_indicators(rf):
    import yfinance as yf

    indicators = {}

    indicators["Tasa Libre de Riesgo (T-Bill 3M)"] = {
        "valor": rf["display"], "fuente": "Yahoo Finance · ^IRX",
        "color": COLORS["gold"], "descripcion": "Rendimiento anualizado T-Bill 3 meses"
    }

    try:
        tyx = yf.download("^TNX", period="5d", progress=False,
                           auto_adjust=True, multi_level_index=False)
        tyx.columns.name = None
        tyx10 = float(tyx["Close"].dropna().iloc[-1])
        indicators["Rendimiento Tesoro 10A"] = {
            "valor": f"{tyx10:.2f}%", "fuente": "Yahoo Finance · ^TNX",
            "color": COLORS["sky"], "descripcion": "Proxy de expectativas de inflación a largo plazo"
        }
        spread = tyx10 - rf["annual"] * 100
        indicators["Spread 10Y−3M (Curva)"] = {
            "valor": f"{spread:+.2f}%",
            "fuente": "Calculado",
            "color": COLORS["emerald"] if spread > 0 else COLORS["rose"],
            "descripcion": "Spread positivo → curva normal; negativo → inversión (señal recesión)"
        }
    except Exception:
        pass

    try:
        cop = yf.download("USDCOP=X", period="5d", progress=False,
                           auto_adjust=True, multi_level_index=False)
        cop.columns.name = None
        cop_val = float(cop["Close"].dropna().iloc[-1])
        indicators["USD / COP"] = {
            "valor": f"${cop_val:,.0f}", "fuente": "Yahoo Finance · USDCOP=X",
            "color": COLORS["violet"], "descripcion": "Pesos colombianos por dólar"
        }
    except Exception:
        indicators["USD / COP"] = {
            "valor": "N/D", "fuente": "No disponible",
            "color": COLORS["text3"], "descripcion": "Tipo de cambio peso colombiano"
        }

    try:
        vix = yf.download("^VIX", period="5d", progress=False,
                           auto_adjust=True, multi_level_index=False)
        vix.columns.name = None
        vix_val = float(vix["Close"].dropna().iloc[-1])
        nivel_vix = "Alta" if vix_val > 30 else "Moderada" if vix_val > 20 else "Baja"
        indicators["VIX (Volatilidad impl.)"] = {
            "valor": f"{vix_val:.2f}",
            "fuente": "Yahoo Finance · ^VIX",
            "color": COLORS["rose"] if vix_val > 30 else COLORS["gold"],
            "descripcion": f"Índice de miedo del mercado — Volatilidad {nivel_vix}"
        }
    except Exception:
        pass

    try:
        eur = yf.download("EURUSD=X", period="5d", progress=False,
                           auto_adjust=True, multi_level_index=False)
        eur.columns.name = None
        eur_val = float(eur["Close"].dropna().iloc[-1])
        indicators["EUR / USD"] = {
            "valor": f"{eur_val:.4f}",
            "fuente": "Yahoo Finance · EURUSD=X",
            "color": COLORS["sky2"],
            "descripcion": "Tipo de cambio euro / dólar estadounidense"
        }
    except Exception:
        pass

    return indicators


# ── Métricas de desempeño ─────────────────────────────────────

def compute_performance(port_ret, bench_ret, rf_daily, TICKERS):
    df = pd.concat([port_ret, bench_ret], axis=1).dropna()
    df.columns = ["port", "bench"]

    excess_port  = df["port"]  - rf_daily
    excess_bench = df["bench"] - rf_daily
    active_ret   = df["port"]  - df["bench"]

    slope, intercept, r, p, se = stats.linregress(excess_bench, excess_port)

    beta_jensen   = slope
    alpha_jensen  = intercept * 252
    alpha_p_valor = p

    ann_ret_port  = df["port"].mean()  * 252
    ann_ret_bench = df["bench"].mean() * 252
    vol_port      = df["port"].std()   * np.sqrt(252)
    vol_bench     = df["bench"].std()  * np.sqrt(252)

    sharpe_port  = (ann_ret_port  - rf_daily * 252) / vol_port  if vol_port  > 0 else 0
    sharpe_bench = (ann_ret_bench - rf_daily * 252) / vol_bench if vol_bench > 0 else 0

    tracking_error = active_ret.std() * np.sqrt(252)
    info_ratio     = (active_ret.mean() * 252) / tracking_error if tracking_error > 0 else 0

    dd_port  = ((1 + df["port"]).cumprod()  / (1 + df["port"]).cumprod().cummax()  - 1).min()
    dd_bench = ((1 + df["bench"]).cumprod() / (1 + df["bench"]).cumprod().cummax() - 1).min()

    return {
        "alpha_jensen"   : alpha_jensen,
        "alpha_p_valor"  : alpha_p_valor,
        "beta_jensen"    : beta_jensen,
        "ann_ret_port"   : ann_ret_port,
        "ann_ret_bench"  : ann_ret_bench,
        "vol_port"       : vol_port,
        "vol_bench"      : vol_bench,
        "sharpe_port"    : sharpe_port,
        "sharpe_bench"   : sharpe_bench,
        "tracking_error" : tracking_error,
        "info_ratio"     : info_ratio,
        "dd_port"        : dd_port,
        "dd_bench"       : dd_bench,
        "r2"             : r ** 2,
        "active_ret"     : active_ret,
        "df"             : df,
    }


def compute_optimal_weights(log_ret, rf_daily, TICKERS):
    from scipy.optimize import minimize
    mu  = log_ret[TICKERS].mean().values
    cov = log_ret[TICKERS].cov().values
    n   = len(mu)

    def neg_sharpe(w):
        r = np.dot(w, mu) * 252
        v = np.sqrt(w @ cov @ w) * np.sqrt(252)
        return -(r - rf_daily * 252) / v if v > 0 else 0

    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                   bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}])
    return res.x if res.success else np.ones(n) / n


# ── Gráficos ──────────────────────────────────────────────────

def fig_acumulado(prices_df, w_port, bench_col, port_label, TICKERS):
    port_norm  = (prices_df[TICKERS] * w_port).sum(axis=1)
    port_norm  = port_norm / port_norm.iloc[0] * 100
    bench_norm = prices_df[bench_col] / prices_df[bench_col].iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bench_norm.index, y=bench_norm.values,
        name="S&P 500 (benchmark)",
        line=dict(color=COLORS["text3"], width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=port_norm.index, y=port_norm.values,
        name=port_label,
        line=dict(color=COLORS["gold"], width=2.2),
        fill="tonexty",
        fillcolor="rgba(139,105,20,0.06)",
    ))
    fig.update_layout(**plotly_base(360),
        title=dict(text=f"Rendimiento Acumulado — {port_label} vs S&P 500 (base 100)",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")))
    return fig


def fig_retorno_activo(active_ret):
    colors = [COLORS["emerald"] if v >= 0 else COLORS["rose"] for v in active_ret.values]
    fig    = go.Figure()
    fig.add_trace(go.Bar(
        x=active_ret.index, y=active_ret.values,
        marker_color=colors, marker_line_width=0, opacity=0.85,
        name="Retorno activo",
    ))
    fig.add_hline(y=0, line=dict(color=COLORS["border2"], width=0.8))
    fig.update_layout(**plotly_base(260),
        title=dict(text="Retorno Activo Diario (Portafolio − S&P 500)",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")))
    return fig


def fig_rolling_alpha(df_ret, rf_daily, window=60):
    active = df_ret["port"] - df_ret["bench"]
    rolling_alpha = active.rolling(window).mean() * 252
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=COLORS["border2"], width=0.8, dash="dash"))
    fig.add_trace(go.Scatter(
        x=rolling_alpha.index, y=rolling_alpha.values,
        name=f"Alpha rodante ({window}d)",
        line=dict(color=COLORS["emerald"], width=1.6),
        fill="tozeroy",
        fillcolor="rgba(26,107,74,0.07)",
    ))
    fig.update_layout(**plotly_base(280),
        title=dict(text=f"Alpha de Jensen Rodante — Ventana {window} días",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")))
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
                Módulo 08
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                Macro & Benchmark
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
                    color:#8896A8;letter-spacing:0.08em;">
            Contexto macroeconómico · Alpha de Jensen · Tracking Error · Information Ratio · vs S&P 500
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Cargando datos de mercado y macro..."):
        import pandas as pd, numpy as np
        all_log = {}
        for t in TICKERS + [BENCHMARK]:
            d = get_precios(t, years=3)
            idx = pd.to_datetime([p["fecha"] for p in d["precios"]])
            closes = [p["close"] for p in d["precios"]]
            s = pd.Series(closes, index=idx)
            all_log[t] = s
        prices_df = pd.DataFrame(all_log).dropna()
        log_ret = np.log(prices_df / prices_df.shift(1)).dropna()
        macro_d = get_macro()
        rf = {"annual": macro_d["tasa_libre_riesgo"]["valor"],
              "daily": macro_d["tasa_libre_riesgo"]["valor"]/252,
              "display": macro_d["tasa_libre_riesgo"]["display"],
              "source": macro_d["tasa_libre_riesgo"]["fuente"]}

    c1, c2 = st.columns([2, 1])
    with c1:
        tipo_port = st.radio(
            "Portafolio a comparar",
            ["Equi-ponderado (20% cada activo)", "Máximo Sharpe (optimizado Markowitz)"],
            horizontal=True,
        )
    with c2:
        ventana_alpha = st.slider("Ventana alpha rodante (días)", 20, 120, 60)

    if "Máximo" in tipo_port:
        w_port = compute_optimal_weights(log_ret, rf["daily"], TICKERS)
        port_label = "Portafolio Máx. Sharpe"
    else:
        w_port = np.ones(len(TICKERS)) / len(TICKERS)
        port_label = "Portafolio Equi-ponderado"

    port_ret  = (log_ret[TICKERS] * w_port).sum(axis=1)
    bench_ret = log_ret[BENCHMARK]

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    sec_title("① Panel de Indicadores Macroeconómicos")

    with st.spinner("Obteniendo indicadores macro..."):
        macro = get_macro_indicators(rf)

    macro_cols = st.columns(len(macro))
    for i, (nombre, datos) in enumerate(macro.items()):
        with macro_cols[i]:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                        border-top:2px solid {datos['color']};border-radius:6px;
                        padding:1rem;text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.5rem;
                            color:#8896A8;letter-spacing:0.12em;text-transform:uppercase;
                            margin-bottom:8px;line-height:1.4;">{nombre}</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.15rem;
                            font-weight:700;color:{datos['color']};margin-bottom:4px;">
                    {datos['valor']}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.48rem;
                            color:#8896A8;margin-bottom:4px;">{datos['fuente']}</div>
                <div style="font-family:'Inter',sans-serif;font-size:0.68rem;
                            color:#4A5568;line-height:1.4;">{datos['descripcion']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    sec_title("② Rendimiento Acumulado — Portafolio vs S&P 500 (Base 100)", COLORS["sky"])
    st.plotly_chart(fig_acumulado(prices_df, w_port, BENCHMARK, port_label, TICKERS),
                    use_container_width=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    perf = compute_performance(port_ret, bench_ret, rf["daily"], TICKERS)

    sec_title("③ Métricas de Desempeño vs Benchmark", COLORS["gold"])

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    alpha_sig = perf["alpha_p_valor"] < 0.05
    k1.metric("Alpha de Jensen (anual)", f"{perf['alpha_jensen']:+.4f}",
              "✓ Significativo (p<5%)" if alpha_sig else "No significativo")
    k2.metric("Beta (Jensen)", f"{perf['beta_jensen']:.4f}")
    k3.metric("Tracking Error (anual)", f"{perf['tracking_error']:.2%}")
    k4.metric("Information Ratio", f"{perf['info_ratio']:.3f}",
              "Bueno (>0.5)" if perf["info_ratio"] > 0.5 else "Bajo (<0.5)")
    k5.metric("Sharpe Portafolio", f"{perf['sharpe_port']:.3f}",
              f"Benchmark: {perf['sharpe_bench']:.3f}")
    k6.metric("R² (vs S&P 500)", f"{perf['r2']:.4f}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    sec_title("④ Tabla Comparativa de Desempeño", COLORS["emerald"])

    filas_tabla = [
        ("Rendimiento Anualizado", f"{perf['ann_ret_port']:.2%}", f"{perf['ann_ret_bench']:.2%}"),
        ("Volatilidad Anualizada", f"{perf['vol_port']:.2%}", f"{perf['vol_bench']:.2%}"),
        ("Ratio de Sharpe", f"{perf['sharpe_port']:.3f}", f"{perf['sharpe_bench']:.3f}"),
        ("Máximo Drawdown", f"{perf['dd_port']:.2%}", f"{perf['dd_bench']:.2%}"),
        ("Alpha de Jensen (anual)", f"{perf['alpha_jensen']:+.4f}", "—"),
        ("Beta (regresión)", f"{perf['beta_jensen']:.4f}", "1.0000"),
        ("Tracking Error", f"{perf['tracking_error']:.2%}", "—"),
        ("Information Ratio", f"{perf['info_ratio']:.3f}", "—"),
        ("R² vs S&P 500", f"{perf['r2']:.4f}", "1.0000"),
    ]

    df_tabla = pd.DataFrame(filas_tabla, columns=["Métrica", port_label, "S&P 500"])
    st.dataframe(df_tabla, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        sec_title("⑤ Retorno Activo Diario", COLORS["rose"])
        st.plotly_chart(fig_retorno_activo(perf["active_ret"]), use_container_width=True)
    with col_r:
        sec_title(f"Alpha Rodante ({ventana_alpha}d)", COLORS["violet"])
        st.plotly_chart(fig_rolling_alpha(perf["df"], rf["daily"], ventana_alpha),
                        use_container_width=True)

    with st.expander("Interpretación — ¿El portafolio supera al benchmark?"):
        supera    = perf["ann_ret_port"] > perf["ann_ret_bench"]
        alpha_pos = perf["alpha_jensen"] > 0
        ir_ok     = perf["info_ratio"] > 0.3

        st.markdown(f"""
        ### Veredicto: {port_label}

        **Rendimiento:** el portafolio generó **{perf['ann_ret_port']:.2%}** anualizado frente a
        **{perf['ann_ret_bench']:.2%}** del S&P 500.
        {'✅ **Superó** al benchmark en términos de retorno bruto.' if supera
         else '⛔ **No superó** al benchmark en retorno bruto.'}

        **Alpha de Jensen:** {perf['alpha_jensen']:+.4f} anualizado, p-valor = {perf['alpha_p_valor']:.4f}.
        {'✅ El alpha es **estadísticamente significativo** al 5%: el portafolio genera retorno extra genuino.' if alpha_sig
         else '⚠️ El alpha **no es significativo**: el exceso de retorno podría deberse al azar.'}

        **Information Ratio ({perf['info_ratio']:.3f}):**
        {'✅ IR > 0.5 indica gestión activa competente.' if perf['info_ratio'] > 0.5
         else '⚠️ IR < 0.5: el exceso de retorno no justifica el riesgo activo asumido.'
         if perf['info_ratio'] > 0 else '⛔ IR negativo: el portafolio destruye valor frente al benchmark.'}

        **Tracking Error ({perf['tracking_error']:.2%}):** mide la volatilidad del retorno activo.

        **Conclusión:** {'El portafolio **añade valor** consistente sobre el benchmark.' if (supera and alpha_pos and ir_ok)
         else 'El portafolio **no demuestra** ventaja estadística sostenida sobre el S&P 500. '
              'Considera ajustar la selección de activos o pesos.'}
        """)

    with st.expander("Definición de métricas — Referencia académica"):
        st.markdown("""
        | Métrica | Fórmula | Interpretación |
        |---------|---------|----------------|
        | **Alpha de Jensen** | αⱼ = E[Rₚ] − [Rf + β·(E[Rm]−Rf)] | Retorno no explicado por CAPM |
        | **Tracking Error** | TE = σ(Rₚ − Rm) | Volatilidad del retorno activo |
        | **Information Ratio** | IR = (E[Rₚ]−E[Rm]) / TE | Retorno activo por unidad de riesgo activo |
        | **Beta (Jensen)** | β = Cov(Rₚ,Rm) / Var(Rm) | Sensibilidad al mercado |
        | **Sharpe Ratio** | SR = (E[R]−Rf) / σ | Retorno por unidad de riesgo total |
        | **Máximo Drawdown** | MDD = max[(Pₜ−Pₘₐₓ)/Pₘₐₓ] | Peor caída desde máximo histórico |

        *Fuentes: Jensen (1968), Grinold & Kahn (2000), Sharpe (1966)*
        """)