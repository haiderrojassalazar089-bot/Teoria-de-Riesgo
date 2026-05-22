"""
pages/m12_renta_fija.py — Módulo 12: Renta Fija
Streamlit + Plotly | scipy · numpy
Curva del Tesoro US · Nelson-Siegel · Duración · Convexidad
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.theme import plotly_base, COLORS
from utils.dynamic_tickers import render_portafolio_badge


# ══════════════════════════════════════════════════════════════
# DATOS — Curva del Tesoro US (FRED o muestra)
# ══════════════════════════════════════════════════════════════

TREASURY_MATURITIES = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
TREASURY_LABELS     = ["3M", "1Y", "2Y", "5Y", "10Y", "30Y"]

def fetch_treasury_curve(fred_api_key: str = ""):
    """
    Intenta obtener la curva del Tesoro US desde FRED.
    Si no hay API key, usa datos de muestra realistas.
    """
    if fred_api_key:
        try:
            from fredapi import Fred
            fred   = Fred(api_key=fred_api_key)
            series = ["DGS3MO","DGS1","DGS2","DGS5","DGS10","DGS30"]
            yields = []
            for s in series:
                val = fred.get_series(s).dropna()
                yields.append(float(val.iloc[-1]))
            date   = str(fred.get_series("DGS10").dropna().index[-1].date())
            source = "FRED (live)"
            return yields, date, source
        except Exception:
            pass

    # Muestra representativa (curva invertida 2024-2025)
    yields = [5.25, 4.95, 4.60, 4.25, 4.15, 4.05]
    return yields, "2025-01-15 (muestra)", "Datos de muestra — agrega FRED_API_KEY para datos live"

def identify_shape(maturities, yields):
    spread = yields[4] - yields[2]   # 10Y − 2Y
    if   spread >  0.50: shape, color = "Normal",   COLORS["emerald"]
    elif spread < -0.10: shape, color = "Invertida", COLORS["rose"]
    else:                shape, color = "Plana",     COLORS["gold"]
    return shape, color, round(spread * 100, 2)


# ══════════════════════════════════════════════════════════════
# CLASE YieldCurve
# ══════════════════════════════════════════════════════════════

class YieldCurve:
    def __init__(self, maturities, yields):
        self.tau = np.array(maturities, dtype=float)
        self.y   = np.array(yields,     dtype=float)
        self._cs = CubicSpline(self.tau, self.y)
        self.params = None   # (β0, β1, β2, λ)

    def spot_curve(self, taus):
        return self._cs(np.asarray(taus))

    @staticmethod
    def _ns(tau, b0, b1, b2, lam):
        x  = tau / lam
        ex = np.exp(-x)
        f1 = (1 - ex) / x
        f2 = f1 - ex
        return b0 + b1 * f1 + b2 * f2

    def fit_nelson_siegel(self):
        tau, y = self.tau, self.y

        def res(p):
            return self._ns(tau, *p) - y

        x0     = [y[-1], y[0] - y[-1], 0.0, 1.5]
        bounds = ([-np.inf, -np.inf, -np.inf, 0.01],
                  [ np.inf,  np.inf,  np.inf, 30.0])
        sol    = least_squares(res, x0, bounds=bounds, method="trf")
        b0, b1, b2, lam = sol.x
        fitted = self._ns(tau, b0, b1, b2, lam)
        rmse   = float(np.sqrt(np.mean((fitted - y) ** 2)))
        self.params = (b0, b1, b2, lam)
        return b0, b1, b2, lam, rmse, fitted

    def ns_curve(self, taus):
        if self.params is None:
            raise ValueError("Primero llama fit_nelson_siegel()")
        return self._ns(np.asarray(taus), *self.params)


# ══════════════════════════════════════════════════════════════
# CLASE Bond
# ══════════════════════════════════════════════════════════════

class Bond:
    def __init__(self, face=1000.0, coupon=0.05, maturity=10.0, freq=2, ytm=0.045):
        self.F, self.c, self.T, self.m, self.y = face, coupon, maturity, freq, ytm
        n          = int(maturity * freq)
        pmt        = face * coupon / freq
        self.tp    = np.arange(1, n + 1)               # periodos
        self.times = self.tp / freq                    # años
        self.cfs   = np.full(n, pmt)
        self.cfs[-1] += face

    def price(self, ytm=None):
        r = (ytm or self.y) / self.m
        return float(np.sum(self.cfs / (1 + r) ** self.tp))

    def macaulay(self, ytm=None):
        r    = (ytm or self.y) / self.m
        pv   = self.cfs / (1 + r) ** self.tp
        return float(np.sum(self.tp * pv) / np.sum(pv) / self.m)

    def modified(self, ytm=None):
        y = ytm or self.y
        return self.macaulay(ytm) / (1 + y / self.m)

    def convexity(self, ytm=None):
        r  = (ytm or self.y) / self.m
        pv = self.cfs / (1 + r) ** self.tp
        P  = np.sum(pv)
        return float(np.sum(self.tp * (self.tp + 1) * pv / (1 + r) ** 2) / (P * self.m ** 2))

    def sensitivity(self, shock_bps):
        dy   = shock_bps / 10_000
        P0   = self.price()
        Dm   = self.modified()
        C    = self.convexity()
        P_lin = P0 * (1 - Dm * dy)
        P_dc  = P0 * (1 - Dm * dy + 0.5 * C * dy ** 2)
        P_ex  = self.price(self.y + dy)
        return P_lin, P_dc, P_ex


# ══════════════════════════════════════════════════════════════
# HELPERS DE LAYOUT (mismo estilo que m4_capm.py)
# ══════════════════════════════════════════════════════════════

def sec_title(text, color=None):
    col = color or COLORS["gold"]
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
                letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.6rem;
                border-left:2px solid {col};padding-left:8px;">
        {text}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════

def fig_yield_curve(maturities, yields, labels, spline_taus, spline_y, ns_taus, ns_y, fitted_obs, date):
    fig = go.Figure()

    # Cubic Spline
    fig.add_trace(go.Scatter(
        x=spline_taus, y=spline_y, mode="lines",
        name="Cubic Spline",
        line=dict(color=COLORS.get("sky","#38BDF8"), width=2, dash="dot"),
    ))
    # Nelson-Siegel continua
    fig.add_trace(go.Scatter(
        x=ns_taus, y=ns_y, mode="lines",
        name="Nelson-Siegel",
        line=dict(color=COLORS["rose"], width=2.5),
    ))
    # NS ajustado en puntos observados
    fig.add_trace(go.Scatter(
        x=maturities, y=fitted_obs, mode="markers",
        name="NS en plazos observados",
        marker=dict(color=COLORS["rose"], size=9, symbol="x", line=dict(width=2)),
    ))
    # Puntos FRED
    fig.add_trace(go.Scatter(
        x=maturities, y=yields, mode="markers+text",
        name="Observado (FRED)",
        text=labels, textposition="top center",
        textfont=dict(family="IBM Plex Mono", size=10, color=COLORS["gold"]),
        marker=dict(color=COLORS["gold"], size=11,
                    line=dict(color="white", width=1.5)),
    ))

    pb = plotly_base(400)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(
            text=f"Curva del Tesoro US · {date}",
            font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Plazo (años)",
        yaxis_title="Rendimiento (%)",
        
    )
    return fig


def fig_sensitivity(bond: Bond, shocks):
    P0   = bond.price()
    p_lin, p_dc, p_ex = [], [], []
    for s in shocks:
        a, b, c = bond.sensitivity(s)
        p_lin.append(a); p_dc.append(b); p_ex.append(c)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shocks, y=p_lin, mode="lines+markers",
        name="Lineal (solo D*)",
        line=dict(color=COLORS["rose"], dash="dot", width=2),
        marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=shocks, y=p_dc, mode="lines+markers",
        name="D* + Convexidad",
        line=dict(color=COLORS["gold"], dash="dash", width=2),
        marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=shocks, y=p_ex, mode="lines+markers",
        name="Reprice exacto",
        line=dict(color=COLORS["emerald"], width=3),
        marker=dict(size=8)))
    fig.add_hline(y=P0, line_dash="dot", line_color=COLORS.get("text3","#8896A8"),
                  annotation_text=f"P₀ = ${P0:.2f}",
                  annotation_font=dict(family="IBM Plex Mono", size=10))
    fig.add_vline(x=0, line_dash="dot", line_color=COLORS.get("text3","#8896A8"))

    pb = plotly_base(380)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text="Precio vs Shock de Tasa — Tres Métodos",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Shock (basis points)",
        yaxis_title="Precio del Bono ($)",
        
    )
    return fig


# ══════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL show()
# ══════════════════════════════════════════════════════════════

def show():
    render_portafolio_badge()

    # ── Header ──────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">
                Módulo 12
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                Renta Fija
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
                    color:#8896A8;letter-spacing:0.08em;">
            Curva del Tesoro US · Nelson-Siegel · Duración de Macaulay · Convexidad
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌐 Curva de Rendimiento", "🔢 Bono Sintético"])

    # ══════════════════════════════════════════════════════
    # TAB 1 — CURVA DE RENDIMIENTO
    # ══════════════════════════════════════════════════════
    with tab1:

        with st.expander("⚙️ Configuración FRED (opcional)"):
            fred_key = st.text_input(
                "FRED API Key",
                type="password",
                help="Gratis en fred.stlouisfed.org. Sin key usa datos de muestra.",
            )

        if st.button("📡 Cargar Curva del Tesoro", type="primary"):
            with st.spinner("Obteniendo datos..."):
                yields, date, source = fetch_treasury_curve(fred_key)

            maturities = TREASURY_MATURITIES
            labels     = TREASURY_LABELS
            shape, shape_color, spread = identify_shape(maturities, yields)

            # ── KPIs ────────────────────────────────────────
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Forma de la Curva", shape)
            k2.metric("Spread 10Y−2Y", f"{spread:.0f} pb")
            k3.metric("Tasa 3M", f"{yields[0]:.2f}%")
            k4.metric("Tasa 10Y", f"{yields[4]:.2f}%")

            # ── Implicación macroeconómica ───────────────────
            macro = {
                "Normal":   "📈 Curva **normal** (pendiente positiva): el mercado anticipa crecimiento económico y posibles alzas de tasas en el mediano plazo. Los inversionistas exigen mayor compensación por plazos más largos.",
                "Invertida":"📉 Curva **invertida** (10Y < 2Y): indicador histórico de recesión en 12-18 meses. El mercado anticipa recortes de tasas por parte de la Fed.",
                "Plana":    "➡️ Curva **plana**: incertidumbre sobre el ciclo económico. Puede ser una transición entre curva normal e invertida, o reflejar política monetaria restrictiva.",
            }
            st.info(macro[shape])
            st.caption(f"Fuente: {source}")

            # ── Nelson-Siegel ────────────────────────────────
            yc = YieldCurve(maturities, yields)
            b0, b1, b2, lam, rmse, fitted_obs = yc.fit_nelson_siegel()

            # Curvas continuas
            taus      = np.linspace(0.25, 30, 200)
            spline_y  = yc.spot_curve(taus)
            ns_y      = yc.ns_curve(taus)

            # ── Gráfico principal ────────────────────────────
            sec_title("① Curva Spot + Nelson-Siegel", COLORS.get("sky","#38BDF8"))
            st.plotly_chart(
                fig_yield_curve(maturities, yields, labels,
                                taus, spline_y, taus, ns_y,
                                fitted_obs, date),
                use_container_width=True,
            )

            # ── Parámetros NS ────────────────────────────────
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            sec_title("② Parámetros Nelson-Siegel", COLORS["gold"])

            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("β₀ Nivel LP",   f"{b0:.4f}%")
            p2.metric("β₁ Pendiente",  f"{b1:.4f}%")
            p3.metric("β₂ Curvatura",  f"{b2:.4f}%")
            p4.metric("λ  Decay",      f"{lam:.4f}")
            rmse_delta = "✅ Bueno" if rmse < 0.05 else "⚠️ Aceptable" if rmse < 0.15 else "❌ Alto"
            p5.metric("RMSE",          f"{rmse*100:.4f}%", rmse_delta)

            with st.expander("📖 Interpretación de parámetros"):
                st.markdown(f"""
                | Parámetro | Valor | Significado |
                |-----------|-------|-------------|
                | **β₀ = {b0:.4f}** | Nivel LP | Tasa a la que converge la curva en el largo plazo |
                | **β₁ = {b1:.4f}** | Pendiente | Diferencia corto − largo; {'positivo = curva empinada' if b1>0 else 'negativo = curva invertida'} |
                | **β₂ = {b2:.4f}** | Curvatura | {'Joroba positiva en plazos medios' if b2>0 else 'Concavidad en plazos medios'} |
                | **λ  = {lam:.4f}** | Decay | La curvatura máxima ocurre en τ* ≈ {lam*1.7933:.2f} años |
                """)

            # ── Tabla observado vs ajustado ──────────────────
            sec_title("③ Observado vs Ajustado", COLORS["emerald"])
            df = pd.DataFrame({
                "Plazo":           labels,
                "Madurez (años)":  maturities,
                "Observado (%)":   [round(y, 4) for y in yields],
                "NS Ajustado (%)": [round(f, 4) for f in fitted_obs],
                "Error (pb)":      [round((f - y) * 100, 2)
                                    for f, y in zip(fitted_obs, yields)],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════
    # TAB 2 — BONO SINTÉTICO
    # ══════════════════════════════════════════════════════
    with tab2:

        sec_title("Parámetros del Bono Sintético", COLORS["gold"])
        col_l, col_r = st.columns(2)

        with col_l:
            face     = st.number_input("Valor nominal F ($)", 100.0, 1_000_000.0, 1000.0, 100.0)
            coupon   = st.slider("Tasa de cupón anual (%)", 0.1, 15.0, 5.0, 0.1) / 100
            maturity = st.slider("Vencimiento (años)", 1, 30, 10)

        with col_r:
            freq     = st.selectbox(
                "Frecuencia de pago",
                [1, 2, 4, 12],
                format_func=lambda x: {1:"Anual",2:"Semestral",4:"Trimestral",12:"Mensual"}[x],
                index=1,
            )
            ytm      = st.slider("YTM — Rendimiento al vencimiento (%)", 0.1, 20.0, 4.5, 0.1) / 100

        # Indicador prima/descuento
        if   coupon > ytm: st.success(f"💡 Cupón ({coupon*100:.1f}%) > YTM ({ytm*100:.1f}%) → bono **sobre la par** (prima)")
        elif coupon < ytm: st.warning(f"💡 Cupón ({coupon*100:.1f}%) < YTM ({ytm*100:.1f}%) → bono **bajo la par** (descuento)")
        else:              st.info   ("💡 Cupón = YTM → bono **a la par**")

        bond = Bond(face=face, coupon=coupon, maturity=maturity, freq=freq, ytm=ytm)

        P0   = bond.price()
        D_mac = bond.macaulay()
        D_mod = bond.modified()
        C    = bond.convexity()

        # ── KPIs ──────────────────────────────────────────
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        sec_title("① Métricas Principales", COLORS["emerald"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💰 Precio",          f"${P0:.2f}")
        m2.metric("⏱ Dur. Macaulay",    f"{D_mac:.4f} años")
        m3.metric("📉 Dur. Modificada",  f"{D_mod:.4f}")
        m4.metric("📐 Convexidad",       f"{C:.4f}")

        with st.expander("📖 Interpretación"):
            st.markdown(f"""
            | Métrica | Valor | Interpretación |
            |---------|-------|----------------|
            | **Precio** | ${P0:.2f} | {'Prima sobre par' if P0>face else 'Descuento bajo par' if P0<face else 'A la par'} (cupón {'>' if coupon>ytm else '<' if coupon<ytm else '='} YTM) |
            | **Dur. Macaulay** | {D_mac:.4f} años | Tiempo promedio ponderado para recuperar la inversión |
            | **Dur. Modificada** | {D_mod:.4f} | Un alza de 100 pb reduce el precio ≈ {D_mod:.2f}% |
            | **Convexidad** | {C:.4f} | {'Alta: el bono gana más de lo esperado ante bajas y pierde menos ante alzas' if C>100 else 'Moderada: corrección de segundo orden relevante en shocks grandes'} |
            """)

        # ── Tabla de sensibilidad ──────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("② Sensibilidad ante Shocks de Tasa", COLORS["rose"])

        shocks = [-200, -100, -50, 50, 100, 200]
        rows   = []
        for s in shocks:
            pl, pd_, pe = bond.sensitivity(s)
            rows.append({
                "Shock (pb)":        s,
                "Precio Lineal ($)":  f"${pl:.2f}  ({(pl/P0-1)*100:+.2f}%)",
                "D* + Convex ($)":    f"${pd_:.2f}  ({(pd_/P0-1)*100:+.2f}%)",
                "Reprice Exacto ($)": f"${pe:.2f}  ({(pe/P0-1)*100:+.2f}%)",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Gráfico sensibilidad ───────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("③ Gráfico de Sensibilidad — Tres Métodos", COLORS.get("violet","#A78BFA"))
        st.plotly_chart(fig_sensitivity(bond, shocks), use_container_width=True)

        # ── Fórmulas ───────────────────────────────────────
        with st.expander("📐 Fórmulas utilizadas"):
            st.latex(r"D_{Mac} = \frac{\displaystyle\sum_{t} t_p \cdot \frac{CF_t}{(1+y/m)^{t_p}}}{P \cdot m}")
            st.latex(r"D^* = \frac{D_{Mac}}{1 + y/m}")
            st.latex(r"C = \frac{\displaystyle\sum_{t} t_p(t_p+1)\cdot\frac{CF_t}{(1+y/m)^{t_p+2}}}{P \cdot m^2}")
            st.latex(r"\frac{\Delta P}{P} \approx -D^* \cdot \Delta y + \frac{1}{2}\,C\cdot(\Delta y)^2")