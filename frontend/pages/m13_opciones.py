"""
pages/m13_opciones.py — Módulo 13: Opciones
Streamlit + Plotly | scipy · numpy
Black-Scholes · 5 Greeks · Volatilidad Implícita · Paridad Put-Call
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import newton
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.theme import plotly_base, COLORS
from utils.dynamic_tickers import get_tickers, render_portafolio_badge


# ══════════════════════════════════════════════════════════════
# CLASE OptionPricer — Black-Scholes + Greeks
# ══════════════════════════════════════════════════════════════

class OptionPricer:
    """
    Valoración de opciones europeas con Black-Scholes.
    
    Parámetros:
    -----------
    S     : float — precio del subyacente
    K     : float — strike
    T     : float — tiempo al vencimiento (años)
    r     : float — tasa libre de riesgo (anual)
    sigma : float — volatilidad (anual)
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        # Precalcular d1 y d2
        self.d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)
    
    # ── Black-Scholes ──────────────────────────────────────
    def call_price(self) -> float:
        """Precio de una call europea."""
        return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
    
    def put_price(self) -> float:
        """Precio de una put europea."""
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)
    
    # ── Greeks ─────────────────────────────────────────────
    def delta_call(self) -> float:
        return norm.cdf(self.d1)
    
    def delta_put(self) -> float:
        return norm.cdf(self.d1) - 1.0
    
    def gamma(self) -> float:
        """Gamma es igual para call y put."""
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float:
        """Vega es igual para call y put (retorna en %)."""
        return self.S * np.sqrt(self.T) * norm.pdf(self.d1) / 100  # dividir 100 para escalar
    
    def theta_call(self) -> float:
        """Theta call (por día)."""
        t1 = -self.S * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T))
        t2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        return (t1 + t2) / 365  # convertir a diario
    
    def theta_put(self) -> float:
        """Theta put (por día)."""
        t1 = -self.S * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T))
        t2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        return (t1 + t2) / 365
    
    def rho_call(self) -> float:
        """Rho call (por 1% cambio en r)."""
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
    
    def rho_put(self) -> float:
        """Rho put (por 1% cambio en r)."""
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100
    
    # ── Paridad Put-Call ───────────────────────────────────
    def put_call_parity_check(self) -> dict:
        """
        Verifica C - P = S - K·e^(-rT)
        """
        C = self.call_price()
        P = self.put_price()
        lhs = C - P
        rhs = self.S - self.K * np.exp(-self.r * self.T)
        error = abs(lhs - rhs)
        return {
            "call_price": C,
            "put_price": P,
            "lhs": lhs,
            "rhs": rhs,
            "error": error,
            "holds": error < 1e-6,
        }


# ══════════════════════════════════════════════════════════════
# VOLATILIDAD IMPLÍCITA (Newton-Raphson)
# ══════════════════════════════════════════════════════════════

def implied_volatility(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "call",
    sigma_init: float = 0.3,
) -> float:
    """
    Encuentra σ implícita usando Newton-Raphson.
    """
    def objective(sigma):
        try:
            pricer = OptionPricer(S, K, T, r, sigma)
            if option_type == "call":
                theo = pricer.call_price()
            else:
                theo = pricer.put_price()
            return theo - market_price
        except:
            return 1e10
    
    def vega_fn(sigma):
        try:
            pricer = OptionPricer(S, K, T, r, sigma)
            return pricer.vega() * 100  # reescalar
        except:
            return 1e-10
    
    try:
        sigma_imp = newton(objective, sigma_init, fprime=vega_fn, maxiter=50, tol=1e-6)
        return max(sigma_imp, 0.01)  # evitar negativos
    except:
        return np.nan


# ══════════════════════════════════════════════════════════════
# HELPERS DE LAYOUT
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

def fig_payoff(K: float, tipo: str):
    """Payoff a vencimiento vs spot."""
    spots = np.linspace(K * 0.5, K * 1.5, 200)
    if tipo == "call":
        payoff = np.maximum(spots - K, 0)
        color = COLORS["emerald"]
    else:
        payoff = np.maximum(K - spots, 0)
        color = COLORS["rose"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=payoff, mode="lines",
        name=f"Payoff {tipo.upper()}",
        line=dict(color=color, width=3)))
    fig.add_vline(x=K, line_dash="dot", line_color=COLORS.get("text3","#8896A8"),
                  annotation_text=f"Strike K={K:.2f}")
    
    pb = plotly_base(360)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text=f"Payoff a Vencimiento — {tipo.upper()}",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Precio del Subyacente (S)",
        yaxis_title="Payoff",
    )
    return fig


def fig_price_vs_spot(K: float, T: float, r: float, sigma: float, tipo: str):
    """Precio de la opción hoy vs spot (T fijo)."""
    spots = np.linspace(K * 0.5, K * 1.5, 200)
    prices = []
    intrinsic = []
    
    for S in spots:
        pricer = OptionPricer(S, K, T, r, sigma)
        if tipo == "call":
            prices.append(pricer.call_price())
            intrinsic.append(max(S - K, 0))
        else:
            prices.append(pricer.put_price())
            intrinsic.append(max(K - S, 0))
    
    color = COLORS["emerald"] if tipo == "call" else COLORS["rose"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=intrinsic, mode="lines",
        name="Valor Intrínseco",
        line=dict(color=COLORS.get("text3","#8896A8"), dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=spots, y=prices, mode="lines",
        name=f"Precio {tipo.upper()} (T={T:.2f}y)",
        line=dict(color=color, width=3)))
    fig.add_vline(x=K, line_dash="dot", line_color=COLORS.get("text3","#8896A8"),
                  annotation_text=f"K={K:.2f}")
    
    pb = plotly_base(380)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text=f"Precio vs Spot — {tipo.upper()}",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Precio del Subyacente (S)",
        yaxis_title="Precio de la Opción ($)",
    )
    return fig


def fig_delta_surface(K: float, r: float, sigma: float):
    """Delta vs spot para distintos vencimientos."""
    spots = np.linspace(K * 0.7, K * 1.3, 200)
    vencimientos = [0.01, 0.1, 0.25, 0.5, 1.0]  # años
    
    fig = go.Figure()
    for T in vencimientos:
        deltas = []
        for S in spots:
            pricer = OptionPricer(S, K, T, r, sigma)
            deltas.append(pricer.delta_call())
        fig.add_trace(go.Scatter(x=spots, y=deltas, mode="lines",
            name=f"T = {T:.2f}y",
            line=dict(width=2)))
    
    fig.add_vline(x=K, line_dash="dot", line_color=COLORS.get("text3","#8896A8"),
                  annotation_text=f"ATM (K={K:.2f})")
    
    pb = plotly_base(380)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text="Delta de Call vs Spot — Convergencia a Step Function",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Precio del Subyacente (S)",
        yaxis_title="Delta (Δ)",
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
                Módulo 13
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                Opciones
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
                    color:#8896A8;letter-spacing:0.08em;">
            Black-Scholes · 5 Greeks (Δ, Γ, ν, Θ, ρ) · Volatilidad Implícita · Paridad Put-Call
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["💰 Valoración B-S", "📊 Greeks", "🔬 Vol. Implícita"])

    # ══════════════════════════════════════════════════════
    # TAB 1 — VALORACIÓN BLACK-SCHOLES
    # ══════════════════════════════════════════════════════
    with tab1:
        sec_title("Parámetros de la Opción", COLORS["gold"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            S = st.number_input("Precio Subyacente (S)", 50.0, 500.0, 100.0, 1.0)
            K = st.number_input("Strike (K)", 50.0, 500.0, 100.0, 1.0)
        
        with col2:
            T = st.slider("Vencimiento (años)", 0.01, 3.0, 1.0, 0.01)
            r = st.slider("Tasa libre de riesgo (%)", 0.0, 20.0, 5.0, 0.1) / 100
        
        with col3:
            sigma = st.slider("Volatilidad anual (%)", 5.0, 100.0, 25.0, 1.0) / 100
            tipo = st.selectbox("Tipo", ["call", "put"])
        
        pricer = OptionPricer(S, K, T, r, sigma)
        
        # ── Precios ─────────────────────────────────────────
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        sec_title("① Precios Black-Scholes", COLORS["emerald"])
        
        C = pricer.call_price()
        P = pricer.put_price()
        
        p1, p2, p3 = st.columns(3)
        p1.metric("📈 Call", f"${C:.4f}")
        p2.metric("📉 Put", f"${P:.4f}")
        
        # Valor intrínseco y temporal
        if tipo == "call":
            intrinsic = max(S - K, 0)
            precio = C
        else:
            intrinsic = max(K - S, 0)
            precio = P
        
        time_value = precio - intrinsic
        p3.metric("⏰ Valor Temporal", f"${time_value:.4f}", f"Intrínseco: ${intrinsic:.4f}")
        
        # ── Paridad Put-Call ────────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("② Paridad Put-Call", COLORS.get("sky","#38BDF8"))
        
        pcp = pricer.put_call_parity_check()
        
        col_l, col_r = st.columns(2)
        with col_l:
            st.latex(r"C - P = S - K \cdot e^{-rT}")
            st.write(f"**LHS (C − P):** {pcp['lhs']:.6f}")
            st.write(f"**RHS (S − K·e^(−rT)):** {pcp['rhs']:.6f}")
        
        with col_r:
            st.write(f"**Error:** {pcp['error']:.2e}")
            if pcp['holds']:
                st.success("✅ Paridad verificada (error < 1e-6)")
            else:
                st.warning("⚠️ Error mayor al esperado")
        
        # ── Gráficos ────────────────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("③ Payoff y Precio", COLORS.get("violet","#A78BFA"))
        
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.plotly_chart(fig_payoff(K, tipo), use_container_width=True)
        with col_r2:
            st.plotly_chart(fig_price_vs_spot(K, T, r, sigma, tipo), use_container_width=True)

    # ══════════════════════════════════════════════════════
    # TAB 2 — GREEKS
    # ══════════════════════════════════════════════════════
    with tab2:
        sec_title("Las 5 Greeks — Sensibilidades", COLORS["rose"])
        
        if tipo == "call":
            delta = pricer.delta_call()
            theta = pricer.theta_call()
            rho   = pricer.rho_call()
        else:
            delta = pricer.delta_put()
            theta = pricer.theta_put()
            rho   = pricer.rho_put()
        
        gamma = pricer.gamma()
        vega  = pricer.vega()
        
        # ── KPIs ────────────────────────────────────────────
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Δ Delta", f"{delta:.4f}", "∂V/∂S")
        g2.metric("Γ Gamma", f"{gamma:.6f}", "∂²V/∂S²")
        g3.metric("ν Vega", f"{vega:.4f}", "∂V/∂σ (por 1%)")
        g4.metric("Θ Theta", f"{theta:.4f}", "∂V/∂t (diario)")
        g5.metric("ρ Rho", f"{rho:.4f}", "∂V/∂r (por 1%)")
        
        # ── Interpretaciones ────────────────────────────────
        with st.expander("📖 Interpretación de Greeks"):
            st.markdown(f"""
            | Greek | Valor | Significado |
            |-------|-------|-------------|
            | **Delta (Δ)** | {delta:.4f} | Si S sube $1, la opción {'sube' if delta>0 else 'baja'} ${abs(delta):.4f} |
            | **Gamma (Γ)** | {gamma:.6f} | Curvatura del delta — qué tan rápido cambia Δ con S |
            | **Vega (ν)** | {vega:.4f} | Si σ sube 1%, la opción sube ${vega:.4f} |
            | **Theta (Θ)** | {theta:.4f} | Decay temporal — la opción {'pierde' if theta<0 else 'gana'} ${abs(theta):.4f} por día |
            | **Rho (ρ)** | {rho:.4f} | Si r sube 1%, la opción {'sube' if rho>0 else 'baja'} ${abs(rho):.4f} |
            """)
        
        # ── Tabla resumen ───────────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("Tabla de Greeks — Call vs Put", COLORS["gold"])
        
        df_greeks = pd.DataFrame({
            "Greek": ["Delta (Δ)", "Gamma (Γ)", "Vega (ν)", "Theta (Θ)", "Rho (ρ)"],
            "Call": [
                f"{pricer.delta_call():.4f}",
                f"{pricer.gamma():.6f}",
                f"{pricer.vega():.4f}",
                f"{pricer.theta_call():.4f}",
                f"{pricer.rho_call():.4f}",
            ],
            "Put": [
                f"{pricer.delta_put():.4f}",
                f"{pricer.gamma():.6f}",
                f"{pricer.vega():.4f}",
                f"{pricer.theta_put():.4f}",
                f"{pricer.rho_put():.4f}",
            ],
        })
        st.dataframe(df_greeks, use_container_width=True, hide_index=True)
        
        # ── Gráfico Delta Surface ───────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("Delta vs Spot — Convergencia a Step Function", COLORS.get("violet","#A78BFA"))
        st.plotly_chart(fig_delta_surface(K, r, sigma), use_container_width=True)

    # ══════════════════════════════════════════════════════
    # TAB 3 — VOLATILIDAD IMPLÍCITA
    # ══════════════════════════════════════════════════════
    with tab3:
        sec_title("Volatilidad Implícita — Newton-Raphson", COLORS.get("sky","#38BDF8"))
        
        st.info("🔬 Dado un precio de mercado observado, encuentra la σ implícita que hace que Black-Scholes iguale ese precio.")
        
        market_price = st.number_input(
            f"Precio de mercado observado ({tipo.upper()})",
            0.01, 500.0,
            pricer.call_price() if tipo == "call" else pricer.put_price(),
            0.01,
        )
        
        if st.button("🔍 Calcular σ implícita", type="primary"):
            with st.spinner("Resolviendo Newton-Raphson..."):
                sigma_imp = implied_volatility(market_price, S, K, T, r, tipo, sigma_init=sigma)
            
            if np.isnan(sigma_imp):
                st.error("❌ No se pudo converger — verifica que el precio de mercado sea razonable.")
            else:
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                sec_title("Resultado", COLORS["emerald"])
                
                i1, i2, i3 = st.columns(3)
                i1.metric("σ Implícita", f"{sigma_imp*100:.2f}%")
                i2.metric("σ Histórica (input)", f"{sigma*100:.2f}%")
                spread = (sigma_imp - sigma) * 100
                i3.metric("Spread", f"{spread:+.2f}%", f"{'Mercado espera más vol' if spread>0 else 'Mercado espera menos vol'}")
                
                with st.expander("📊 Interpretación del Spread"):
                    if abs(spread) < 2:
                        st.success("✅ σ implícita ≈ σ histórica — el mercado está alineado con el pasado reciente.")
                    elif spread > 0:
                        st.warning(f"⚠️ σ implícita {abs(spread):.2f}% mayor — el mercado anticipa más volatilidad (eventos próximos, incertidumbre).")
                    else:
                        st.info(f"ℹ️ σ implícita {abs(spread):.2f}% menor — el mercado espera calma relativa.")