"""
pages/m14_stress.py — Módulo 14: Stress Testing
Streamlit + Plotly | numpy · pandas
Escenarios extremos · VaR estresado · Sensibilidad por activo
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.client import get_precios
from utils.theme import plotly_base, COLORS
from utils.dynamic_tickers import get_tickers, render_portafolio_badge


# ══════════════════════════════════════════════════════════════
# CLASE StressTester
# ══════════════════════════════════════════════════════════════

class StressTester:
    """
    Aplica escenarios de stress sobre un portafolio.
    
    Escenarios:
    1. Shock de tasa: ±200 bp
    2. Caída del mercado: -20%, -30% (según beta)
    3. Shock de volatilidad: σ × 2
    """
    
    def __init__(self, tickers: list, weights: list, years: int = 3):
        self.tickers = tickers
        self.weights = np.array(weights)
        self.years = years
        
        # Obtener precios y calcular retornos
        self.prices = {}
        self.returns = {}
        
        for ticker in tickers:
            data = get_precios(ticker, years=years)
            closes = [p["close"] for p in data["precios"]]
            self.prices[ticker] = closes[-1]  # último precio
            ret = np.diff(np.log(closes))
            self.returns[ticker] = ret
        
        # Calcular betas (simplificado: correlación con portafolio)
        self.betas = self._compute_betas()
        
        # Métricas base
        self.base_value = sum(self.prices[t] * w for t, w in zip(tickers, weights))
        self.base_vol = self._portfolio_volatility()
        self.base_var_99 = self._var_parametric(0.99)
    
    def _compute_betas(self) -> dict:
        """Estima beta de cada activo vs el portafolio equiponderado."""
        # Crear retorno del portafolio
        all_returns = np.array([self.returns[t] for t in self.tickers])
        port_ret = np.mean(all_returns, axis=0)
        
        betas = {}
        for ticker in self.tickers:
            ret = self.returns[ticker]
            # Alinear longitudes
            min_len = min(len(ret), len(port_ret))
            cov = np.cov(ret[:min_len], port_ret[:min_len])[0, 1]
            var_port = np.var(port_ret[:min_len])
            betas[ticker] = cov / var_port if var_port > 0 else 1.0
        
        return betas
    
    def _portfolio_volatility(self) -> float:
        """Volatilidad anualizada del portafolio."""
        all_returns = np.array([self.returns[t] for t in self.tickers])
        # Alinear longitudes
        min_len = min(len(r) for r in all_returns)
        rets = np.array([r[:min_len] for r in all_returns]).T
        port_ret = rets @ self.weights
        return float(np.std(port_ret) * np.sqrt(252))
    
    def _var_parametric(self, confidence: float) -> float:
        """VaR paramétrico (en $)."""
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)
        return -self.base_value * z * self.base_vol / np.sqrt(252)
    
    # ── Escenarios ─────────────────────────────────────────────
    def apply_rate_shock(self, shock_bp: int) -> dict:
        """
        Escenario 1: Shock de tasa de ±200 bp.
        Simplificación: reduce el precio de todos los activos por el shock.
        """
        shock_pct = shock_bp / 10_000
        new_prices = {t: self.prices[t] * (1 - shock_pct * 0.5) for t in self.tickers}
        new_value = sum(new_prices[t] * w for t, w in zip(self.tickers, self.weights))
        loss = self.base_value - new_value
        loss_pct = (loss / self.base_value) * 100
        
        return {
            "scenario": f"Shock de tasa {shock_bp:+d} bp",
            "new_value": new_value,
            "loss": loss,
            "loss_pct": loss_pct,
            "prices": new_prices,
        }
    
    def apply_market_crash(self, crash_pct: float) -> dict:
        """
        Escenario 2: Caída del mercado -X%.
        Cada activo cae según su beta: ΔRᵢ = βᵢ · shock_market
        """
        shock = crash_pct / 100
        new_prices = {}
        for ticker in self.tickers:
            beta = self.betas[ticker]
            new_prices[ticker] = self.prices[ticker] * (1 + beta * shock)
        
        new_value = sum(new_prices[t] * w for t, w in zip(self.tickers, self.weights))
        loss = self.base_value - new_value
        loss_pct = (loss / self.base_value) * 100
        
        return {
            "scenario": f"Caída del mercado {crash_pct:.0f}%",
            "new_value": new_value,
            "loss": loss,
            "loss_pct": loss_pct,
            "prices": new_prices,
        }
    
    def apply_vol_shock(self, multiplier: float) -> dict:
        """
        Escenario 3: Shock de volatilidad σ → σ × multiplier.
        Recalcula VaR con nueva vol.
        """
        new_vol = self.base_vol * multiplier
        from scipy.stats import norm
        z = norm.ppf(1 - 0.99)
        new_var = -self.base_value * z * new_vol / np.sqrt(252)
        
        var_increase = new_var - self.base_var_99
        var_increase_pct = (var_increase / self.base_var_99) * 100
        
        return {
            "scenario": f"Shock de volatilidad σ×{multiplier:.1f}",
            "base_vol": self.base_vol,
            "new_vol": new_vol,
            "base_var": self.base_var_99,
            "new_var": new_var,
            "var_increase": var_increase,
            "var_increase_pct": var_increase_pct,
        }
    
    def apply_combined(self, rate_bp: int, crash_pct: float, vol_mult: float) -> dict:
        """
        Escenario 4: Combinado (tormenta perfecta).
        """
        # Shock de tasa
        shock_rate = rate_bp / 10_000
        prices_after_rate = {t: self.prices[t] * (1 - shock_rate * 0.5) for t in self.tickers}
        
        # Crash del mercado sobre precios ya ajustados
        shock_mkt = crash_pct / 100
        new_prices = {}
        for ticker in self.tickers:
            beta = self.betas[ticker]
            new_prices[ticker] = prices_after_rate[ticker] * (1 + beta * shock_mkt)
        
        new_value = sum(new_prices[t] * w for t, w in zip(self.tickers, self.weights))
        loss = self.base_value - new_value
        loss_pct = (loss / self.base_value) * 100
        
        # VaR con nueva vol
        new_vol = self.base_vol * vol_mult
        from scipy.stats import norm
        z = norm.ppf(1 - 0.99)
        new_var = -new_value * z * new_vol / np.sqrt(252)
        
        return {
            "scenario": "Combinado (tormenta perfecta)",
            "new_value": new_value,
            "loss": loss,
            "loss_pct": loss_pct,
            "new_vol": new_vol,
            "new_var": new_var,
            "prices": new_prices,
        }


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

def fig_loss_by_scenario(scenarios: list):
    """Bar chart de pérdida por escenario."""
    names = [s["scenario"] for s in scenarios]
    losses = [s.get("loss", 0) for s in scenarios]
    
    colors = [COLORS["rose"] if l > 0 else COLORS["emerald"] for l in losses]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=losses,
        marker_color=colors,
        text=[f"${l:,.0f}" for l in losses],
        textposition="outside",
    ))
    
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS.get("text3","#8896A8"))
    
    pb = plotly_base(400)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text="Pérdida del Portafolio por Escenario",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Escenario",
        yaxis_title="Pérdida ($)",
    )
    return fig


def fig_var_comparison(base_var: float, stressed_vars: dict):
    """Comparación VaR base vs VaR estresado."""
    scenarios = list(stressed_vars.keys())
    vars_stressed = list(stressed_vars.values())
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Base"] + scenarios,
        y=[base_var] + vars_stressed,
        marker_color=[COLORS["emerald"]] + [COLORS["rose"]] * len(scenarios),
        text=[f"${v:,.0f}" for v in [base_var] + vars_stressed],
        textposition="outside",
    ))
    
    pb = plotly_base(380)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text="VaR 99% — Base vs Estresado",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Escenario",
        yaxis_title="VaR 99% ($)",
    )
    return fig


def fig_heatmap_sensitivity(tickers: list, scenarios: list, base_prices: dict):
    """Heatmap de sensibilidad: Δprecio % por activo y escenario."""
    data = []
    for scenario in scenarios:
        if "prices" not in scenario:
            continue
        row = []
        for ticker in tickers:
            new_price = scenario["prices"][ticker]
            base_price = base_prices[ticker]
            pct_change = ((new_price - base_price) / base_price) * 100
            row.append(pct_change)
        data.append(row)
    
    scenario_names = [s["scenario"] for s in scenarios if "prices" in s]
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=tickers,
        y=scenario_names,
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"{v:+.1f}%" for v in row] for row in data],
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="Δ%"),
    ))
    
    pb = plotly_base(400)
    pb["xaxis"]["type"] = "-"
    pb["yaxis"]["type"] = "-"
    fig.update_layout(**pb,
        title=dict(text="Sensibilidad por Activo y Escenario",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
        xaxis_title="Activo",
        yaxis_title="Escenario",
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
                Módulo 14
            </span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;letter-spacing:-0.01em;">
                Stress Testing
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
                    color:#8896A8;letter-spacing:0.08em;">
            Escenarios extremos · VaR estresado · Análisis forward-looking
        </div>
    </div>
    """, unsafe_allow_html=True)

    TICKERS = get_tickers()
    
    if len(TICKERS) < 2:
        st.warning("⚠️ Selecciona al menos 2 activos en el Selector de Activos.")
        return
    
    # ── Configuración ───────────────────────────────────────
    st.info("🔬 **Stress Testing** aplica escenarios hipotéticos extremos para estimar pérdidas potenciales bajo condiciones adversas (enfoque forward-looking, complementario al backtesting de Kupiec).")
    
    with st.expander("⚙️ Configuración del Portafolio"):
        st.write("Pesos equiponderados por defecto. Puedes ajustarlos:")
        weights = []
        cols = st.columns(len(TICKERS))
        for i, ticker in enumerate(TICKERS):
            with cols[i]:
                w = st.number_input(ticker, 0.0, 1.0, 1.0/len(TICKERS), 0.01, key=f"w_{ticker}")
                weights.append(w)
        
        # Normalizar
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Suma de pesos: {total:.2f} — se normalizará a 1.0")
            weights = [w / total for w in weights]
    
    if st.button("🚨 Ejecutar Stress Testing", type="primary"):
        with st.spinner("Aplicando escenarios extremos..."):
            tester = StressTester(TICKERS, weights, years=3)
        
        # ── Métricas base ───────────────────────────────────
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        sec_title("① Métricas Base del Portafolio", COLORS["emerald"])
        
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("💰 Valor", f"${tester.base_value:,.2f}")
        b2.metric("📊 Vol. Anual", f"{tester.base_vol*100:.2f}%")
        b3.metric("⚠️ VaR 99%", f"${tester.base_var_99:,.2f}")
        b4.metric("Activos", len(TICKERS))
        
        # ── Aplicar escenarios ──────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("② Escenarios de Stress", COLORS["rose"])
        
        s1 = tester.apply_rate_shock(200)
        s2 = tester.apply_rate_shock(-200)
        s3 = tester.apply_market_crash(-20)
        s4 = tester.apply_market_crash(-30)
        s5 = tester.apply_vol_shock(2.0)
        s6 = tester.apply_combined(200, -20, 2.0)
        
        scenarios = [s1, s2, s3, s4, s6]  # s5 es solo VaR
        
        # Tabla resumen
        rows = []
        for s in scenarios:
            rows.append({
                "Escenario": s["scenario"],
                "Valor Final": f"${s['new_value']:,.2f}",
                "Pérdida ($)": f"${s['loss']:,.2f}" if s['loss'] > 0 else f"+${-s['loss']:,.2f}",
                "Pérdida (%)": f"{s['loss_pct']:+.2f}%",
            })
        
        df_scenarios = pd.DataFrame(rows)
        st.dataframe(df_scenarios, use_container_width=True, hide_index=True)
        
        # ── Gráfico pérdidas ────────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("③ Pérdida por Escenario", COLORS.get("violet","#A78BFA"))
        st.plotly_chart(fig_loss_by_scenario(scenarios), use_container_width=True)
        
        # ── VaR Estresado ───────────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("④ VaR 99% — Base vs Estresado", COLORS.get("sky","#38BDF8"))
        
        stressed_vars = {
            "σ×2": s5["new_var"],
            "Combinado": s6["new_var"],
        }
        
        st.plotly_chart(fig_var_comparison(tester.base_var_99, stressed_vars), use_container_width=True)
        
        v1, v2, v3 = st.columns(3)
        v1.metric("VaR Base", f"${tester.base_var_99:,.2f}")
        v2.metric("VaR σ×2", f"${s5['new_var']:,.2f}", f"{s5['var_increase_pct']:+.1f}%")
        v3.metric("VaR Combinado", f"${s6['new_var']:,.2f}")
        
        # ── Heatmap sensibilidad ────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        sec_title("⑤ Sensibilidad por Activo", COLORS["gold"])
        st.plotly_chart(fig_heatmap_sensitivity(TICKERS, scenarios, tester.prices), use_container_width=True)
        
        # ── Interpretaciones ────────────────────────────────
        with st.expander("📖 Interpretación de Escenarios"):
            st.markdown(f"""
            **1. Shock de tasa ±200 bp:**
            - Simula un cambio drástico en política monetaria (Fed subiendo/bajando tasas agresivamente)
            - Pérdida {s1['scenario']}: **${s1['loss']:,.2f}** ({s1['loss_pct']:+.2f}%)
            - Pérdida {s2['scenario']}: **${s2['loss']:,.2f}** ({s2['loss_pct']:+.2f}%)
            
            **2. Caída del mercado:**
            - Escenario −20%: corrección severa (crisis moderada)
            - Escenario −30%: crash tipo 2008 o COVID
            - Pérdida proporcional al beta de cada activo
            - Pérdida −20%: **${s3['loss']:,.2f}** ({s3['loss_pct']:+.2f}%)
            - Pérdida −30%: **${s4['loss']:,.2f}** ({s4['loss_pct']:+.2f}%)
            
            **3. Shock de volatilidad (σ×2):**
            - Duplica la volatilidad histórica → mercados en pánico
            - VaR base: **${tester.base_var_99:,.2f}**
            - VaR estresado: **${s5['new_var']:,.2f}** ({s5['var_increase_pct']:+.1f}% más)
            
            **4. Combinado (tormenta perfecta):**
            - Tasa +200bp + Mercado −20% + σ×2 simultáneos
            - Pérdida total: **${s6['loss']:,.2f}** ({s6['loss_pct']:+.2f}%)
            - Este es el **peor escenario** — probabilidad muy baja pero impacto máximo
            """)
        
        with st.expander("🏦 Stress Testing vs Backtesting"):
            st.markdown("""
            | Aspecto | Backtesting (Kupiec - M5) | Stress Testing (M14) |
            |---------|---------------------------|----------------------|
            | **Enfoque** | Histórico (backward-looking) | Hipotético (forward-looking) |
            | **Pregunta** | ¿El VaR estimado fue preciso? | ¿Qué pasa en escenarios extremos? |
            | **Input** | Pérdidas reales observadas | Shocks sintéticos |
            | **Objetivo** | Validar modelo VaR | Estimar pérdidas en crisis |
            | **Basilea III** | Obligatorio para entidades | Obligatorio para entidades |
            
            **Ambos son complementarios:** Kupiec valida el pasado, Stress Testing prepara para el futuro.
            """)