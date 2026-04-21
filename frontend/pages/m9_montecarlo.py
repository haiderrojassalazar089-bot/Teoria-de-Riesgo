"""
frontend/pages/m9_montecarlo.py — Módulo 9: Monte Carlo Visual
1,000 trayectorias simuladas del portafolio con percentiles animados.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests

from utils.theme import plotly_base, COLORS

BACKEND_URL = "http://localhost:8002"


def get_tickers():
    return st.session_state.get("tickers_seleccionados", [])


def post_montecarlo(payload: dict) -> dict:
    r = requests.post(f"{BACKEND_URL}/montecarlo", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def fig_trayectorias(data: dict, tickers: list[str]) -> go.Figure:
    fechas = data["fechas_sim"]
    trays  = data["trayectorias"]
    p5     = data["percentil_5"]
    p50    = data["percentil_50"]
    p95    = data["percentil_95"]

    fig = go.Figure()

    # Banda IC 90%
    fig.add_trace(go.Scatter(
        x=fechas + fechas[::-1],
        y=p95 + p5[::-1],
        fill="toself",
        fillcolor="rgba(139,105,20,0.07)",
        line=dict(width=0),
        name="IC 90%",
        hoverinfo="skip",
    ))

    # Trayectorias individuales (muestra)
    for i, tray in enumerate(trays[:80]):
        final = tray[-1]
        col = "rgba(139,105,20,0.15)" if final >= 100 else "rgba(139,42,42,0.12)"
        fig.add_trace(go.Scatter(
            x=fechas, y=tray,
            line=dict(color=col, width=0.7),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Percentiles
    fig.add_trace(go.Scatter(x=fechas, y=p95, name="Percentil 95%",
        line=dict(color=COLORS["emerald"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=fechas, y=p50, name="Mediana (P50)",
        line=dict(color=COLORS["gold"], width=2.5)))
    fig.add_trace(go.Scatter(x=fechas, y=p5, name="Percentil 5%",
        line=dict(color=COLORS["rose"], width=2, dash="dot")))

    # Línea base 100
    fig.add_hline(y=100, line=dict(color=COLORS["border2"], width=1, dash="dash"),
                  annotation_text="Valor inicial", annotation_font=dict(size=9))

    fig.update_layout(
        **plotly_base(480),
        title=dict(
            text=f"Monte Carlo · {data['n_simulations']:,} simulaciones · {data['horizon_days']} días",
            font=dict(size=13, color=COLORS["text"], family="Playfair Display"),
        ),
        yaxis_title="Valor del portafolio (base 100)",
    )
    return fig


def show():
    tickers = get_tickers()

    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">Módulo 09</span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;">Monte Carlo Visual</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;color:#8896A8;">
            Simulación estocástica · Trayectorias futuras · Percentiles · Probabilidad de pérdida
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Controles ──
    col1, col2, col3 = st.columns(3)
    with col1:
        horizonte = st.selectbox("Horizonte de simulación",
            ["3 meses (63 días)", "6 meses (126 días)", "1 año (252 días)", "2 años (504 días)"],
            index=2)
        horizonte_map = {
            "3 meses (63 días)": 63,
            "6 meses (126 días)": 126,
            "1 año (252 días)": 252,
            "2 años (504 días)": 504,
        }
        horizon_days = horizonte_map[horizonte]

    with col2:
        n_sim = st.selectbox("Número de simulaciones", [200, 500, 1000], index=1)

    with col3:
        years_hist = st.slider("Años de historia para calibrar", 1, 10, 3)

    # ── Pesos ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;letter-spacing:0.16em;
    text-transform:uppercase;color:#8896A8;margin:.75rem 0 .5rem;
    border-left:2px solid #8B6914;padding-left:8px">Pesos del portafolio</div>
    """, unsafe_allow_html=True)

    n = len(tickers)
    peso_default = round(1.0 / n, 4)
    cols_pesos = st.columns(n)
    pesos = []
    colors_tick = ["#8B6914","#1A6B4A","#1A4F6E","#3D2F6B","#8B2A2A",
                   "#2266A0","#5A3E7A","#2A6B4A","#8B5A14","#6B2A1A"]
    for i, ticker in enumerate(tickers):
        with cols_pesos[i]:
            p = st.number_input(
                ticker,
                min_value=0.0, max_value=1.0,
                value=peso_default, step=0.01,
                key=f"mc_w_{ticker}",
                format="%.2f",
            )
            pesos.append(p)

    suma = sum(pesos)
    color_suma = "#1A6B4A" if abs(suma - 1.0) <= 0.01 else "#8B2A2A"
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.68rem;
    text-align:right;color:{color_suma};margin-bottom:.5rem">
    Σ = {suma:.4f} {"✓" if abs(suma-1.0)<=0.01 else "✗ — debe ser 1.0"}
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶ Simular trayectorias", type="primary", use_container_width=True):
        if abs(suma - 1.0) > 0.01:
            st.error(f"Los pesos suman {suma:.4f} — deben sumar 1.0")
            return
        with st.spinner(f"Simulando {n_sim:,} trayectorias para {horizonte}..."):
            try:
                payload = {
                    "tickers": tickers,
                    "weights": pesos,
                    "horizon_days": horizon_days,
                    "n_simulations": n_sim,
                    "years_history": years_hist,
                }
                data = post_montecarlo(payload)

                # ── KPIs ──
                prob_gan = 1 - data["prob_perdida"]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Retorno esperado", f"{data['retorno_esperado']*100:.2f}%",
                          f"Horizonte {horizonte}")
                k2.metric("Probabilidad de ganancia", f"{prob_gan*100:.1f}%",
                          "Simulaciones que superan base 100")
                k3.metric("VaR del horizonte (P5)", f"{data['var_horizonte']*100:.2f}%",
                          "Pérdida máxima al 95%")
                k4.metric("Mediana final (P50)",
                          f"{data['percentil_50'][-1]:.1f}",
                          "Base 100 = inversión inicial")

                st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

                # ── Gráfica principal ──
                st.plotly_chart(fig_trayectorias(data, tickers), use_container_width=True)

                with st.expander("Interpretación — Monte Carlo"):
                    st.markdown(f"""
                    Se simularon **{n_sim:,} escenarios** de rendimientos futuros calibrados
                    con los parámetros históricos del portafolio ({years_hist} años).
                    Cada línea gris es un posible camino del valor de tu inversión.

                    **Percentil 95%:** en el 5% de los mejores escenarios, el portafolio
                    vale **{data['percentil_95'][-1]:.1f}** (base 100 = inversión inicial).

                    **Mediana:** en el escenario típico, el portafolio vale
                    **{data['percentil_50'][-1]:.1f}** al final del horizonte.

                    **Percentil 5%:** en el 5% de los peores escenarios, vale
                    **{data['percentil_5'][-1]:.1f}** — esto es el **VaR del horizonte**.

                    La **probabilidad de pérdida** es **{data['prob_perdida']*100:.1f}%** —
                    porcentaje de simulaciones que terminan por debajo de la inversión inicial.
                    """)

            except requests.exceptions.HTTPError as e:
                st.error(f"Error del backend: {e.response.json().get('detail', str(e))}")
            except Exception as e:
                st.error(f"Error: {e}")