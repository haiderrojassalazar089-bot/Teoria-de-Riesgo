"""
frontend/pages/m10_duelo.py — Módulo 10: Duelo de Portafolios
Dos portafolios enfrentados en todas las métricas de riesgo.
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


def post_duelo(payload: dict) -> dict:
    r = requests.post(f"{BACKEND_URL}/duelo", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def render_metric_row(metrica, val_a, val_b, ganador, fmt=".4f"):
    col_a = "#1A6B4A" if ganador == "A" else "#8896A8"
    col_b = "#1A6B4A" if ganador == "B" else "#8896A8"
    badge_a = "✓" if ganador == "A" else ""
    badge_b = "✓" if ganador == "B" else ""
    emp = "🤝 Empate" if ganador == "empate" else ""
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 140px 1fr;gap:.5rem;
    align-items:center;padding:.5rem 0;border-bottom:1px solid #F4F6FB">
        <div style="text-align:right;font-family:'Playfair Display',serif;
        font-size:1rem;font-weight:600;color:{col_a}">{val_a:{fmt}} {badge_a}</div>
        <div style="text-align:center;font-family:'IBM Plex Mono',monospace;
        font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;color:#8896A8">
        {metrica}<br><span style="font-size:.62rem;color:#8B6914">{emp}</span></div>
        <div style="font-family:'Playfair Display',serif;font-size:1rem;
        font-weight:600;color:{col_b}">{badge_b} {val_b:{fmt}}</div>
    </div>
    """, unsafe_allow_html=True)


def fig_radar(a: dict, b: dict) -> go.Figure:
    categorias = ["Retorno", "Sharpe", "Bajo riesgo", "Bajo VaR", "Alpha", "Bajo drawdown"]

    def norm(val, higher_better=True):
        return max(0, min(1, (val + 0.5) if higher_better else (0.5 - val)))

    vals_a = [
        norm(a["retorno_anual"]),
        norm(a["sharpe"] / 3),
        norm(-a["volatilidad"] * 5),
        norm(-a["var_95"] * 10),
        norm(a["alpha"] * 20),
        norm(-abs(a["max_drawdown"])),
    ]
    vals_b = [
        norm(b["retorno_anual"]),
        norm(b["sharpe"] / 3),
        norm(-b["volatilidad"] * 5),
        norm(-b["var_95"] * 10),
        norm(b["alpha"] * 20),
        norm(-abs(b["max_drawdown"])),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_a + [vals_a[0]],
        theta=categorias + [categorias[0]],
        fill="toself",
        fillcolor="rgba(26,79,110,0.15)",
        line=dict(color=COLORS["sky"], width=2),
        name="Portafolio A",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b + [vals_b[0]],
        theta=categorias + [categorias[0]],
        fill="toself",
        fillcolor="rgba(139,42,42,0.15)",
        line=dict(color=COLORS["rose"], width=2),
        name="Portafolio B",
    ))
    fig.update_layout(
        **plotly_base(380),
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(family="IBM Plex Mono", size=10, color=COLORS["text2"])),
        ),
        title=dict(text="Perfil de riesgo — Portafolio A vs B",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
    )
    return fig


def show():
    tickers = get_tickers()
    n = len(tickers)

    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">Módulo 10</span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;">Duelo de Portafolios</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;color:#8896A8;">
            Dos portafolios · Seis métricas · Un veredicto
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:.83rem;color:#4A5568;margin-bottom:1.5rem;line-height:1.65;max-width:680px">
    Configura dos portafolios distintos con los activos seleccionados y distribuciones de peso diferentes.
    El sistema los enfrentará en retorno, Sharpe, volatilidad, VaR, alpha y drawdown.
    </div>
    """, unsafe_allow_html=True)

    peso_eq = round(1.0 / n, 4)
    colors_tick = ["#8B6914","#1A6B4A","#1A4F6E","#3D2F6B","#8B2A2A",
                   "#2266A0","#5A3E7A","#2A6B4A","#8B5A14","#6B2A1A"]

    col_a, col_sep, col_b = st.columns([5, 1, 5])

    with col_a:
        st.markdown("""
        <div style="background:#EEF4FB;border:1px solid #C4CBD8;border-top:3px solid #1A4F6E;
        border-radius:8px;padding:1rem;margin-bottom:.75rem">
            <div style="font-family:'Playfair Display',serif;font-size:1rem;
            font-weight:700;color:#1A4F6E">Portafolio A</div>
        </div>
        """, unsafe_allow_html=True)
        pesos_a = []
        for i, t in enumerate(tickers):
            p = st.number_input(f"{t} (A)", 0.0, 1.0, peso_eq, 0.01,
                                key=f"da_{t}", format="%.2f")
            pesos_a.append(p)
        suma_a = sum(pesos_a)
        ca = "#1A6B4A" if abs(suma_a - 1.0) <= 0.01 else "#8B2A2A"
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.65rem;'
                    f'text-align:right;color:{ca}">Σ = {suma_a:.4f} {"✓" if abs(suma_a-1.0)<=0.01 else "✗"}</div>',
                    unsafe_allow_html=True)

    with col_sep:
        st.markdown("<div style='display:flex;align-items:center;justify-content:center;"
                    "height:100%;font-family:\"Playfair Display\",serif;font-size:1.5rem;"
                    "color:#8896A8;padding-top:3rem'>VS</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style="background:#FAF0F0;border:1px solid #C4CBD8;border-top:3px solid #8B2A2A;
        border-radius:8px;padding:1rem;margin-bottom:.75rem">
            <div style="font-family:'Playfair Display',serif;font-size:1rem;
            font-weight:700;color:#8B2A2A">Portafolio B</div>
        </div>
        """, unsafe_allow_html=True)
        pesos_b = []
        for i, t in enumerate(tickers):
            # Pesos inversos como sugerencia para B
            p_sug = round(1.0 / n, 4)
            p = st.number_input(f"{t} (B)", 0.0, 1.0, p_sug, 0.01,
                                key=f"db_{t}", format="%.2f")
            pesos_b.append(p)
        suma_b = sum(pesos_b)
        cb = "#1A6B4A" if abs(suma_b - 1.0) <= 0.01 else "#8B2A2A"
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.65rem;'
                    f'text-align:right;color:{cb}">Σ = {suma_b:.4f} {"✓" if abs(suma_b-1.0)<=0.01 else "✗"}</div>',
                    unsafe_allow_html=True)

    years = st.slider("Años de historia", 1, 10, 3)

    if st.button("⚔️ Iniciar duelo", type="primary", use_container_width=True):
        if abs(suma_a - 1.0) > 0.01:
            st.error("Los pesos del Portafolio A no suman 1.0")
            return
        if abs(suma_b - 1.0) > 0.01:
            st.error("Los pesos del Portafolio B no suman 1.0")
            return

        with st.spinner("Calculando métricas y determinando el veredicto..."):
            try:
                payload = {
                    "portafolio_a": {"tickers": tickers, "weights": pesos_a,
                                     "confidence": 0.95, "years": years},
                    "portafolio_b": {"tickers": tickers, "weights": pesos_b,
                                     "confidence": 0.95, "years": years},
                    "years": years,
                }
                data = post_duelo(payload)
                a = data["portafolio_a"]
                b = data["portafolio_b"]
                verd = data["veredicto"]
                col_verd = COLORS["emerald"] if verd == "A" else COLORS["rose"] if verd == "B" else COLORS["gold"]

                # ── Banner veredicto ──
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1A2035,#2A3550);
                border-radius:10px;padding:1.5rem;text-align:center;margin:1rem 0">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
                    letter-spacing:.22em;text-transform:uppercase;color:#8896A8;margin-bottom:.5rem">
                    Veredicto final</div>
                    <div style="font-family:'Playfair Display',serif;font-size:2rem;
                    font-weight:700;color:{col_verd}">
                    {"🏆 Portafolio A" if verd=="A" else "🏆 Portafolio B" if verd=="B" else "🤝 Empate"}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.7rem;
                    color:#8896A8;margin-top:.4rem">
                    {data['puntos_a']} pts · vs · {data['puntos_b']} pts</div>
                    <div style="font-size:.8rem;color:#C4CBD8;margin-top:.5rem;
                    font-style:italic">{data['resumen']}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Métricas ──
                st.markdown("""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
                letter-spacing:.16em;text-transform:uppercase;color:#8896A8;
                margin:1rem 0 .5rem;border-left:2px solid #8B6914;padding-left:8px">
                Comparación métrica a métrica
                </div>
                """, unsafe_allow_html=True)

                metricas = [
                    ("Retorno anual",  a["retorno_anual"],  b["retorno_anual"],  a["ganador_metricas"].get("Retorno anual",""), ".2%"),
                    ("Sharpe",         a["sharpe"],          b["sharpe"],          a["ganador_metricas"].get("Sharpe",""),        ".4f"),
                    ("Volatilidad",    a["volatilidad"],     b["volatilidad"],     a["ganador_metricas"].get("Volatilidad",""),   ".2%"),
                    ("Max Drawdown",   a["max_drawdown"],    b["max_drawdown"],    a["ganador_metricas"].get("Max Drawdown",""),  ".2%"),
                    ("VaR 95%",        a["var_95"],           b["var_95"],           a["ganador_metricas"].get("VaR 95%",""),       ".4f"),
                    ("Alpha",          a["alpha"],            b["alpha"],            a["ganador_metricas"].get("Alpha",""),         ".4f"),
                ]

                st.markdown("""
                <div style="display:grid;grid-template-columns:1fr 140px 1fr;gap:.5rem;
                padding:.4rem 0;margin-bottom:.25rem">
                    <div style="text-align:right;font-family:'IBM Plex Mono',monospace;
                    font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;
                    color:#1A4F6E">PORTAFOLIO A</div>
                    <div></div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.58rem;
                    letter-spacing:.12em;text-transform:uppercase;color:#8B2A2A">PORTAFOLIO B</div>
                </div>
                """, unsafe_allow_html=True)

                for met, va, vb, gan, fmt in metricas:
                    render_metric_row(met, va, vb, gan, fmt)

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

                # ── Radar ──
                st.plotly_chart(fig_radar(a, b), use_container_width=True)

            except requests.exceptions.HTTPError as e:
                st.error(f"Error: {e.response.json().get('detail', str(e))}")
            except Exception as e:
                st.error(f"Error: {e}")