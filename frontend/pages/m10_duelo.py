"""
frontend/pages/m10_duelo.py — Módulo 10: Duelo de Portafolios
Dos portafolios con activos independientes enfrentados en métricas de riesgo.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
 
from utils.theme import plotly_base, COLORS
from utils.dynamic_tickers import get_tickers, get_ticker_colors, get_nombre, get_sector
 
BACKEND_URL = "http://localhost:8002"
 
 
def get_sp500_list() -> list[dict]:
    """Carga la lista completa del S&P 500 para el selector."""
    try:
        r = requests.get(f"{BACKEND_URL}/sp500/tickers", timeout=30)
        r.raise_for_status()
        return r.json()["tickers"]
    except Exception:
        return []
 
 
def post_duelo(payload: dict) -> dict:
    r = requests.post(f"{BACKEND_URL}/duelo", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()
 
 
def render_metric_row(metrica, val_a, val_b, ganador, fmt=".4f"):
    col_a  = "#1A6B4A" if ganador == "A" else "#8896A8"
    col_b  = "#1A6B4A" if ganador == "B" else "#8896A8"
    badge_a = "✓" if ganador == "A" else ""
    badge_b = "✓" if ganador == "B" else ""
    emp    = "🤝 Empate" if ganador == "empate" else ""
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
        r=vals_a + [vals_a[0]], theta=categorias + [categorias[0]],
        fill="toself", fillcolor="rgba(26,79,110,0.15)",
        line=dict(color=COLORS["sky"], width=2), name="Portafolio A",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b + [vals_b[0]], theta=categorias + [categorias[0]],
        fill="toself", fillcolor="rgba(139,42,42,0.15)",
        line=dict(color=COLORS["rose"], width=2), name="Portafolio B",
    ))
    fig.update_layout(
        **plotly_base(380),
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(family="IBM Plex Mono", size=10,
                                           color=COLORS["text2"])),
        ),
        title=dict(text="Perfil de riesgo — Portafolio A vs B",
                   font=dict(size=12, color=COLORS["text"], family="Playfair Display")),
    )
    return fig
 
 
def render_empresa_chips(tickers: list[str], color_borde: str):
    """Muestra chips con ticker + nombre + sector para cada empresa."""
    chips_html = ""
    for t in tickers:
        nombre = get_nombre(t)
        sector = get_sector(t)
        chips_html += f"""
        <div style="background:#F4F6FB;border:1px solid #D8DDE8;border-left:3px solid {color_borde};
        border-radius:6px;padding:.4rem .7rem;margin-bottom:.3rem">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:.7rem;
            font-weight:700;color:{color_borde}">{t}</span>
            <span style="font-size:.68rem;color:#4A5568;margin-left:.5rem">{nombre}</span>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.5rem;
            color:#8896A8;letter-spacing:.06em;text-transform:uppercase;margin-top:1px">{sector}</div>
        </div>"""
    st.markdown(chips_html, unsafe_allow_html=True)
 
 
def show():
    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">Módulo 10</span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;">Duelo de Portafolios</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;color:#8896A8;">
            Dos portafolios independientes · Seis métricas · Un veredicto
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("""
    <div style="font-size:.83rem;color:#4A5568;margin-bottom:1.5rem;line-height:1.65;max-width:680px">
    Arma dos portafolios con empresas <strong>completamente distintas</strong> del S&P 500
    y enfréntales en retorno, Sharpe, volatilidad, VaR, alpha y drawdown.
    </div>
    """, unsafe_allow_html=True)
 
    # ── Cargar lista S&P 500 ──
    with st.spinner("Cargando lista del S&P 500..."):
        sp500 = get_sp500_list()
 
    if not sp500:
        st.error("No se pudo cargar la lista del S&P 500. Verifica que el backend esté activo.")
        return
 
    opciones = {f"{i['ticker']} — {i['name']}": i["ticker"] for i in sp500}
    opciones_list = list(opciones.keys())
 
    col_a, col_sep, col_b = st.columns([5, 1, 5])
 
    # ── Portafolio A ──
    with col_a:
        st.markdown("""
        <div style="background:#EEF4FB;border:1px solid #C4CBD8;border-top:3px solid #1A4F6E;
        border-radius:8px;padding:1rem;margin-bottom:.75rem">
            <div style="font-family:'Playfair Display',serif;font-size:1rem;
            font-weight:700;color:#1A4F6E">Portafolio A</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
            color:#8896A8;margin-top:2px">Selecciona 1–5 empresas</div>
        </div>
        """, unsafe_allow_html=True)
 
        sel_a_raw = st.multiselect(
            "Empresas A",
            options=opciones_list,
            default=[],
            max_selections=5,
            key="duelo_sel_a",
            label_visibility="collapsed",
            placeholder="Busca y selecciona empresas para el Portafolio A...",
        )
        tickers_a = [opciones[s] for s in sel_a_raw]
 
        if tickers_a:
            render_empresa_chips(tickers_a, "#1A4F6E")
            st.markdown("<div style='margin-top:.75rem'></div>", unsafe_allow_html=True)
            pesos_a = []
            peso_eq_a = round(1.0 / len(tickers_a), 4)
            for t in tickers_a:
                p = st.number_input(
                    f"Peso {t} (A)", 0.0, 1.0, peso_eq_a, 0.01,
                    key=f"da_{t}", format="%.2f",
                )
                pesos_a.append(p)
            suma_a = sum(pesos_a)
            ca = "#1A6B4A" if abs(suma_a - 1.0) <= 0.01 else "#8B2A2A"
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.65rem;'
                f'text-align:right;color:{ca}">Σ = {suma_a:.4f} '
                f'{"✓" if abs(suma_a-1.0)<=0.01 else "✗"}</div>',
                unsafe_allow_html=True,
            )
        else:
            pesos_a = []
            suma_a = 0.0
 
    with col_sep:
        st.markdown(
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:100%;font-family:\"Playfair Display\",serif;font-size:1.5rem;"
            "color:#8896A8;padding-top:5rem'>VS</div>",
            unsafe_allow_html=True,
        )
 
    # ── Portafolio B ──
    with col_b:
        st.markdown("""
        <div style="background:#FAF0F0;border:1px solid #C4CBD8;border-top:3px solid #8B2A2A;
        border-radius:8px;padding:1rem;margin-bottom:.75rem">
            <div style="font-family:'Playfair Display',serif;font-size:1rem;
            font-weight:700;color:#8B2A2A">Portafolio B</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
            color:#8896A8;margin-top:2px">Selecciona 1–5 empresas distintas</div>
        </div>
        """, unsafe_allow_html=True)
 
        sel_b_raw = st.multiselect(
            "Empresas B",
            options=opciones_list,
            default=[],
            max_selections=5,
            key="duelo_sel_b",
            label_visibility="collapsed",
            placeholder="Busca y selecciona empresas para el Portafolio B...",
        )
        tickers_b = [opciones[s] for s in sel_b_raw]
 
        if tickers_b:
            render_empresa_chips(tickers_b, "#8B2A2A")
            st.markdown("<div style='margin-top:.75rem'></div>", unsafe_allow_html=True)
            pesos_b = []
            peso_eq_b = round(1.0 / len(tickers_b), 4)
            for t in tickers_b:
                p = st.number_input(
                    f"Peso {t} (B)", 0.0, 1.0, peso_eq_b, 0.01,
                    key=f"db_{t}", format="%.2f",
                )
                pesos_b.append(p)
            suma_b = sum(pesos_b)
            cb = "#1A6B4A" if abs(suma_b - 1.0) <= 0.01 else "#8B2A2A"
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.65rem;'
                f'text-align:right;color:{cb}">Σ = {suma_b:.4f} '
                f'{"✓" if abs(suma_b-1.0)<=0.01 else "✗"}</div>',
                unsafe_allow_html=True,
            )
        else:
            pesos_b = []
            suma_b = 0.0
 
    years = st.slider("Años de historia", 1, 10, 3)
 
    # ── Validaciones ──
    listo = (
        len(tickers_a) >= 1 and len(tickers_b) >= 1
        and abs(suma_a - 1.0) <= 0.01 and abs(suma_b - 1.0) <= 0.01
    )
 
    overlap = set(tickers_a) & set(tickers_b)
    if overlap:
        st.warning(f"⚠️ Los portafolios comparten: {', '.join(overlap)}. "
                   f"Para un duelo justo usa empresas distintas.")
 
    if not listo:
        st.info("Selecciona al menos 1 empresa en cada portafolio y ajusta los pesos a 1.0 para continuar.")
 
    if st.button("⚔️ Iniciar duelo", type="primary",
                 use_container_width=True, disabled=not listo):
        with st.spinner("Calculando métricas y determinando el veredicto..."):
            try:
                payload = {
                    "portafolio_a": {
                        "tickers": tickers_a, "weights": pesos_a,
                        "confidence": 0.95, "years": years,
                    },
                    "portafolio_b": {
                        "tickers": tickers_b, "weights": pesos_b,
                        "confidence": 0.95, "years": years,
                    },
                    "years": years,
                }
                data = post_duelo(payload)
                a    = data["portafolio_a"]
                b    = data["portafolio_b"]
                verd = data["veredicto"]
                col_verd = (COLORS["emerald"] if verd == "A"
                            else COLORS["rose"] if verd == "B"
                            else COLORS["gold"])
 
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
 
                # ── Composición de cada portafolio ──
                comp_col_a, comp_col_b = st.columns(2)
                with comp_col_a:
                    st.markdown("""
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
                    letter-spacing:.14em;text-transform:uppercase;color:#1A4F6E;
                    margin-bottom:.4rem">Composición A</div>
                    """, unsafe_allow_html=True)
                    render_empresa_chips(tickers_a, "#1A4F6E")
                with comp_col_b:
                    st.markdown("""
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
                    letter-spacing:.14em;text-transform:uppercase;color:#8B2A2A;
                    margin-bottom:.4rem">Composición B</div>
                    """, unsafe_allow_html=True)
                    render_empresa_chips(tickers_b, "#8B2A2A")
 
                st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
 
                # ── Métricas ──
                st.markdown("""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
                letter-spacing:.16em;text-transform:uppercase;color:#8896A8;
                margin:1rem 0 .5rem;border-left:2px solid #8B6914;padding-left:8px">
                Comparación métrica a métrica
                </div>
                """, unsafe_allow_html=True)
 
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
 
                metricas = [
                    ("Retorno anual",  a["retorno_anual"],  b["retorno_anual"],
                     a["ganador_metricas"].get("Retorno anual",""), ".2%"),
                    ("Sharpe",         a["sharpe"],          b["sharpe"],
                     a["ganador_metricas"].get("Sharpe",""),        ".4f"),
                    ("Volatilidad",    a["volatilidad"],     b["volatilidad"],
                     a["ganador_metricas"].get("Volatilidad",""),   ".2%"),
                    ("Max Drawdown",   a["max_drawdown"],    b["max_drawdown"],
                     a["ganador_metricas"].get("Max Drawdown",""),  ".2%"),
                    ("VaR 95%",        a["var_95"],           b["var_95"],
                     a["ganador_metricas"].get("VaR 95%",""),       ".4f"),
                    ("Alpha",          a["alpha"],            b["alpha"],
                     a["ganador_metricas"].get("Alpha",""),         ".4f"),
                ]
 
                for met, va, vb, gan, fmt in metricas:
                    render_metric_row(met, va, vb, gan, fmt)
 
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                st.plotly_chart(fig_radar(a, b), use_container_width=True)
 
            except requests.exceptions.HTTPError as e:
                st.error(f"Error: {e.response.json().get('detail', str(e))}")
            except Exception as e:
                st.error(f"Error: {e}")