"""
frontend/pages/m11_tiempo.py — Módulo 11: Máquina del Tiempo
"""
from __future__ import annotations
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests

from utils.theme import plotly_base, COLORS

BACKEND_URL = "http://localhost:8002"

TICKER_COLORS = [
    "#8B6914","#1A6B4A","#1A4F6E","#3D2F6B","#8B2A2A",
    "#2266A0","#5A3E7A","#2A6B4A","#8B5A14","#6B2A1A",
]


def get_tickers():
    return st.session_state.get("tickers_seleccionados", [])


def post_maquina(payload: dict) -> dict:
    r = requests.post(f"{BACKEND_URL}/maquina-tiempo", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def fig_tiempo(data: dict, tickers: list[str]) -> go.Figure:
    fechas = data["fechas"]
    fig = go.Figure()
    if data["benchmark_norm"]:
        n_bench = len(data["benchmark_norm"])
        fig.add_trace(go.Scatter(
            x=fechas[:n_bench], y=data["benchmark_norm"],
            name="S&P 500", line=dict(color="#C4CBD8", width=1.5, dash="dot"),
        ))
    for i, ticker in enumerate(tickers):
        if ticker not in data["retornos_norm"]:
            continue
        vals = data["retornos_norm"][ticker]
        col  = TICKER_COLORS[i % len(TICKER_COLORS)]
        es_mejor = ticker == data["mejor_activo"]
        es_peor  = ticker == data["peor_activo"]
        name = f"{ticker} {'🏆' if es_mejor else '📉' if es_peor else ''}"
        fig.add_trace(go.Scatter(
            x=fechas[:len(vals)], y=vals, name=name,
            line=dict(color=col, width=2.5 if (es_mejor or es_peor) else 1.6),
        ))
    fig.add_hline(y=100, line=dict(color=COLORS["border2"], width=1, dash="dash"),
                  annotation_text="Base 100", annotation_font=dict(size=9))
    fig.update_layout(
        **plotly_base(480),
        title=dict(
            text=f"Portafolio en el tiempo · {data['start_date']} → {data['end_date']}",
            font=dict(size=13, color=COLORS["text"], family="Playfair Display"),
        ),
        yaxis_title="Valor normalizado (base 100)",
    )
    return fig


def show():
    tickers = get_tickers()
    hoy = date.today()

    st.markdown("""
    <div style="margin-bottom:2rem;padding-bottom:1.2rem;border-bottom:1px solid #D8DDE8;">
        <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                         color:#8896A8;letter-spacing:0.2em;text-transform:uppercase;">Módulo 11</span>
            <span style="font-family:'Playfair Display',serif;font-size:1.65rem;
                         font-weight:700;color:#1A2035;">Máquina del Tiempo</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.63rem;color:#8896A8;">
            Viaja a cualquier período histórico · Últimos 10 años · Calendarios interactivos
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:.83rem;color:#4A5568;margin-bottom:1.5rem;line-height:1.65;max-width:680px">
    Selecciona un rango de fechas y observa cómo se habría comportado tu portafolio en ese período exacto.
    Incluye comparación contra el S&P 500, estadísticas del período y clasificación de los activos.
    </div>
    """, unsafe_allow_html=True)

    # ── Períodos predefinidos — diseño mejorado ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;letter-spacing:.16em;
    text-transform:uppercase;color:#8896A8;margin-bottom:.8rem;
    border-left:2px solid #8B6914;padding-left:8px">Períodos históricos relevantes</div>
    """, unsafe_allow_html=True)

    periodos = {
        "COVID\n(2020)":            (date(2020, 1, 1),  date(2020, 12, 31), "#8B2A2A", "Caída del mercado"),
        "Recuperación\n2021":       (date(2021, 1, 1),  date(2021, 12, 31), "#1A6B4A", "Rally post-COVID"),
        "Bear Market\n2022":        (date(2022, 1, 1),  date(2022, 12, 31), "#8B2A2A", "Caída por inflación"),
        "Rally IA\n2023":           (date(2023, 1, 1),  date(2023, 12, 31), "#1A6B4A", "Boom inteligencia artificial"),
        "Guerra/Aranceles\n2025":   (date(2025, 1, 1),  hoy,                "#8B2A2A", "Impacto aranceles Trump"),
        "Últimos\n12 meses":        (hoy - timedelta(days=365),  hoy,       "#1A4F6E", "Año reciente"),
        "Últimos\n3 años":          (hoy - timedelta(days=1095), hoy,       "#1A4F6E", "Período largo"),
        "Últimos\n5 años":          (hoy - timedelta(days=1825), hoy,       "#1A4F6E", "Período largo"),
    }

    # Renderizar botones como tarjetas HTML con st.button encima
    cols_p = st.columns(len(periodos))
    for idx, (nombre, (fi, ff, color, desc)) in enumerate(periodos.items()):
        nombre_limpio = nombre.replace("\n", " ")
        activo = (
            st.session_state.get("mt_fecha_ini") == fi and
            st.session_state.get("mt_fecha_fin") == ff
        )
        bg     = f"{color}15" if activo else "#FFFFFF"
        border = f"2px solid {color}" if activo else f"1px solid #D8DDE8"
        text_c = color if activo else "#1A2035"

        with cols_p[idx]:
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:8px;
            padding:.6rem .5rem;text-align:center;margin-bottom:.3rem;
            cursor:pointer;transition:all .15s">
                <div style="font-family:'Playfair Display',serif;font-size:.72rem;
                font-weight:600;color:{text_c};line-height:1.3">
                {nombre.replace(chr(10), '<br>')}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.45rem;
                color:#8896A8;margin-top:3px;line-height:1.3">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("→", key=f"per_{idx}", use_container_width=True,
                         help=f"{nombre_limpio}: {fi} → {ff}"):
                st.session_state["mt_fecha_ini"] = fi
                st.session_state["mt_fecha_fin"] = ff
                st.rerun()

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Calendarios ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;letter-spacing:.16em;
    text-transform:uppercase;color:#8896A8;margin-bottom:.5rem;
    border-left:2px solid #8B6914;padding-left:8px">O selecciona fechas personalizadas</div>
    """, unsafe_allow_html=True)

    fecha_min     = hoy - timedelta(days=3650)
    fecha_ini_def = hoy - timedelta(days=365 * 3)
    fecha_fin_def = hoy

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fecha_ini = st.date_input("📅 Fecha de inicio",
            value=st.session_state.get("mt_fecha_ini", fecha_ini_def),
            min_value=fecha_min, max_value=hoy - timedelta(days=31))
    with col_f2:
        fecha_fin = st.date_input("📅 Fecha de fin",
            value=st.session_state.get("mt_fecha_fin", fecha_fin_def),
            min_value=fecha_min + timedelta(days=31), max_value=hoy)

    st.session_state["mt_fecha_ini"] = fecha_ini
    st.session_state["mt_fecha_fin"] = fecha_fin
    delta_dias = (fecha_fin - fecha_ini).days

    if fecha_fin <= fecha_ini:
        st.error("La fecha de fin debe ser posterior a la fecha de inicio.")
        return
    if delta_dias < 30:
        st.error("El rango mínimo es 30 días.")
        return

    st.markdown(f"""
    <div style="background:#F4F6FB;border:1px solid #D8DDE8;border-radius:8px;
    padding:.8rem 1.2rem;margin:.6rem 0 1rem;
    display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem">
        <div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.52rem;
            color:#8896A8;letter-spacing:.1em;text-transform:uppercase">Período seleccionado</div>
            <div style="font-family:'Playfair Display',serif;font-size:1rem;
            font-weight:700;color:#1A2035;margin-top:2px">
            {fecha_ini.strftime('%d %b %Y')} → {fecha_fin.strftime('%d %b %Y')}
            </div>
        </div>
        <div style="text-align:right">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#8896A8">
            {delta_dias} días</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
            color:#8B6914;font-weight:600">{delta_dias//30} meses aprox.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⏱ Viajar a este período", type="primary", use_container_width=True):
        with st.spinner(f"Reconstruyendo portafolio {fecha_ini} → {fecha_fin}..."):
            try:
                payload = {"tickers": tickers, "years": 10,
                           "start_date": fecha_ini.strftime("%Y-%m-%d"),
                           "end_date":   fecha_fin.strftime("%Y-%m-%d")}
                data  = post_maquina(payload)
                mejor = data["mejor_activo"]
                peor  = data["peor_activo"]
                est   = data["estadisticas"]

                k1, k2, k3, k4 = st.columns(4)
                k1.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                border-top:3px solid #8B6914;border-radius:8px;padding:1rem">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.5rem;
                    color:#8896A8;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.3rem">
                    Días analizados</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                    font-weight:700;color:#1A2035">{data['n_dias']}</div>
                    <div style="font-size:.62rem;color:#8896A8;margin-top:2px">
                    {data['start_date']} → {data['end_date']}</div>
                </div>
                """, unsafe_allow_html=True)

                ret_mejor = est.get(mejor, {}).get('retorno_total', 0) * 100
                k2.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                border-top:3px solid #1A6B4A;border-radius:8px;padding:1rem">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.5rem;
                    color:#8896A8;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.3rem">
                    🏆 Mejor activo</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                    font-weight:700;color:#1A6B4A">{mejor}</div>
                    <div style="font-size:.62rem;color:#1A6B4A;font-weight:600;margin-top:2px">
                    +{ret_mejor:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                ret_peor = est.get(peor, {}).get('retorno_total', 0) * 100
                k3.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                border-top:3px solid #8B2A2A;border-radius:8px;padding:1rem">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.5rem;
                    color:#8896A8;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.3rem">
                    📉 Peor activo</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                    font-weight:700;color:#8B2A2A">{peor}</div>
                    <div style="font-size:.62rem;color:#8B2A2A;font-weight:600;margin-top:2px">
                    {ret_peor:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                k4.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #D8DDE8;
                border-top:3px solid #8896A8;border-radius:8px;padding:1rem">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.5rem;
                    color:#8896A8;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.3rem">
                    Activos analizados</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                    font-weight:700;color:#1A2035">{len(tickers)}</div>
                    <div style="font-size:.62rem;color:#8896A8;margin-top:2px">
                    vs S&P 500 benchmark</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
                st.plotly_chart(fig_tiempo(data, tickers), use_container_width=True)

                st.markdown("""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
                letter-spacing:.16em;text-transform:uppercase;color:#8896A8;
                margin:1rem 0 .5rem;border-left:2px solid #8B6914;padding-left:8px">
                Estadísticas del período</div>
                """, unsafe_allow_html=True)

                rows = []
                for i, ticker in enumerate(tickers):
                    if ticker not in est:
                        continue
                    e = est[ticker]
                    rows.append({
                        "Ticker":        ticker,
                        "Retorno total": f"{e['retorno_total']*100:+.2f}%",
                        "Volatilidad":   f"{e['volatilidad']*100:.2f}%",
                        "Sharpe aprox.": f"{e['sharpe_aprox']:.3f}",
                        "Max Drawdown":  f"{e['max_drawdown']*100:.2f}%",
                        "JB p-valor":    f"{e['jarque_bera_p']:.4f}",
                        "Normal?":       "No" if e['jarque_bera_p'] < 0.05 else "Sí",
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                with st.expander("Interpretación — Máquina del tiempo"):
                    st.markdown(f"""
El análisis cubre **{data['n_dias']} días hábiles** entre **{data['start_date']}** y **{data['end_date']}**.
**{mejor}** fue el activo con mejor desempeño ({ret_mejor:+.2f}%).
**{peor}** fue el activo con peor desempeño ({ret_peor:.2f}%).
El gráfico muestra precios normalizados a base 100. La línea punteada gris es el S&P 500.
                    """)

            except requests.exceptions.HTTPError as e:
                st.error(f"Error del backend: {e.response.json().get('detail', str(e))}")
            except Exception as e:
                st.error(f"Error: {e}")