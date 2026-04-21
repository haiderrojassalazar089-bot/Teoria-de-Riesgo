"""
frontend/pages/m11_tiempo.py — Módulo 11: Máquina del Tiempo
Reconstruye el portafolio en cualquier período histórico de los últimos 10 años.
Calendarios desplegables para seleccionar el rango de fechas.
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

    # Benchmark
    if data["benchmark_norm"]:
        n_bench = len(data["benchmark_norm"])
        fechas_b = fechas[:n_bench]
        fig.add_trace(go.Scatter(
            x=fechas_b, y=data["benchmark_norm"],
            name="S&P 500",
            line=dict(color="#C4CBD8", width=1.5, dash="dot"),
        ))

    # Cada activo
    for i, ticker in enumerate(tickers):
        if ticker not in data["retornos_norm"]:
            continue
        vals = data["retornos_norm"][ticker]
        n_v = len(vals)
        fechas_t = fechas[:n_v]
        col = TICKER_COLORS[i % len(TICKER_COLORS)]
        es_mejor = ticker == data["mejor_activo"]
        es_peor  = ticker == data["peor_activo"]
        width = 2.5 if (es_mejor or es_peor) else 1.6
        dash  = "solid"
        name  = f"{ticker} {'🏆' if es_mejor else '📉' if es_peor else ''}"
        fig.add_trace(go.Scatter(
            x=fechas_t, y=vals, name=name,
            line=dict(color=col, width=width, dash=dash),
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

    # ── Períodos predefinidos ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;letter-spacing:.16em;
    text-transform:uppercase;color:#8896A8;margin-bottom:.5rem;
    border-left:2px solid #8B6914;padding-left:8px">Períodos históricos relevantes</div>
    """, unsafe_allow_html=True)

    hoy = date.today()
    periodos = {
        "COVID (2020)":          (date(2020, 1, 1),  date(2020, 12, 31)),
        "Recuperación 2021":     (date(2021, 1, 1),  date(2021, 12, 31)),
        "Bear Market 2022":      (date(2022, 1, 1),  date(2022, 12, 31)),
        "Rally IA 2023":         (date(2023, 1, 1),  date(2023, 12, 31)),
        "Últimos 12 meses":      (hoy - timedelta(days=365), hoy),
        "Últimos 3 años":        (hoy - timedelta(days=1095), hoy),
        "Últimos 5 años":        (hoy - timedelta(days=1825), hoy),
    }

    cols_p = st.columns(len(periodos))
    fecha_ini_def = hoy - timedelta(days=365 * 3)
    fecha_fin_def = hoy

    for idx, (nombre, (fi, ff)) in enumerate(periodos.items()):
        with cols_p[idx]:
            if st.button(nombre, use_container_width=True, key=f"per_{idx}"):
                st.session_state["mt_fecha_ini"] = fi
                st.session_state["mt_fecha_fin"] = ff
                st.rerun()

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Calendarios desplegables ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;letter-spacing:.16em;
    text-transform:uppercase;color:#8896A8;margin-bottom:.5rem;
    border-left:2px solid #8B6914;padding-left:8px">O selecciona fechas personalizadas</div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    fecha_min = hoy - timedelta(days=3650)  # 10 años

    with col_f1:
        fecha_ini = st.date_input(
            "📅 Fecha de inicio",
            value=st.session_state.get("mt_fecha_ini", fecha_ini_def),
            min_value=fecha_min,
            max_value=hoy - timedelta(days=31),
            help="Selecciona la fecha de inicio del análisis (máx. 10 años atrás)",
        )

    with col_f2:
        fecha_fin = st.date_input(
            "📅 Fecha de fin",
            value=st.session_state.get("mt_fecha_fin", fecha_fin_def),
            min_value=fecha_min + timedelta(days=31),
            max_value=hoy,
            help="Selecciona la fecha de fin del análisis",
        )

    # Guardar en session_state
    st.session_state["mt_fecha_ini"] = fecha_ini
    st.session_state["mt_fecha_fin"] = fecha_fin

    # Validación visual
    delta_dias = (fecha_fin - fecha_ini).days
    if fecha_fin <= fecha_ini:
        st.error("La fecha de fin debe ser posterior a la fecha de inicio.")
        return
    if delta_dias < 30:
        st.error("El rango mínimo es 30 días.")
        return

    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
    color:#8896A8;margin:.4rem 0 1rem;text-align:center">
    Período seleccionado: <strong style="color:#1A2035">{fecha_ini.strftime('%d %b %Y')}</strong>
    → <strong style="color:#1A2035">{fecha_fin.strftime('%d %b %Y')}</strong>
    · {delta_dias} días ({delta_dias//30} meses aprox.)
    </div>
    """, unsafe_allow_html=True)

    if st.button("⏱ Viajar a este período", type="primary", use_container_width=True):
        with st.spinner(f"Reconstruyendo portafolio en {fecha_ini} → {fecha_fin}..."):
            try:
                payload = {
                    "tickers": tickers,
                    "years": 10,
                    "start_date": fecha_ini.strftime("%Y-%m-%d"),
                    "end_date":   fecha_fin.strftime("%Y-%m-%d"),
                }
                data = post_maquina(payload)

                # ── KPIs ──
                mejor = data["mejor_activo"]
                peor  = data["peor_activo"]
                est   = data["estadisticas"]

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Días analizados", f"{data['n_dias']}", f"{data['start_date']} → {data['end_date']}")
                k2.metric("🏆 Mejor activo", mejor,
                          f"+{est[mejor]['retorno_total']*100:.2f}%" if mejor in est else "")
                k3.metric("📉 Peor activo", peor,
                          f"{est[peor]['retorno_total']*100:.2f}%" if peor in est else "")
                if "^GSPC" in data.get("benchmark_norm", []):
                    bench_ret = (data["benchmark_norm"][-1] / 100 - 1) * 100
                    k4.metric("S&P 500 en el período", f"{bench_ret:+.2f}%")
                else:
                    k4.metric("Activos analizados", len(tickers))

                st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

                # ── Gráfica principal ──
                st.plotly_chart(fig_tiempo(data, tickers), use_container_width=True)

                # ── Tabla de estadísticas del período ──
                st.markdown("""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
                letter-spacing:.16em;text-transform:uppercase;color:#8896A8;
                margin:1rem 0 .5rem;border-left:2px solid #8B6914;padding-left:8px">
                Estadísticas del período
                </div>
                """, unsafe_allow_html=True)

                rows = []
                for i, ticker in enumerate(tickers):
                    if ticker not in est:
                        continue
                    e = est[ticker]
                    col = TICKER_COLORS[i % len(TICKER_COLORS)]
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
                    df_stats = pd.DataFrame(rows)
                    st.dataframe(df_stats, use_container_width=True, hide_index=True)

                with st.expander("Interpretación — Máquina del tiempo"):
                    st.markdown(f"""
                    El análisis cubre **{data['n_dias']} días hábiles** entre
                    **{data['start_date']}** y **{data['end_date']}**.

                    **{mejor}** fue el activo con mejor desempeño en este período
                    ({est.get(mejor, {}).get('retorno_total', 0)*100:+.2f}% de retorno total).

                    **{peor}** fue el activo con peor desempeño
                    ({est.get(peor, {}).get('retorno_total', 0)*100:+.2f}%).

                    El gráfico muestra los precios normalizados a base 100 —
                    cualquier valor sobre 100 representa ganancia desde el inicio del período.
                    La línea punteada gris es el S&P 500 como benchmark de referencia.
                    """)

            except requests.exceptions.HTTPError as e:
                detail = e.response.json().get("detail", str(e))
                st.error(f"Error del backend: {detail}")
            except Exception as e:
                st.error(f"Error: {e}")