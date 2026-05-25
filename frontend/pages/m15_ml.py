"""
pages/m15_ml.py — Módulo 15: Machine Learning · Señales de Trading
Clasificador Random Forest: BUY / HOLD / SELL
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.theme import plotly_base, COLORS
from utils.dynamic_tickers import get_tickers, get_ticker_colors, render_portafolio_badge

BACKEND_URL = "http://localhost:8002"


# ── Helpers ────────────────────────────────────────────────

def _card(label, value, color="#1A2035", bg="#F4F6FB", border="#D8DDE8"):
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:8px;
    padding:1rem 1.2rem;text-align:center;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.5rem;
        letter-spacing:0.2em;text-transform:uppercase;color:#8896A8;margin-bottom:6px;">
            {label}
        </div>
        <div style="font-family:'Playfair Display',serif;font-size:1.5rem;
        font-weight:700;color:{color};">
            {value}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _seccion(titulo):
    st.markdown(f"""
    <div style="margin:1.8rem 0 0.8rem;">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
        letter-spacing:0.25em;text-transform:uppercase;color:#8896A8;">
            {titulo}
        </span>
        <div style="height:1px;background:#D8DDE8;margin-top:4px;"></div>
    </div>
    """, unsafe_allow_html=True)


def _predict(ticker, rsi_val, macd_val, ret5, ret10, vol20):
    """Llama al endpoint /ml/predict del backend."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/ml/predict",
            json={
                "ticker": ticker,
                "features": [rsi_val, macd_val, ret5, ret10, vol20]
            },
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar al backend en el puerto 8002.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Error del modelo: {e.response.json().get('detail', str(e))}")
        return None


def _get_model_info():
    """Obtiene metadata del modelo."""
    try:
        r = requests.get(f"{BACKEND_URL}/ml/model-info", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _calcular_features_reales(ticker, years=1):
    """
    Intenta calcular features reales desde yfinance directamente.
    Retorna dict con RSI, MACD, ret_5d, ret_10d, vol_20d o None si falla.
    """
    try:
        import yfinance as yf
        start = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, auto_adjust=True,
                         progress=False, multi_level_index=False)
        if df.empty or len(df) < 30:
            return None

        close = df["Close"]

        # RSI 14
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_s = 100 - 100 / (1 + rs)

        # MACD histogram
        ema12    = close.ewm(span=12, adjust=False).mean()
        ema26    = close.ewm(span=26, adjust=False).mean()
        macd_l   = ema12 - ema26
        signal_l = macd_l.ewm(span=9, adjust=False).mean()
        hist     = macd_l - signal_l

        # Retornos
        ret5  = close.pct_change(5)
        ret10 = close.pct_change(10)

        # Volatilidad 20d
        vol20 = close.pct_change().rolling(20).std() * np.sqrt(252)

        return {
            "rsi":   round(float(rsi_s.iloc[-1]),  2),
            "macd":  round(float(hist.iloc[-1]),    4),
            "ret5":  round(float(ret5.iloc[-1]),    4),
            "ret10": round(float(ret10.iloc[-1]),   4),
            "vol20": round(float(vol20.iloc[-1]),   4),
        }
    except Exception:
        return None


# ── Main ───────────────────────────────────────────────────

def show():
    # Header
    st.markdown("""
    <div style="margin-bottom:1.6rem;">
        <div style="font-family:'Playfair Display',serif;font-size:1.6rem;
        font-weight:700;color:#1A2035;line-height:1.2;">
            Machine Learning
            <span style="color:#8B6914;font-style:italic;"> · Señales de Trading</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
        color:#8896A8;margin-top:4px;letter-spacing:0.05em;">
            Clasificador Random Forest · BUY / HOLD / SELL · Modelo v1.0.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_portafolio_badge()

    # Info del modelo
    info = _get_model_info()
    if info and info.get("status") == "loaded":
        st.markdown(f"""
        <div style="background:rgba(26,107,74,0.06);border:1px solid rgba(26,107,74,0.2);
        border-radius:7px;padding:0.6rem 1rem;margin-bottom:1.2rem;
        font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#1A6B4A;">
            ● Modelo cargado · {info.get('model_version','v1.0.0')} ·
            Features: {' · '.join(info.get('feature_names', []))}
        </div>
        """, unsafe_allow_html=True)
    elif info and info.get("status") == "not_trained":
        st.error("⚠️ El modelo no está entrenado. Ejecuta `python -m app.ml.train` dentro del contenedor.")
        return

    tickers = get_tickers()
    colors  = get_ticker_colors()

    # ── Tabs ──────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📡 Predicción Manual", "🔄 Predicción Automática"])

    # ════════════════════════════════════════════════
    # TAB 1 — PREDICCIÓN MANUAL CON SLIDERS
    # ════════════════════════════════════════════════
    with tab1:
        _seccion("Configurar Features")

        col_tk, col_exp = st.columns([1, 2])
        with col_tk:
            ticker_sel = st.selectbox("Activo", tickers, key="ml_ticker_manual")

        with col_exp:
            st.markdown("""
            <div style="background:#F4F6FB;border:1px solid #D8DDE8;border-radius:7px;
            padding:0.7rem 1rem;font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
            color:#4A5568;line-height:1.9;">
                <b style="color:#1A2035;">Features del modelo:</b><br>
                RSI · Momentum relativo [0–100] &nbsp;|&nbsp;
                MACD hist · Divergencia de medias móviles<br>
                Ret 5d · Retorno últimos 5 días &nbsp;|&nbsp;
                Ret 10d · Retorno últimos 10 días &nbsp;|&nbsp;
                Vol 20d · Volatilidad anualizada
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            rsi_val = st.slider(
                "RSI (14 períodos)",
                min_value=0.0, max_value=100.0, value=55.0, step=0.5,
                help="<30 sobrevendido · >70 sobrecomprado"
            )
            macd_val = st.slider(
                "MACD Histograma",
                min_value=-5.0, max_value=5.0, value=0.5, step=0.05,
                help="Positivo = impulso alcista · Negativo = impulso bajista"
            )
            ret5_val = st.slider(
                "Retorno 5 días (%)",
                min_value=-10.0, max_value=10.0, value=1.5, step=0.1,
                help="Retorno porcentual de los últimos 5 días"
            )

        with col2:
            ret10_val = st.slider(
                "Retorno 10 días (%)",
                min_value=-15.0, max_value=15.0, value=2.2, step=0.1,
                help="Retorno porcentual de los últimos 10 días"
            )
            vol20_val = st.slider(
                "Volatilidad 20 días (anualizada %)",
                min_value=5.0, max_value=80.0, value=18.0, step=0.5,
                help="Volatilidad anualizada calculada sobre 20 días"
            )

        # Convertir retornos a decimal para el modelo
        ret5_dec  = ret5_val  / 100
        ret10_dec = ret10_val / 100
        vol20_dec = vol20_val / 100

        # Indicadores visuales de los features
        _seccion("Resumen de Features")
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)

        rsi_color = "#1A6B4A" if rsi_val < 30 else ("#8B2A2A" if rsi_val > 70 else "#8B6914")
        with fc1:
            _card("RSI", f"{rsi_val:.1f}", color=rsi_color)
        with fc2:
            mc = "#1A6B4A" if macd_val > 0 else "#8B2A2A"
            _card("MACD hist", f"{macd_val:+.2f}", color=mc)
        with fc3:
            rc = "#1A6B4A" if ret5_val > 0 else "#8B2A2A"
            _card("Ret 5d", f"{ret5_val:+.1f}%", color=rc)
        with fc4:
            rc2 = "#1A6B4A" if ret10_val > 0 else "#8B2A2A"
            _card("Ret 10d", f"{ret10_val:+.1f}%", color=rc2)
        with fc5:
            vc = "#8B2A2A" if vol20_val > 40 else ("#8B6914" if vol20_val > 20 else "#1A6B4A")
            _card("Vol 20d", f"{vol20_val:.1f}%", color=vc)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Botón predecir
        col_btn, col_sp = st.columns([1, 3])
        with col_btn:
            predecir = st.button("⚡ Predecir Señal", use_container_width=True, type="primary")

        if predecir:
            with st.spinner("Consultando modelo..."):
                result = _predict(
                    ticker_sel, rsi_val, macd_val,
                    ret5_dec, ret10_dec, vol20_dec
                )

            if result:
                _seccion("Resultado del Modelo")

                signal = result["signal"]
                conf   = result["confidence"]
                probs  = result["probabilities"]

                # Color y emoji por señal
                sig_config = {
                    "BUY":  {"color": "#1A6B4A", "bg": "rgba(26,107,74,0.08)",
                              "border": "rgba(26,107,74,0.3)", "emoji": "▲"},
                    "HOLD": {"color": "#8B6914", "bg": "rgba(139,105,20,0.08)",
                              "border": "rgba(139,105,20,0.3)", "emoji": "◆"},
                    "SELL": {"color": "#8B2A2A", "bg": "rgba(139,42,42,0.08)",
                              "border": "rgba(139,42,42,0.3)", "emoji": "▼"},
                }
                cfg = sig_config.get(signal, sig_config["HOLD"])

                # Panel principal de señal
                st.markdown(f"""
                <div style="background:{cfg['bg']};border:2px solid {cfg['border']};
                border-radius:12px;padding:1.8rem 2rem;text-align:center;margin:1rem 0;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
                    letter-spacing:0.25em;text-transform:uppercase;color:#8896A8;margin-bottom:8px;">
                        {ticker_sel} · Señal del modelo
                    </div>
                    <div style="font-family:'Playfair Display',serif;font-size:3rem;
                    font-weight:700;color:{cfg['color']};line-height:1;">
                        {cfg['emoji']} {signal}
                    </div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;
                    color:{cfg['color']};margin-top:8px;font-weight:600;">
                        Confianza: {conf*100:.1f}%
                    </div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
                    color:#8896A8;margin-top:4px;">
                        Modelo: {result.get('model_version','v1.0.0')} · Random Forest
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probabilidades por clase
                pc1, pc2, pc3 = st.columns(3)
                prob_sell = probs.get("SELL", 0)
                prob_hold = probs.get("HOLD", 0)
                prob_buy  = probs.get("BUY",  0)

                with pc1:
                    _card("P(SELL)", f"{prob_sell*100:.1f}%", color="#8B2A2A",
                          bg="rgba(139,42,42,0.05)", border="rgba(139,42,42,0.2)")
                with pc2:
                    _card("P(HOLD)", f"{prob_hold*100:.1f}%", color="#8B6914",
                          bg="rgba(139,105,20,0.05)", border="rgba(139,105,20,0.2)")
                with pc3:
                    _card("P(BUY)", f"{prob_buy*100:.1f}%", color="#1A6B4A",
                          bg="rgba(26,107,74,0.05)", border="rgba(26,107,74,0.2)")

                # Gráfico de barras de probabilidades
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["SELL", "HOLD", "BUY"],
                    y=[prob_sell * 100, prob_hold * 100, prob_buy * 100],
                    marker_color=["#8B2A2A", "#8B6914", "#1A6B4A"],
                    marker_line_width=0,
                    text=[f"{prob_sell*100:.1f}%",
                          f"{prob_hold*100:.1f}%",
                          f"{prob_buy*100:.1f}%"],
                    textposition="outside",
                    textfont=dict(
                        family="IBM Plex Mono",
                        size=11,
                        color="#1A2035"
                    ),
                ))

                layout = plotly_base(height=280)
                layout.update(
                    title=dict(
                        text=f"Distribución de probabilidades — {ticker_sel}",
                        font=dict(family="IBM Plex Mono", size=11, color="#8896A8"),
                        x=0,
                    ),
                    xaxis=dict(
                        gridcolor="#E6EAF2", showline=False, zeroline=False,
                        tickfont=dict(color="#1A2035", size=12,
                                      family="IBM Plex Mono"),
                        type="category",
                    ),
                    yaxis=dict(
                        gridcolor="#E6EAF2", showline=False, zeroline=False,
                        tickfont=dict(color="#8896A8", size=9),
                        ticksuffix="%",
                        range=[0, 110],
                    ),
                    showlegend=False,
                    bargap=0.35,
                )
                fig.update_layout(layout)
                st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════
    # TAB 2 — PREDICCIÓN AUTOMÁTICA CON DATOS REALES
    # ════════════════════════════════════════════════
    with tab2:
        _seccion("Features calculados desde Yahoo Finance")

        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8896A8;
        margin-bottom:1rem;line-height:1.8;">
            Los features se calculan automáticamente desde los precios reales del activo.
            RSI(14) · MACD histograma · Retornos 5d y 10d · Volatilidad 20d anualizada.
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            ticker_auto = st.selectbox("Activo", tickers, key="ml_ticker_auto")

        with col_b:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            calcular = st.button(
                "🔄 Calcular Features y Predecir",
                use_container_width=True,
                type="primary",
                key="btn_auto"
            )

        if calcular:
            with st.spinner(f"Descargando datos de {ticker_auto} y calculando features..."):
                feats = _calcular_features_reales(ticker_auto)

            if feats is None:
                st.error(f"No se pudieron calcular los features para {ticker_auto}. "
                         "Verifica la conexión a internet o prueba otro ticker.")
            else:
                _seccion("Features calculados")

                fa1, fa2, fa3, fa4, fa5 = st.columns(5)
                rsi_c = "#1A6B4A" if feats["rsi"] < 30 else ("#8B2A2A" if feats["rsi"] > 70 else "#8B6914")
                with fa1:
                    _card("RSI", f"{feats['rsi']:.1f}", color=rsi_c)
                with fa2:
                    mc = "#1A6B4A" if feats["macd"] > 0 else "#8B2A2A"
                    _card("MACD hist", f"{feats['macd']:+.4f}", color=mc)
                with fa3:
                    rc = "#1A6B4A" if feats["ret5"] > 0 else "#8B2A2A"
                    _card("Ret 5d", f"{feats['ret5']*100:+.2f}%", color=rc)
                with fa4:
                    rc2 = "#1A6B4A" if feats["ret10"] > 0 else "#8B2A2A"
                    _card("Ret 10d", f"{feats['ret10']*100:+.2f}%", color=rc2)
                with fa5:
                    vc = "#8B2A2A" if feats["vol20"]*100 > 40 else (
                         "#8B6914" if feats["vol20"]*100 > 20 else "#1A6B4A")
                    _card("Vol 20d", f"{feats['vol20']*100:.1f}%", color=vc)

                # Predecir con features reales
                with st.spinner("Consultando modelo..."):
                    result = _predict(
                        ticker_auto,
                        feats["rsi"],
                        feats["macd"],
                        feats["ret5"],
                        feats["ret10"],
                        feats["vol20"],
                    )

                if result:
                    _seccion("Resultado del Modelo")

                    signal = result["signal"]
                    conf   = result["confidence"]
                    probs  = result["probabilities"]

                    sig_config = {
                        "BUY":  {"color": "#1A6B4A", "bg": "rgba(26,107,74,0.08)",
                                  "border": "rgba(26,107,74,0.3)", "emoji": "▲"},
                        "HOLD": {"color": "#8B6914", "bg": "rgba(139,105,20,0.08)",
                                  "border": "rgba(139,105,20,0.3)", "emoji": "◆"},
                        "SELL": {"color": "#8B2A2A", "bg": "rgba(139,42,42,0.08)",
                                  "border": "rgba(139,42,42,0.3)", "emoji": "▼"},
                    }
                    cfg = sig_config.get(signal, sig_config["HOLD"])

                    st.markdown(f"""
                    <div style="background:{cfg['bg']};border:2px solid {cfg['border']};
                    border-radius:12px;padding:1.8rem 2rem;text-align:center;margin:1rem 0;">
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
                        letter-spacing:0.25em;text-transform:uppercase;color:#8896A8;margin-bottom:8px;">
                            {ticker_auto} · Señal con datos reales · {datetime.today().strftime('%Y-%m-%d')}
                        </div>
                        <div style="font-family:'Playfair Display',serif;font-size:3rem;
                        font-weight:700;color:{cfg['color']};line-height:1;">
                            {cfg['emoji']} {signal}
                        </div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;
                        color:{cfg['color']};margin-top:8px;font-weight:600;">
                            Confianza: {conf*100:.1f}%
                        </div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
                        color:#8896A8;margin-top:4px;">
                            Features calculados desde precios reales de Yahoo Finance
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    pb1, pb2, pb3 = st.columns(3)
                    prob_sell = probs.get("SELL", 0)
                    prob_hold = probs.get("HOLD", 0)
                    prob_buy  = probs.get("BUY",  0)

                    with pb1:
                        _card("P(SELL)", f"{prob_sell*100:.1f}%", color="#8B2A2A",
                              bg="rgba(139,42,42,0.05)", border="rgba(139,42,42,0.2)")
                    with pb2:
                        _card("P(HOLD)", f"{prob_hold*100:.1f}%", color="#8B6914",
                              bg="rgba(139,105,20,0.05)", border="rgba(139,105,20,0.2)")
                    with pb3:
                        _card("P(BUY)", f"{prob_buy*100:.1f}%", color="#1A6B4A",
                              bg="rgba(26,107,74,0.05)", border="rgba(26,107,74,0.2)")

                    # Nota interpretativa
                    st.markdown(f"""
                    <div style="background:#F4F6FB;border:1px solid #D8DDE8;border-radius:7px;
                    padding:0.8rem 1rem;margin-top:1rem;font-family:'IBM Plex Mono',monospace;
                    font-size:0.57rem;color:#4A5568;line-height:1.9;">
                        <b style="color:#1A2035;">Nota sobre la predicción:</b><br>
                        El modelo fue entrenado con datos sintéticos que simulan la relación entre
                        indicadores técnicos y retornos futuros. Un RSI de {feats['rsi']:.1f}
                        {"(sobrecomprado)" if feats['rsi'] > 70 else "(sobrevendido)" if feats['rsi'] < 30 else "(zona neutra)"},
                        retornos recientes de {feats['ret5']*100:+.2f}% (5d) y {feats['ret10']*100:+.2f}% (10d),
                        y volatilidad de {feats['vol20']*100:.1f}% generan esta señal.
                        La confianza del {conf*100:.1f}% refleja la proporción de árboles del
                        Random Forest que votaron por {signal}.
                    </div>
                    """, unsafe_allow_html=True)

    # ── Footer con info del modelo ─────────────────────────
    _seccion("Sobre el Modelo")
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:0.5rem;">
        <div style="background:#F4F6FB;border:1px solid #D8DDE8;border-radius:8px;padding:1rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.5rem;letter-spacing:0.2em;
            text-transform:uppercase;color:#8896A8;margin-bottom:6px;">Algoritmo</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#1A2035;
            font-weight:600;">Random Forest</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;color:#4A5568;
            margin-top:4px;">100 árboles · max_depth=10<br>Entrenado con 2000 muestras</div>
        </div>
        <div style="background:#F4F6FB;border:1px solid #D8DDE8;border-radius:8px;padding:1rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.5rem;letter-spacing:0.2em;
            text-transform:uppercase;color:#8896A8;margin-bottom:6px;">Pipeline</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#1A2035;
            font-weight:600;">StandardScaler → RF</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;color:#4A5568;
            margin-top:4px;">Normalización automática<br>shuffle=False · split 80/20</div>
        </div>
        <div style="background:#F4F6FB;border:1px solid #D8DDE8;border-radius:8px;padding:1rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.5rem;letter-spacing:0.2em;
            text-transform:uppercase;color:#8896A8;margin-bottom:6px;">Clases</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#1A2035;
            font-weight:600;">SELL · HOLD · BUY</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;color:#4A5568;
            margin-top:4px;">Retorno futuro esperado<br>&lt;-1% · entre ±1% · &gt;+1%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)