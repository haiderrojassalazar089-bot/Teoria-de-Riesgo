"""
frontend/app.py — RiskLab USTA v2
Streamlit | Selector de activos dinámico + Módulos M1–M8 + Módulos avanzados
"""
import streamlit as st

st.set_page_config(
    page_title="RiskLab · USTA",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.styles import GLOBAL_CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
.stMultiSelect [data-baseweb="select"] {
    border-color: #D8DDE8 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Inicializar session_state ──────────────────────────────
if "tickers_seleccionados" not in st.session_state:
    st.session_state["tickers_seleccionados"] = []
if "portafolio_confirmado" not in st.session_state:
    st.session_state["portafolio_confirmado"] = False
if "tickers_info" not in st.session_state:
    st.session_state["tickers_info"] = {}

tickers_activos = st.session_state.get("tickers_seleccionados", [])
confirmado = st.session_state.get("portafolio_confirmado", False)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.8rem 1.2rem 1.4rem;border-bottom:1px solid #D8DDE8;margin-bottom:0.8rem;">
        <div style="font-family:'Playfair Display',serif;font-size:1.35rem;
                    font-weight:700;color:#1A2035;letter-spacing:0.01em;line-height:1.1;">
            Risk<span style="color:#8B6914;font-style:italic;">Lab</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.52rem;
                    color:#8896A8;letter-spacing:0.22em;text-transform:uppercase;margin-top:5px;">
            USTA · Teoría del Riesgo
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mostrar portafolio activo
    if confirmado and tickers_activos:
        st.markdown(f"""
        <div style="padding:.6rem 1.2rem;margin-bottom:.5rem;
        background:rgba(139,105,20,.06);border-radius:6px">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.48rem;
            letter-spacing:.2em;text-transform:uppercase;color:#8896A8;margin-bottom:4px">
            Portafolio activo</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
            color:#8B6914;font-weight:500">{' · '.join(tickers_activos)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.5rem;color:#8896A8;
                letter-spacing:0.3em;text-transform:uppercase;padding:0.2rem 1.2rem 0.5rem;">
        Navegación
    </div>
    """, unsafe_allow_html=True)

    # Opciones del sidebar — sin separador
    opciones_nav = ["◈ Selector de Activos"]

    if confirmado and len(tickers_activos) >= 2:
        opciones_nav += [
            "Vista General",
            "M1 · Análisis Técnico",
            "M2 · Rendimientos",
            "M3 · ARCH / GARCH",
            "M4 · CAPM & Beta",
            "M5 · VaR & CVaR",
            "M6 · Markowitz",
            "M7 · Señales & Alertas",
            "M8 · Macro & Benchmark",
            "M9 · Monte Carlo Visual",
            "M10 · Duelo de Portafolios",
            "M11 · Máquina del Tiempo",
        ]

    modulo = st.radio(
        label="navegacion",
        options=opciones_nav,
        label_visibility="collapsed",
    )

    # Botón cambiar portafolio
    if confirmado:
        st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
        if st.button("↺ Cambiar portafolio", use_container_width=True):
            st.session_state["portafolio_confirmado"] = False
            st.session_state["tickers_seleccionados"] = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2rem;padding:1rem 1.2rem;border-top:1px solid #D8DDE8;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.54rem;
                    color:#8896A8;line-height:2.0;">
            <span style="color:#1A6B4A;font-size:0.48rem;">●</span>
            <span style="color:#1A6B4A;"> API activa</span><br>
            Yahoo Finance · Tiempo real<br>
            <span style="color:#C4CBD8;">S&P 500 · Benchmark</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Routing ────────────────────────────────────────────────

if modulo != "◈ Selector de Activos" and not confirmado:
    st.warning("⚠️ Primero selecciona tus activos en el **Selector de Activos**.")
    modulo = "◈ Selector de Activos"

if modulo == "◈ Selector de Activos":
    from pages.selector import show
    show()

elif modulo == "Vista General":
    from pages.overview import show
    show()

elif modulo == "M1 · Análisis Técnico":
    from pages.m1_technical import show
    show()

elif modulo == "M2 · Rendimientos":
    from pages.m2_returns import show
    show()

elif modulo == "M3 · ARCH / GARCH":
    from pages.m3_garch import show
    show()

elif modulo == "M4 · CAPM & Beta":
    from pages.m4_capm import show
    show()

elif modulo == "M5 · VaR & CVaR":
    from pages.m5_var import show
    show()

elif modulo == "M6 · Markowitz":
    from pages.m6_markowitz import show
    show()

elif modulo == "M7 · Señales & Alertas":
    from pages.m7_signals import show
    show()

elif modulo == "M8 · Macro & Benchmark":
    from pages.m8_macro import show
    show()

elif modulo == "M9 · Monte Carlo Visual":
    from pages.m9_montecarlo import show
    show()

elif modulo == "M10 · Duelo de Portafolios":
    from pages.m10_duelo import show
    show()

elif modulo == "M11 · Máquina del Tiempo":
    from pages.m11_tiempo import show
    show()