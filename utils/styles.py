"""
utils/styles.py — CSS global modo claro refinado
Estilo: Minimalismo académico — Journal of Finance aesthetic
"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&family=IBM+Plex+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
}

.stApp { background-color: #EEF1F6 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #D8DDE8 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04) !important;
}

[data-testid="stSidebarNav"] { display: none !important; }

/* ── Radio sidebar ── */
[data-testid="stSidebar"] .stRadio > div { gap: 0 !important; padding: 0 0.5rem !important; }
[data-testid="stSidebar"] .stRadio label {
    display: flex !important; align-items: center !important;
    padding: 0.45rem 0.8rem !important; border-radius: 5px !important;
    margin: 1px 0 !important; color: #4A5568 !important;
    font-size: 0.83rem !important; font-family: 'Inter', sans-serif !important;
    transition: all 0.15s !important; cursor: pointer !important; width: 100% !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #1A2035 !important; background: rgba(139,105,20,0.06) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background-color: #FFFFFF !important;
    border: 1px solid #C4CBD8 !important;
    border-radius: 6px !important;
    color: #1A2035 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important;
}
.stSelectbox > div > div:hover { border-color: #8B6914 !important; }

/* ── Slider ── */
.stSlider > div > div > div > div { background-color: #8B6914 !important; }

/* ── Métricas ── */
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    color: #1A2035 !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #8896A8 !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid #D8DDE8 !important;
    gap: 0 !important; padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important; color: #8896A8 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.68rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    border-bottom: 1px solid transparent !important; padding: 0.6rem 1rem !important;
    transition: all 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #4A5568 !important; }
.stTabs [aria-selected="true"] {
    color: #8B6914 !important;
    border-bottom: 1px solid #8B6914 !important;
    background-color: transparent !important;
}

/* ── Botones ── */
.stButton > button {
    background-color: transparent !important;
    border: 1px solid #C4CBD8 !important; color: #4A5568 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important;
    letter-spacing: 0.1em !important; border-radius: 5px !important;
    transition: all 0.15s !important; padding: 0.4rem 1rem !important;
}
.stButton > button:hover {
    border-color: #8B6914 !important; color: #8B6914 !important;
    background-color: rgba(139,105,20,0.04) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D8DDE8 !important; border-radius: 6px !important;
}

/* ── Expander ── */
details > summary {
    background-color: #FFFFFF !important; border: 1px solid #D8DDE8 !important;
    border-radius: 6px !important; color: #4A5568 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important;
    padding: 0.6rem 0.9rem !important; letter-spacing: 0.06em !important;
}
details[open] > summary {
    border-bottom-left-radius: 0 !important; border-bottom-right-radius: 0 !important;
    color: #8B6914 !important;
}
details > div {
    background: #FFFFFF !important; border: 1px solid #D8DDE8 !important;
    border-top: none !important; border-radius: 0 0 6px 6px !important;
    padding: 0.8rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #EEF1F6; }
::-webkit-scrollbar-thumb { background: #C4CBD8; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #8B6914; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #D8DDE8 !important; margin: 0.8rem 0 !important; }

/* ── Tipografía ── */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #1A2035 !important; letter-spacing: -0.01em !important; line-height: 1.2 !important;
}
h1 { font-size: 1.7rem !important; font-weight: 700 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; }

p, li { font-family: 'Inter', sans-serif !important; color: #4A5568 !important; font-size: 0.83rem !important; line-height: 1.65 !important; }

code, pre {
    font-family: 'IBM Plex Mono', monospace !important;
    background: #F4F6FB !important; color: #8B6914 !important;
    border-radius: 4px !important; font-size: 0.78rem !important;
}

/* ── Ocultar branding Streamlit ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Contenido ── */
.main .block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1440px !important; }

/* ── Alerts ── */
.stAlert {
    background-color: #FFFFFF !important; border: 1px solid #D8DDE8 !important;
    border-radius: 6px !important; color: #4A5568 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important;
}
</style>
"""