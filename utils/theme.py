# utils/theme.py — Paleta modo claro refinado — RiskLab

COLORS = {
    # Fondos
    "bg"       : "#EEF1F6",
    "bg2"      : "#E6EAF2",
    "surface"  : "#FFFFFF",
    "surface2" : "#F4F6FB",
    "border"   : "#D8DDE8",
    "border2"  : "#C4CBD8",

    # Acentos
    "gold"     : "#8B6914",
    "gold2"    : "#A07820",
    "emerald"  : "#1A6B4A",
    "emerald2" : "#228A5E",
    "rose"     : "#8B2A2A",
    "rose2"    : "#A83232",
    "sky"      : "#1A4F6E",
    "sky2"     : "#2266A0",
    "violet"   : "#3D2F6B",
    "violet2"  : "#5242A0",

    # Texto
    "text"     : "#1A2035",
    "text2"    : "#4A5568",
    "text3"    : "#8896A8",
    "white"    : "#FFFFFF",
}

TICKER_COLORS = {
    "AAPL": "#8B6914",
    "JPM" : "#1A6B4A",
    "XOM" : "#1A4F6E",
    "JNJ" : "#3D2F6B",
    "AMZN": "#8B2A2A",
}

def plotly_base(height=400):
    return dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="IBM Plex Mono, monospace",
            color="#8896A8",
            size=10,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        xaxis=dict(
            gridcolor="#E6EAF2",
            gridwidth=0.5,
            showline=False,
            zeroline=False,
            tickfont=dict(color="#8896A8", size=9),
            type="date",
        ),
        yaxis=dict(
            gridcolor="#E6EAF2",
            gridwidth=0.5,
            showline=False,
            zeroline=False,
            tickfont=dict(color="#8896A8", size=9),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#D8DDE8",
            font=dict(family="IBM Plex Mono", size=11, color="#1A2035"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=9, color="#4A5568"),
            bordercolor="#D8DDE8",
        ),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="#8896A8",
            activecolor="#8B6914",
        ),
    )