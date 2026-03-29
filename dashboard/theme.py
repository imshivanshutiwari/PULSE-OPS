BG_PRIMARY = "#050a06"
BG_PANEL = "#080d08"
BG_CARD = "#0a1008"
BORDER = "#0f1e0a"
ACCENT_GRN = "#639922"
ACCENT_DIM = "#172a06"
SUCCESS = "#97c459"
WARNING = "#ba7517"
CRITICAL = "#e24b4a"
INFO = "#378add"
TEXT = "#c0dd97"
TEXT_DIM = "#1a3008"
FONT = "JetBrains Mono, monospace"

PLOT_LAYOUT = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_PANEL,
    font=dict(family=FONT, color=TEXT, size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER,
        tickcolor=BORDER,
        zerolinecolor=BORDER,
    ),
    yaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER,
        tickcolor=BORDER,
        zerolinecolor=BORDER,
    ),
    legend=dict(
        bgcolor=BG_CARD,
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(color=TEXT),
    ),
    colorway=[ACCENT_GRN, INFO, WARNING, CRITICAL, SUCCESS, "#9b59b6", "#1abc9c", "#e67e22"],
)

CARD_STYLE = {
    "backgroundColor": BG_CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "4px",
    "padding": "12px",
    "marginBottom": "12px",
}

HEADER_STYLE = {
    "backgroundColor": BG_PANEL,
    "borderBottom": f"2px solid {ACCENT_GRN}",
    "padding": "10px 20px",
    "fontFamily": FONT,
    "color": TEXT,
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
}

TAB_STYLE = {
    "backgroundColor": BG_PANEL,
    "color": TEXT_DIM,
    "border": f"1px solid {BORDER}",
    "borderBottom": "none",
    "fontFamily": FONT,
    "fontSize": "12px",
    "padding": "8px 16px",
}

TAB_SELECTED_STYLE = {
    "backgroundColor": ACCENT_DIM,
    "color": ACCENT_GRN,
    "border": f"1px solid {ACCENT_GRN}",
    "borderBottom": f"2px solid {ACCENT_GRN}",
    "fontFamily": FONT,
    "fontSize": "12px",
    "padding": "8px 16px",
    "fontWeight": "bold",
}

STATUS_COLORS = {
    "healthy": SUCCESS,
    "degraded": WARNING,
    "critical": CRITICAL,
    "unknown": TEXT_DIM,
    "staging": INFO,
    "production": ACCENT_GRN,
    "archived": TEXT_DIM,
}
