"""
PAGE 2 – Drift Analysis Lab
VIZ08: Evidently drift report embed
VIZ09: Feature distribution comparison (violin)
VIZ10: Model performance degradation
VIZ11: Target drift analysis
VIZ12: Data quality scorecard (heatmap)
VIZ13: Drift history calendar
"""

import datetime
import os
import glob as _glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import (
    BG_PRIMARY,
    BG_PANEL,
    BG_CARD,
    BORDER,
    ACCENT_GRN,
    ACCENT_DIM,
    SUCCESS,
    WARNING,
    CRITICAL,
    INFO,
    TEXT,
    TEXT_DIM,
    FONT,
    PLOT_LAYOUT,
    CARD_STYLE,
)

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "drift", "reports"
)
FEATURES = ["age", "income", "credit_score", "debt_ratio", "num_accounts", "years_employed"]

np.random.seed(21)


def _find_latest_report():
    pattern = os.path.join(REPORTS_DIR, "*.html")
    files = _glob.glob(pattern)
    if files:
        return sorted(files)[-1]
    return None


# ---------------------------------------------------------------------------
# VIZ08 – Evidently drift report embed
# ---------------------------------------------------------------------------
def build_viz08():
    report_path = _find_latest_report()
    if report_path:
        try:
            with open(report_path, "r", encoding="utf-8") as fh:
                html_content = fh.read()
            return html.Div(
                [
                    html.Div(
                        "VIZ08 — Evidently Drift Report",
                        style={
                            "color": ACCENT_GRN,
                            "fontSize": "12px",
                            "marginBottom": "8px",
                            "fontFamily": FONT,
                        },
                    ),
                    html.Iframe(
                        srcDoc=html_content,
                        style={
                            "width": "100%",
                            "height": "520px",
                            "border": f"1px solid {BORDER}",
                            "borderRadius": "4px",
                            "backgroundColor": BG_CARD,
                        },
                    ),
                ],
                style=CARD_STYLE,
            )
        except Exception:
            pass

    # Fallback styled placeholder
    drift_rows = [
        ("age", 0.21, "PSI", "⚠ DRIFT"),
        ("income", 0.08, "KS test", "✓ OK"),
        ("credit_score", 0.34, "PSI", "⚠ DRIFT"),
        ("debt_ratio", 0.05, "KS test", "✓ OK"),
        ("num_accounts", 0.19, "Chi²", "⚠ DRIFT"),
        ("years_employed", 0.03, "KS test", "✓ OK"),
    ]
    rows_html = []
    for feat, score, method, result in drift_rows:
        color = WARNING if "⚠" in result else SUCCESS
        rows_html.append(
            html.Tr(
                [
                    html.Td(
                        feat,
                        style={
                            "padding": "4px 10px",
                            "color": TEXT,
                            "fontFamily": FONT,
                            "fontSize": "11px",
                        },
                    ),
                    html.Td(
                        f"{score:.2f}",
                        style={
                            "padding": "4px 10px",
                            "color": INFO,
                            "fontFamily": FONT,
                            "fontSize": "11px",
                        },
                    ),
                    html.Td(
                        method,
                        style={
                            "padding": "4px 10px",
                            "color": TEXT_DIM,
                            "fontFamily": FONT,
                            "fontSize": "11px",
                        },
                    ),
                    html.Td(
                        result,
                        style={
                            "padding": "4px 10px",
                            "color": color,
                            "fontFamily": FONT,
                            "fontSize": "11px",
                            "fontWeight": "bold",
                        },
                    ),
                ]
            )
        )
    return html.Div(
        [
            html.Div(
                "VIZ08 — Evidently Drift Report (run `make fetch` to load real report)",
                style={
                    "color": ACCENT_GRN,
                    "fontSize": "12px",
                    "marginBottom": "8px",
                    "fontFamily": FONT,
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Dataset Drift: ",
                                style={"color": TEXT_DIM, "fontFamily": FONT, "fontSize": "11px"},
                            ),
                            html.Span(
                                "DETECTED",
                                style={
                                    "color": WARNING,
                                    "fontFamily": FONT,
                                    "fontSize": "11px",
                                    "fontWeight": "bold",
                                },
                            ),
                            html.Span(
                                "   |   Drifted features: ",
                                style={"color": TEXT_DIM, "fontFamily": FONT, "fontSize": "11px"},
                            ),
                            html.Span(
                                "3 / 6",
                                style={"color": WARNING, "fontFamily": FONT, "fontSize": "11px"},
                            ),
                            html.Span(
                                "   |   Share: ",
                                style={"color": TEXT_DIM, "fontFamily": FONT, "fontSize": "11px"},
                            ),
                            html.Span(
                                "50%",
                                style={"color": WARNING, "fontFamily": FONT, "fontSize": "11px"},
                            ),
                        ],
                        style={
                            "marginBottom": "10px",
                            "padding": "8px",
                            "backgroundColor": BG_PANEL,
                            "border": f"1px solid {BORDER}",
                            "borderRadius": "3px",
                        },
                    ),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th(
                                            h,
                                            style={
                                                "padding": "4px 10px",
                                                "color": ACCENT_GRN,
                                                "fontFamily": FONT,
                                                "fontSize": "11px",
                                                "borderBottom": f"1px solid {BORDER}",
                                            },
                                        )
                                        for h in ["Feature", "Drift Score", "Method", "Status"]
                                    ]
                                )
                            ),
                            html.Tbody(rows_html),
                        ],
                        style={
                            "width": "100%",
                            "borderCollapse": "collapse",
                            "backgroundColor": BG_CARD,
                        },
                    ),
                ]
            ),
        ],
        style=CARD_STYLE,
    )


# ---------------------------------------------------------------------------
# VIZ09 – Feature distribution comparison (violin)
# ---------------------------------------------------------------------------
def build_viz09():
    rng = np.random.default_rng(33)
    ref_data = {
        "age": rng.normal(42, 12, 500),
        "income": rng.lognormal(10.5, 0.6, 500),
        "credit_score": rng.normal(680, 60, 500),
        "debt_ratio": rng.beta(2, 5, 500) * 100,
        "num_accounts": rng.poisson(4.5, 500).astype(float),
        "years_employed": rng.exponential(5, 500),
    }
    cur_data = {
        "age": rng.normal(38, 13, 500),  # shifted
        "income": rng.lognormal(10.6, 0.6, 500),
        "credit_score": rng.normal(645, 70, 500),  # shifted
        "debt_ratio": rng.beta(2.5, 4, 500) * 100,
        "num_accounts": rng.poisson(5.5, 500).astype(float),  # shifted
        "years_employed": rng.exponential(5.2, 500),
    }
    fig = go.Figure()
    for feat in FEATURES:
        ref_norm = (ref_data[feat] - ref_data[feat].mean()) / (ref_data[feat].std() + 1e-9)
        cur_norm = (cur_data[feat] - ref_data[feat].mean()) / (ref_data[feat].std() + 1e-9)
        fig.add_trace(
            go.Violin(
                y=ref_norm,
                x=[feat] * len(ref_norm),
                name="Reference",
                legendgroup="Reference",
                showlegend=(feat == FEATURES[0]),
                side="negative",
                line_color=INFO,
                fillcolor="rgba(55,138,221,0.25)",
                opacity=0.9,
            )
        )
        fig.add_trace(
            go.Violin(
                y=cur_norm,
                x=[feat] * len(cur_norm),
                name="Current",
                legendgroup="Current",
                showlegend=(feat == FEATURES[0]),
                side="positive",
                line_color=WARNING,
                fillcolor="rgba(186,117,23,0.25)",
                opacity=0.9,
            )
        )
    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ09 — Feature Distribution Comparison (normalised)",
        font=dict(color=ACCENT_GRN, size=12),
    )
    layout["violingap"] = 0
    layout["violinmode"] = "overlay"
    layout["height"] = 340
    layout["yaxis"] = {
        **PLOT_LAYOUT.get("yaxis", {}),
        "title": "Normalised Value",
        "zeroline": False,
    }
    fig.update_layout(**layout)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ10 – Model performance degradation + confusion matrix
# ---------------------------------------------------------------------------
def build_viz10():
    days = pd.date_range(end=datetime.date.today(), periods=30, freq="D")
    rng = np.random.default_rng(5)
    drift_day = 20
    acc = np.concatenate(
        [
            np.clip(0.873 + rng.normal(0, 0.005, drift_day), 0.85, 0.92),
            np.clip(
                0.873 - np.linspace(0, 0.06, 30 - drift_day) + rng.normal(0, 0.007, 30 - drift_day),
                0.80,
                0.92,
            ),
        ]
    )
    f1 = acc - rng.uniform(0.008, 0.018, 30)
    recall = f1 - rng.uniform(0.005, 0.015, 30)

    # Confusion matrix at latest point
    cm = np.array([[432, 51], [28, 489]])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Performance Over Time", "Confusion Matrix (latest)"],
        column_widths=[0.65, 0.35],
    )
    fig.add_trace(
        go.Scatter(x=days, y=acc, name="Accuracy", line=dict(color=ACCENT_GRN, width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=days, y=f1, name="F1 Score", line=dict(color=INFO, width=2)), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=days, y=recall, name="Recall", line=dict(color=WARNING, width=2)), row=1, col=1
    )
    fig.add_vline(
        x=days[drift_day].timestamp() * 1000,
        line_dash="dot",
        line_color=CRITICAL,
        annotation_text="Drift detected",
        annotation_font=dict(color=CRITICAL, size=9),
    )

    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=["Predicted NEG", "Predicted POS"],
            y=["Actual NEG", "Actual POS"],
            colorscale=[[0, BG_PANEL], [1, ACCENT_GRN]],
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont=dict(color=TEXT, size=14, family=FONT),
        ),
        row=1,
        col=2,
    )

    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ10 — Model Performance Degradation", font=dict(color=ACCENT_GRN, size=12)
    )
    layout["height"] = 320
    for k in ["xaxis", "yaxis"]:
        layout.pop(k, None)
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(color=TEXT, size=10, family=FONT))
    for i in range(1, 3):
        s = "" if i == 1 else str(i)
        fig.update_layout(
            **{
                f"xaxis{s}": dict(
                    gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_DIM, size=9)
                ),
                f"yaxis{s}": dict(
                    gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_DIM, size=9)
                ),
            }
        )
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ11 – Target drift analysis
# ---------------------------------------------------------------------------
def build_viz11():
    rng = np.random.default_rng(17)
    ref_target = rng.binomial(1, 0.38, 1000)
    cur_target = rng.binomial(1, 0.51, 1000)  # distribution shifted

    ref_counts = [np.sum(ref_target == 0), np.sum(ref_target == 1)]
    cur_counts = [np.sum(cur_target == 0), np.sum(cur_target == 1)]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Target Class Distribution", "Chi² Test Statistics"],
        column_widths=[0.55, 0.45],
        specs=[[{"type": "xy"}, {"type": "table"}]],
    )
    fig.add_trace(
        go.Bar(
            x=["Class 0 (Neg)", "Class 1 (Pos)"],
            y=[c / 10 for c in ref_counts],
            name="Reference %",
            marker_color=INFO,
            marker_line_color=BORDER,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=["Class 0 (Neg)", "Class 1 (Pos)"],
            y=[c / 10 for c in cur_counts],
            name="Current %",
            marker_color=WARNING,
            marker_line_color=BORDER,
        ),
        row=1,
        col=1,
    )

    stats = {
        "chi2_statistic": 28.43,
        "p_value": 0.0000095,
        "drift_detected": True,
        "reference_rate": 0.38,
        "current_rate": 0.51,
        "abs_change": 0.13,
    }
    labels = list(stats.keys())
    values = [str(v) for v in stats.values()]
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>"],
                fill_color=ACCENT_DIM,
                font=dict(color=ACCENT_GRN, size=11, family=FONT),
                line_color=BORDER,
            ),
            cells=dict(
                values=[labels, values],
                fill_color=[[BG_CARD] * len(labels), [BG_CARD] * len(labels)],
                font=dict(
                    color=[TEXT_DIM, [CRITICAL if "True" in str(v) else TEXT for v in values]],
                    size=10,
                    family=FONT,
                ),
                line_color=BORDER,
            ),
        ),
        row=1,
        col=2,
    )

    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ11 — Target Drift Analysis", font=dict(color=ACCENT_GRN, size=12)
    )
    layout["height"] = 310
    layout["barmode"] = "group"
    for k in ["xaxis", "yaxis"]:
        layout.pop(k, None)
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(color=TEXT, size=10, family=FONT))
    fig.update_layout(
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT)),
        yaxis=dict(
            gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT), title="Count (%)"
        ),
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ12 – Data quality scorecard (heatmap)
# ---------------------------------------------------------------------------
def build_viz12():
    rng = np.random.default_rng(3)
    features_ext = [
        "age",
        "income",
        "credit_score",
        "debt_ratio",
        "num_accounts",
        "years_employed",
        "zip_code",
        "loan_amount",
    ]
    weeks = [f"W{i}" for i in range(1, 9)]

    missing = rng.uniform(0, 0.08, (len(features_ext), len(weeks)))
    missing[2, 5:] = rng.uniform(0.12, 0.22, len(weeks) - 5)  # spike
    outlier = rng.uniform(0, 0.05, (len(features_ext), len(weeks)))
    outlier[4, 3:6] = rng.uniform(0.09, 0.17, 3)  # spike

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Missing Rate (%)", "Outlier Rate (%)"],
    )
    fig.add_trace(
        go.Heatmap(
            z=missing * 100,
            x=weeks,
            y=features_ext,
            colorscale=[[0, BG_PANEL], [0.5, WARNING], [1, CRITICAL]],
            zmin=0,
            zmax=25,
            colorbar=dict(
                x=0.46,
                thickness=10,
                tickfont=dict(color=TEXT_DIM, size=8),
                title=dict(text="%", font=dict(color=TEXT_DIM)),
            ),
            hoverongaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=outlier * 100,
            x=weeks,
            y=features_ext,
            colorscale=[[0, BG_PANEL], [0.5, INFO], [1, CRITICAL]],
            zmin=0,
            zmax=25,
            colorbar=dict(
                x=1.01,
                thickness=10,
                tickfont=dict(color=TEXT_DIM, size=8),
                title=dict(text="%", font=dict(color=TEXT_DIM)),
            ),
            hoverongaps=False,
        ),
        row=1,
        col=2,
    )

    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ12 — Data Quality Scorecard", font=dict(color=ACCENT_GRN, size=12)
    )
    layout["height"] = 340
    for k in ["xaxis", "yaxis"]:
        layout.pop(k, None)
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(color=TEXT, size=10, family=FONT))
    for i in range(1, 3):
        s = "" if i == 1 else str(i)
        fig.update_layout(
            **{
                f"xaxis{s}": dict(
                    gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_DIM, size=9)
                ),
                f"yaxis{s}": dict(
                    gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_DIM, size=9)
                ),
            }
        )
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ13 – Drift history calendar (GitHub-style heatmap)
# ---------------------------------------------------------------------------
def build_viz13():
    rng = np.random.default_rng(44)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=364)
    all_dates = pd.date_range(start_date, end_date, freq="D")

    drift_scores = np.clip(rng.exponential(0.04, len(all_dates)), 0, 0.5)
    drift_scores[rng.choice(len(all_dates), 20, replace=False)] = rng.uniform(0.2, 0.5, 20)

    day_matrix = np.full((7, 53), np.nan)

    for i, (date, score) in enumerate(zip(all_dates, drift_scores)):
        week_num = i // 7
        day_num = i % 7
        if week_num < 53:
            day_matrix[day_num, week_num] = score

    fig = go.Figure(
        go.Heatmap(
            z=day_matrix,
            colorscale=[
                [0.0, BG_PANEL],
                [0.01, ACCENT_DIM],
                [0.3, ACCENT_GRN],
                [0.6, WARNING],
                [1.0, CRITICAL],
            ],
            zmin=0,
            zmax=0.5,
            xgap=2,
            ygap=2,
            colorbar=dict(
                title=dict(text="Drift Score", font=dict(color=TEXT, size=10, family=FONT)),
                tickfont=dict(color=TEXT_DIM, size=8),
                thickness=12,
            ),
            hovertemplate="Week %{x}, Day %{y}<br>Drift: %{z:.3f}<extra></extra>",
        )
    )
    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ13 — Drift History Calendar (365d)", font=dict(color=ACCENT_GRN, size=12)
    )
    layout["height"] = 220
    layout["xaxis"] = dict(
        title="Week",
        gridcolor=BORDER,
        linecolor=BORDER,
        tickfont=dict(color=TEXT_DIM, size=8),
        tickmode="auto",
        nticks=12,
    )
    layout["yaxis"] = dict(
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        tickvals=list(range(7)),
        gridcolor=BORDER,
        linecolor=BORDER,
        tickfont=dict(color=TEXT_DIM, size=8),
    )
    fig.update_layout(**layout)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div(
    style={"backgroundColor": BG_PRIMARY, "padding": "12px", "minHeight": "100vh"},
    children=[
        html.Div(
            "◈ Drift Analysis Lab",
            style={
                "color": ACCENT_GRN,
                "fontFamily": FONT,
                "fontSize": "14px",
                "fontWeight": "bold",
                "marginBottom": "12px",
                "borderBottom": f"1px solid {BORDER}",
                "paddingBottom": "6px",
            },
        ),
        build_viz08(),
        html.Div(
            [
                html.Div(build_viz09(), style={"flex": "1", "minWidth": "400px"}),
                html.Div(build_viz10(), style={"flex": "1.4", "minWidth": "500px"}),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
        html.Div(
            [
                html.Div(build_viz11(), style={"flex": "1", "minWidth": "400px"}),
                html.Div(build_viz12(), style={"flex": "1", "minWidth": "400px"}),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
        build_viz13(),
    ],
)
