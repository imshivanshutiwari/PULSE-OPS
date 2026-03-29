"""
PAGE 1 – MLOps Ops Center
VIZ01: Live model status board
VIZ02: Model performance timeline
VIZ03: Drift score gauges
VIZ04: Retraining activity log
VIZ05: Prefect flow run status
VIZ06: MLflow experiment tracker
VIZ07: Feature store usage metrics
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc
import sys
import os

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

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------
MODELS = [
    "adult/xgboost",
    "wine_quality/xgboost",
    "bike_sharing/lightgbm",
    "german_credit/neural_net",
]
MODEL_SHORT = ["adult", "wine_quality", "bike_sharing", "german_credit"]
ALGO_COLORS = [ACCENT_GRN, INFO, WARNING, SUCCESS]
STATUS_MAP = ["HEALTHY", "HEALTHY", "DEGRADED", "HEALTHY"]
STATUS_COLORS_MAP = [SUCCESS, SUCCESS, WARNING, SUCCESS]

np.random.seed(42)
DAYS = 30
dates = pd.date_range(end=datetime.date.today(), periods=DAYS, freq="D")


def _perf_series(base_acc, base_f1, noise=0.012):
    acc = np.clip(base_acc + np.cumsum(np.random.randn(DAYS) * noise) * 0.1, 0.70, 0.99)
    f1 = np.clip(base_f1 + np.cumsum(np.random.randn(DAYS) * noise) * 0.1, 0.68, 0.99)
    return acc, f1


perf_data = {
    "adult/xgboost": _perf_series(0.873, 0.861),
    "wine_quality/xgboost": _perf_series(0.791, 0.778),
    "bike_sharing/lightgbm": _perf_series(0.912, 0.905),
    "german_credit/neural_net": _perf_series(0.816, 0.803),
}


# ---------------------------------------------------------------------------
# VIZ01 – Live model status board
# ---------------------------------------------------------------------------
def build_viz01():
    model_info = [
        {
            "name": "adult/xgboost",
            "version": "v3.2",
            "accuracy": 0.873,
            "f1": 0.861,
            "last_train": "2024-05-12",
            "status": "HEALTHY",
            "drift": 0.042,
            "requests": 12847,
            "color": SUCCESS,
        },
        {
            "name": "wine_quality/xgboost",
            "version": "v2.1",
            "accuracy": 0.791,
            "f1": 0.778,
            "last_train": "2024-05-10",
            "status": "HEALTHY",
            "drift": 0.063,
            "requests": 4321,
            "color": SUCCESS,
        },
        {
            "name": "bike_sharing/lightgbm",
            "version": "v4.0",
            "accuracy": 0.912,
            "f1": 0.905,
            "last_train": "2024-05-08",
            "status": "DEGRADED",
            "drift": 0.187,
            "requests": 28943,
            "color": WARNING,
        },
        {
            "name": "german_credit/neural_net",
            "version": "v1.5",
            "accuracy": 0.816,
            "f1": 0.803,
            "last_train": "2024-05-14",
            "status": "HEALTHY",
            "drift": 0.031,
            "requests": 7652,
            "color": SUCCESS,
        },
    ]

    cards = []
    for m in model_info:
        status_color = m["color"]
        border_color = status_color
        card = html.Div(
            style={
                "backgroundColor": BG_CARD,
                "border": f"1px solid {border_color}",
                "borderLeft": f"4px solid {border_color}",
                "borderRadius": "4px",
                "padding": "14px",
                "flex": "1",
                "minWidth": "220px",
                "fontFamily": FONT,
            },
            children=[
                html.Div(
                    m["name"], style={"color": ACCENT_GRN, "fontSize": "13px", "fontWeight": "bold"}
                ),
                html.Div(
                    m["version"],
                    style={"color": TEXT_DIM, "fontSize": "10px", "marginBottom": "8px"},
                ),
                html.Div(
                    [
                        html.Span("● ", style={"color": status_color}),
                        html.Span(
                            m["status"],
                            style={"color": status_color, "fontSize": "11px", "fontWeight": "bold"},
                        ),
                    ],
                    style={"marginBottom": "6px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("ACC  ", style={"color": TEXT_DIM}),
                                html.Span(f"{m['accuracy']:.3f}", style={"color": TEXT}),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("F1   ", style={"color": TEXT_DIM}),
                                html.Span(f"{m['f1']:.3f}", style={"color": TEXT}),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("DRIFT ", style={"color": TEXT_DIM}),
                                html.Span(
                                    f"{m['drift']:.3f}",
                                    style={"color": WARNING if m["drift"] > 0.1 else SUCCESS},
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("REQS  ", style={"color": TEXT_DIM}),
                                html.Span(f"{m['requests']:,}", style={"color": INFO}),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("TRAIN ", style={"color": TEXT_DIM}),
                                html.Span(m["last_train"], style={"color": TEXT_DIM}),
                            ]
                        ),
                    ],
                    style={"fontSize": "11px", "lineHeight": "1.8"},
                ),
            ],
        )
        cards.append(card)

    return html.Div(
        [
            html.Div(
                "VIZ01 — Live Model Status Board",
                style={
                    "color": ACCENT_GRN,
                    "fontSize": "12px",
                    "marginBottom": "8px",
                    "fontFamily": FONT,
                },
            ),
            html.Div(cards, style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ],
        style={**CARD_STYLE},
    )


# ---------------------------------------------------------------------------
# VIZ02 – Model performance timeline
# ---------------------------------------------------------------------------
def build_viz02():
    fig = go.Figure()
    for i, (model, (acc, f1)) in enumerate(perf_data.items()):
        color = ALGO_COLORS[i]
        short = MODEL_SHORT[i]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=acc,
                name=f"{short} ACC",
                line=dict(color=color, width=2),
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=f1,
                name=f"{short} F1",
                line=dict(color=color, width=1.5, dash="dot"),
                mode="lines",
            )
        )
    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ02 — Model Performance Timeline (30d)", font=dict(color=ACCENT_GRN, size=12)
    )
    layout["yaxis"] = {**PLOT_LAYOUT.get("yaxis", {}), "title": "Score", "range": [0.65, 1.0]}
    layout["xaxis"] = {**PLOT_LAYOUT.get("xaxis", {}), "title": "Date"}
    layout["height"] = 300
    fig.update_layout(**layout)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ03 – Drift score gauges
# ---------------------------------------------------------------------------
def build_viz03():
    gauge_specs = [
        ("Data Drift", 0.187, 0.25, 0.15),
        ("Perf Drop", 0.063, 0.15, 0.08),
        ("Target Drift p-val", 0.031, 0.05, 0.03),
        ("Quality Score", 0.923, 0.80, 0.90),
    ]
    fig = make_subplots(rows=1, cols=4, specs=[[{"type": "indicator"}] * 4])
    for idx, (label, val, warn_thresh, ok_thresh) in enumerate(gauge_specs, 1):
        if label == "Quality Score":
            color = SUCCESS if val >= ok_thresh else (WARNING if val >= warn_thresh else CRITICAL)
        else:
            color = SUCCESS if val <= ok_thresh else (WARNING if val <= warn_thresh else CRITICAL)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val,
                title=dict(text=label, font=dict(color=TEXT, size=10, family=FONT)),
                number=dict(font=dict(color=color, size=18, family=FONT)),
                gauge=dict(
                    axis=dict(
                        range=[0, 1] if label != "Quality Score" else [0, 1],
                        tickcolor=TEXT_DIM,
                        tickfont=dict(color=TEXT_DIM, size=8),
                    ),
                    bar=dict(color=color),
                    bgcolor=BG_PANEL,
                    bordercolor=BORDER,
                    steps=[
                        dict(
                            range=[0, ok_thresh if label != "Quality Score" else warn_thresh],
                            color=ACCENT_DIM,
                        ),
                        dict(
                            range=[
                                ok_thresh if label != "Quality Score" else warn_thresh,
                                warn_thresh if label != "Quality Score" else ok_thresh,
                            ],
                            color="#1a2010",
                        ),
                    ],
                    threshold=dict(
                        line=dict(color=CRITICAL, width=2),
                        thickness=0.75,
                        value=warn_thresh,
                    ),
                ),
            ),
            row=1,
            col=idx,
        )
    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(text="VIZ03 — Drift Score Gauges", font=dict(color=ACCENT_GRN, size=12))
    layout["height"] = 220
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(**layout)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ04 – Retraining activity log
# ---------------------------------------------------------------------------
def build_viz04():
    log_data = [
        (
            "2024-05-14 09:12",
            "drift_threshold",
            "german_credit/neural_net",
            "v1.4→v1.5",
            "PASS",
            "14m 32s",
        ),
        ("2024-05-12 14:33", "scheduled_daily", "adult/xgboost", "v3.1→v3.2", "PASS", "8m 17s"),
        (
            "2024-05-10 07:55",
            "drift_threshold",
            "wine_quality/xgboost",
            "v2.0→v2.1",
            "PASS",
            "6m 04s",
        ),
        (
            "2024-05-08 18:01",
            "perf_degradation",
            "bike_sharing/lightgbm",
            "v3.9→v4.0",
            "PASS",
            "22m 11s",
        ),
        ("2024-05-07 11:22", "manual", "adult/xgboost", "v3.0→v3.1", "PASS", "9m 48s"),
        (
            "2024-05-05 03:40",
            "scheduled_daily",
            "german_credit/neural_net",
            "v1.3→v1.4",
            "PASS",
            "13m 59s",
        ),
        (
            "2024-05-03 16:15",
            "drift_threshold",
            "bike_sharing/lightgbm",
            "v3.8→v3.9",
            "FAIL",
            "4m 02s",
        ),
        (
            "2024-05-01 09:00",
            "scheduled_daily",
            "wine_quality/xgboost",
            "v1.9→v2.0",
            "PASS",
            "7m 33s",
        ),
    ]
    headers = ["Timestamp", "Trigger", "Model", "Versions", "Gate", "Duration"]
    gate_col_colors = []
    for row in log_data:
        gate_col_colors.append(SUCCESS if row[4] == "PASS" else CRITICAL)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{h}</b>" for h in headers],
                    fill_color=ACCENT_DIM,
                    font=dict(color=ACCENT_GRN, size=11, family=FONT),
                    align="left",
                    line_color=BORDER,
                ),
                cells=dict(
                    values=[
                        [r[0] for r in log_data],
                        [r[1] for r in log_data],
                        [r[2] for r in log_data],
                        [r[3] for r in log_data],
                        [r[4] for r in log_data],
                        [r[5] for r in log_data],
                    ],
                    fill_color=[
                        [BG_CARD] * len(log_data),
                        [BG_CARD] * len(log_data),
                        [BG_CARD] * len(log_data),
                        [BG_CARD] * len(log_data),
                        gate_col_colors,
                        [BG_CARD] * len(log_data),
                    ],
                    font=dict(
                        color=[
                            [TEXT] * len(log_data),
                            [INFO] * len(log_data),
                            [ACCENT_GRN] * len(log_data),
                            [TEXT_DIM] * len(log_data),
                            ["#050a06"] * len(log_data),
                            [TEXT] * len(log_data),
                        ],
                        size=10,
                        family=FONT,
                    ),
                    align="left",
                    line_color=BORDER,
                ),
            )
        ]
    )
    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ04 — Retraining Activity Log", font=dict(color=ACCENT_GRN, size=12)
    )
    layout["height"] = 310
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(**layout)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ05 – Prefect flow run status (Gantt)
# ---------------------------------------------------------------------------
def build_viz05():
    now = datetime.datetime.now()
    flows = [
        ("data_ingestion", now - datetime.timedelta(hours=2, minutes=15), 42, SUCCESS, "Completed"),
        (
            "feature_engineering",
            now - datetime.timedelta(hours=1, minutes=50),
            67,
            SUCCESS,
            "Completed",
        ),
        (
            "model_training",
            now - datetime.timedelta(hours=1, minutes=10),
            1398,
            ACCENT_GRN,
            "Completed",
        ),
        ("drift_detection", now - datetime.timedelta(minutes=30), 18, SUCCESS, "Completed"),
        ("model_evaluation", now - datetime.timedelta(minutes=20), 234, WARNING, "Running"),
        ("registry_promotion", now - datetime.timedelta(minutes=8), 0, TEXT_DIM, "Pending"),
        ("notification_dispatch", now - datetime.timedelta(minutes=5), 0, TEXT_DIM, "Pending"),
    ]

    fig = go.Figure()
    for i, (name, start, dur_sec, color, state) in enumerate(flows):
        end = start + datetime.timedelta(seconds=dur_sec if dur_sec > 0 else 5)
        fig.add_trace(
            go.Bar(
                y=[name],
                x=[(end - start).total_seconds() / 60],
                base=[(start - (now - datetime.timedelta(hours=3))).total_seconds() / 60],
                orientation="h",
                marker_color=color,
                marker_line_color=BORDER,
                marker_line_width=1,
                name=state,
                showlegend=False,
                hovertemplate=(
                    f"<b>{name}</b><br>State: {state}<br>"
                    f"Duration: {dur_sec}s<extra></extra>"
                ),
            )
        )
        fig.add_annotation(
            x=(
                (start - (now - datetime.timedelta(hours=3))).total_seconds() / 60
                + (end - start).total_seconds() / 120
            ),
            y=name,
            text=state,
            showarrow=False,
            font=dict(color="#050a06", size=9, family=FONT),
        )
    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ05 — Prefect Flow Run Status", font=dict(color=ACCENT_GRN, size=12)
    )
    layout["xaxis"] = {**PLOT_LAYOUT.get("xaxis", {}), "title": "Minutes from pipeline start"}
    layout["yaxis"] = {**PLOT_LAYOUT.get("yaxis", {}), "title": ""}
    layout["height"] = 300
    layout["barmode"] = "overlay"
    fig.update_layout(**layout)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ06 – MLflow experiment tracker (parallel coordinates)
# ---------------------------------------------------------------------------
def build_viz06():
    n = 40
    rng = np.random.default_rng(7)
    lr = rng.uniform(0.005, 0.3, n)
    depth = rng.integers(3, 10, n).astype(float)
    nest = rng.integers(50, 400, n).astype(float)
    reg = rng.uniform(0.0, 1.0, n)
    acc = np.clip(0.82 - 0.3 * lr + 0.008 * depth + rng.normal(0, 0.015, n), 0.72, 0.95)
    f1 = acc - rng.uniform(0.005, 0.025, n)
    auc = np.clip(acc + rng.uniform(0.01, 0.04, n), 0.73, 0.99)

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=acc,
                colorscale=[[0, CRITICAL], [0.5, WARNING], [1, SUCCESS]],
                showscale=True,
                cmin=acc.min(),
                cmax=acc.max(),
                colorbar=dict(
                    title=dict(text="Accuracy", font=dict(color=TEXT, family=FONT)),
                    tickfont=dict(color=TEXT, family=FONT),
                    bgcolor=BG_CARD,
                ),
            ),
            dimensions=[
                dict(label="Learning Rate", values=lr, range=[0.005, 0.3]),
                dict(label="Max Depth", values=depth, range=[3, 10]),
                dict(label="N Estimators", values=nest, range=[50, 400]),
                dict(label="Reg Alpha", values=reg, range=[0, 1]),
                dict(label="Accuracy", values=acc, range=[0.72, 0.95]),
                dict(label="F1 Score", values=f1, range=[0.70, 0.93]),
                dict(label="AUC-ROC", values=auc, range=[0.73, 0.99]),
            ],
            labelfont=dict(color=TEXT, size=10, family=FONT),
            tickfont=dict(color=TEXT_DIM, size=8, family=FONT),
            rangefont=dict(color=TEXT_DIM, size=8, family=FONT),
        )
    )
    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ06 — MLflow Experiment Tracker (Parallel Coords)",
        font=dict(color=ACCENT_GRN, size=12),
    )
    layout["height"] = 340
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(**layout)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ07 – Feature store usage metrics
# ---------------------------------------------------------------------------
def build_viz07():
    rng = np.random.default_rng(9)
    hours = pd.date_range(end=datetime.datetime.now(), periods=48, freq="30min")
    req_rate = 800 + 400 * np.sin(np.linspace(0, 4 * np.pi, 48)) + rng.normal(0, 30, 48)
    freshness = np.clip(0.98 + rng.normal(0, 0.005, 48), 0.93, 1.0)
    latency_p50 = 12 + 4 * np.sin(np.linspace(0, 4 * np.pi, 48)) + rng.normal(0, 1, 48)
    latency_p99 = 45 + 15 * np.sin(np.linspace(0, 4 * np.pi, 48)) + rng.normal(0, 3, 48)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Request Rate (req/s)", "Feature Freshness", "Latency (ms)"],
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=req_rate,
            fill="tozeroy",
            fillcolor=ACCENT_DIM,
            line=dict(color=ACCENT_GRN, width=1.5),
            name="Req/s",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=freshness,
            fill="tozeroy",
            fillcolor="#0a1e04",
            line=dict(color=SUCCESS, width=1.5),
            name="Freshness",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=hours, y=latency_p50, line=dict(color=INFO, width=1.5), name="P50 ms"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=hours, y=latency_p99, line=dict(color=WARNING, width=1.5), name="P99 ms"),
        row=3,
        col=1,
    )

    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ07 — Feature Store Usage Metrics (48h)", font=dict(color=ACCENT_GRN, size=12)
    )
    layout["height"] = 420
    for k in ["xaxis", "yaxis"]:
        layout.pop(k, None)
    for i in range(1, 4):
        suffix = "" if i == 1 else str(i)
        layout[f"xaxis{suffix}"] = dict(
            gridcolor=BORDER,
            linecolor=BORDER,
            tickcolor=BORDER,
            zerolinecolor=BORDER,
            tickfont=dict(color=TEXT_DIM),
        )
        layout[f"yaxis{suffix}"] = dict(
            gridcolor=BORDER,
            linecolor=BORDER,
            tickcolor=BORDER,
            zerolinecolor=BORDER,
            tickfont=dict(color=TEXT),
        )
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(color=TEXT, size=10, family=FONT))
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# Layout assembly
# ---------------------------------------------------------------------------
layout = html.Div(
    style={"backgroundColor": BG_PRIMARY, "padding": "12px", "minHeight": "100vh"},
    children=[
        html.Div(
            "◈ MLOps Ops Center",
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
        build_viz01(),
        html.Div(
            [
                html.Div(build_viz02(), style={"flex": "1", "minWidth": "400px"}),
                html.Div(build_viz03(), style={"flex": "1", "minWidth": "400px"}),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
        build_viz04(),
        html.Div(
            [
                html.Div(build_viz05(), style={"flex": "1", "minWidth": "400px"}),
                html.Div(build_viz06(), style={"flex": "1.5", "minWidth": "500px"}),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
        build_viz07(),
    ],
)
