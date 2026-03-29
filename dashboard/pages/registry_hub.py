"""
PAGE 3 – Registry Hub
VIZ14: Model version lifecycle graph (cytoscape)
VIZ15: A/B comparison chart
VIZ16: Hyperparameter importance (Optuna)
VIZ17: Model lineage timeline
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_cytoscape as cyto
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
    STATUS_COLORS,
)

cyto.load_extra_layouts()


# ---------------------------------------------------------------------------
# VIZ14 – Model version lifecycle graph
# ---------------------------------------------------------------------------
def build_viz14():
    elements = [
        # adult/xgboost versions
        {
            "data": {"id": "a-v1", "label": "adult\nv1.0", "stage": "archived"},
            "position": {"x": 80, "y": 60},
        },
        {
            "data": {"id": "a-v2", "label": "adult\nv2.0", "stage": "archived"},
            "position": {"x": 80, "y": 160},
        },
        {
            "data": {"id": "a-v3", "label": "adult\nv3.2", "stage": "production"},
            "position": {"x": 80, "y": 260},
        },
        {
            "data": {"id": "a-v4", "label": "adult\nv3.3", "stage": "staging"},
            "position": {"x": 80, "y": 360},
        },
        # wine_quality/xgboost versions
        {
            "data": {"id": "w-v1", "label": "wine\nv1.0", "stage": "archived"},
            "position": {"x": 260, "y": 60},
        },
        {
            "data": {"id": "w-v2", "label": "wine\nv2.1", "stage": "production"},
            "position": {"x": 260, "y": 160},
        },
        {
            "data": {"id": "w-v3", "label": "wine\nv2.2", "stage": "staging"},
            "position": {"x": 260, "y": 260},
        },
        # bike_sharing/lightgbm versions
        {
            "data": {"id": "b-v1", "label": "bike\nv3.8", "stage": "archived"},
            "position": {"x": 440, "y": 60},
        },
        {
            "data": {"id": "b-v2", "label": "bike\nv3.9", "stage": "archived"},
            "position": {"x": 440, "y": 160},
        },
        {
            "data": {"id": "b-v3", "label": "bike\nv4.0", "stage": "production"},
            "position": {"x": 440, "y": 260},
        },
        {
            "data": {"id": "b-v4", "label": "bike\nv4.1", "stage": "staging"},
            "position": {"x": 440, "y": 360},
        },
        # german_credit/neural_net versions
        {
            "data": {"id": "g-v1", "label": "german\nv1.3", "stage": "archived"},
            "position": {"x": 620, "y": 60},
        },
        {
            "data": {"id": "g-v2", "label": "german\nv1.4", "stage": "archived"},
            "position": {"x": 620, "y": 160},
        },
        {
            "data": {"id": "g-v3", "label": "german\nv1.5", "stage": "production"},
            "position": {"x": 620, "y": 260},
        },
        # Edges
        {"data": {"source": "a-v1", "target": "a-v2", "label": "retrain"}},
        {"data": {"source": "a-v2", "target": "a-v3", "label": "promote"}},
        {"data": {"source": "a-v3", "target": "a-v4", "label": "retrain"}},
        {"data": {"source": "w-v1", "target": "w-v2", "label": "promote"}},
        {"data": {"source": "w-v2", "target": "w-v3", "label": "retrain"}},
        {"data": {"source": "b-v1", "target": "b-v2", "label": "retrain"}},
        {"data": {"source": "b-v2", "target": "b-v3", "label": "promote"}},
        {"data": {"source": "b-v3", "target": "b-v4", "label": "retrain"}},
        {"data": {"source": "g-v1", "target": "g-v2", "label": "retrain"}},
        {"data": {"source": "g-v2", "target": "g-v3", "label": "promote"}},
    ]

    stylesheet = [
        {
            "selector": "node",
            "style": {
                "content": "data(label)",
                "text-wrap": "wrap",
                "text-valign": "center",
                "text-halign": "center",
                "background-color": BG_PANEL,
                "border-width": 2,
                "border-color": BORDER,
                "color": TEXT,
                "font-family": "JetBrains Mono, monospace",
                "font-size": "9px",
                "width": 70,
                "height": 50,
                "shape": "roundrectangle",
            },
        },
        {
            "selector": 'node[stage="production"]',
            "style": {
                "border-color": ACCENT_GRN,
                "background-color": ACCENT_DIM,
                "border-width": 3,
                "color": ACCENT_GRN,
                "font-weight": "bold",
            },
        },
        {
            "selector": 'node[stage="staging"]',
            "style": {
                "border-color": INFO,
                "background-color": "#051a2e",
                "color": INFO,
            },
        },
        {
            "selector": 'node[stage="archived"]',
            "style": {
                "border-color": TEXT_DIM,
                "background-color": BG_CARD,
                "color": TEXT_DIM,
                "opacity": 0.7,
            },
        },
        {
            "selector": "edge",
            "style": {
                "content": "data(label)",
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "arrow-scale": 1.2,
                "line-color": BORDER,
                "target-arrow-color": ACCENT_GRN,
                "color": TEXT_DIM,
                "font-size": "7px",
                "font-family": "JetBrains Mono, monospace",
                "text-rotation": "autorotate",
                "width": 1.5,
            },
        },
    ]

    legend_items = [
        ("Production", ACCENT_GRN, ACCENT_DIM),
        ("Staging", INFO, "#051a2e"),
        ("Archived", TEXT_DIM, BG_CARD),
    ]
    legend = html.Div(
        [
            html.Span(
                [
                    html.Span("■ ", style={"color": color}),
                    html.Span(label + "  ", style={"color": color, "marginRight": "12px"}),
                ]
            )
            for label, color, _ in legend_items
        ],
        style={"fontFamily": FONT, "fontSize": "10px", "marginBottom": "6px"},
    )

    return html.Div(
        [
            html.Div(
                "VIZ14 — Model Version Lifecycle Graph",
                style={
                    "color": ACCENT_GRN,
                    "fontSize": "12px",
                    "marginBottom": "6px",
                    "fontFamily": FONT,
                },
            ),
            legend,
            cyto.Cytoscape(
                id="viz14-cyto",
                elements=elements,
                stylesheet=stylesheet,
                layout={"name": "preset"},
                style={
                    "width": "100%",
                    "height": "430px",
                    "backgroundColor": BG_PANEL,
                    "border": f"1px solid {BORDER}",
                    "borderRadius": "4px",
                },
                userZoomingEnabled=True,
                userPanningEnabled=True,
            ),
        ],
        style=CARD_STYLE,
    )


# ---------------------------------------------------------------------------
# VIZ15 – A/B comparison: champion vs challenger
# ---------------------------------------------------------------------------
def build_viz15():
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall", "AUC-ROC", "Log Loss"]
    champion = [0.873, 0.861, 0.878, 0.844, 0.921, 0.284]
    challenger = [0.881, 0.872, 0.883, 0.861, 0.934, 0.261]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Champion (adult/xgboost v3.2)",
            x=metrics,
            y=champion,
            marker_color=INFO,
            marker_line_color=BORDER,
            marker_line_width=1,
            text=[f"{v:.3f}" for v in champion],
            textposition="outside",
            textfont=dict(color=INFO, size=9, family=FONT),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Challenger (adult/xgboost v3.3)",
            x=metrics,
            y=challenger,
            marker_color=ACCENT_GRN,
            marker_line_color=BORDER,
            marker_line_width=1,
            text=[f"{v:.3f}" for v in challenger],
            textposition="outside",
            textfont=dict(color=ACCENT_GRN, size=9, family=FONT),
        )
    )
    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ15 — Champion vs Challenger Metrics", font=dict(color=ACCENT_GRN, size=12)
    )
    layout["barmode"] = "group"
    layout["height"] = 320
    layout["yaxis"] = {**PLOT_LAYOUT.get("yaxis", {}), "range": [0.2, 1.05], "title": "Score"}
    layout["xaxis"] = {**PLOT_LAYOUT.get("xaxis", {}), "title": ""}
    fig.update_layout(**layout)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ16 – Hyperparameter importance (Optuna)
# ---------------------------------------------------------------------------
def build_viz16():
    params = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_child_weight",
    ]
    importance = np.array([0.342, 0.287, 0.198, 0.068, 0.051, 0.028, 0.017, 0.009])
    colors = [
        ACCENT_GRN if imp > 0.2 else (WARNING if imp > 0.05 else TEXT_DIM) for imp in importance
    ]
    sorted_idx = np.argsort(importance)

    fig = go.Figure(
        go.Bar(
            x=importance[sorted_idx],
            y=[params[i] for i in sorted_idx],
            orientation="h",
            marker_color=[colors[i] for i in sorted_idx],
            marker_line_color=BORDER,
            marker_line_width=1,
            text=[f"{importance[i]:.3f}" for i in sorted_idx],
            textposition="outside",
            textfont=dict(color=TEXT, size=9, family=FONT),
        )
    )
    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ16 — Hyperparameter Importance (Optuna FAnova)",
        font=dict(color=ACCENT_GRN, size=12),
    )
    layout["height"] = 320
    layout["xaxis"] = {
        **PLOT_LAYOUT.get("xaxis", {}),
        "title": "Importance Score",
        "range": [0, 0.42],
    }
    layout["yaxis"] = {**PLOT_LAYOUT.get("yaxis", {}), "title": ""}
    fig.update_layout(**layout)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ17 – Model lineage timeline
# ---------------------------------------------------------------------------
def build_viz17():
    events = [
        ("adult/xgboost", "v1.0", "2024-01-10", "Initial train", SUCCESS, 0.831),
        ("adult/xgboost", "v2.0", "2024-02-18", "Hyperopt sweep", SUCCESS, 0.858),
        ("adult/xgboost", "v3.0", "2024-03-25", "Feature engineering", ACCENT_GRN, 0.866),
        ("adult/xgboost", "v3.1", "2024-04-20", "Data refresh", ACCENT_GRN, 0.871),
        ("adult/xgboost", "v3.2", "2024-05-12", "Drift retrain", ACCENT_GRN, 0.873),
        ("wine_quality/xgboost", "v1.0", "2024-01-15", "Initial train", SUCCESS, 0.771),
        ("wine_quality/xgboost", "v2.0", "2024-03-01", "Feature refresh", SUCCESS, 0.785),
        ("wine_quality/xgboost", "v2.1", "2024-05-10", "Drift retrain", SUCCESS, 0.791),
        ("bike_sharing/lightgbm", "v3.8", "2024-01-20", "Baseline", WARNING, 0.888),
        ("bike_sharing/lightgbm", "v3.9", "2024-03-10", "Hyperopt sweep", WARNING, 0.901),
        ("bike_sharing/lightgbm", "v4.0", "2024-05-08", "Perf. degradation", WARNING, 0.912),
        ("german_credit/neural_net", "v1.3", "2024-02-05", "Architecture v2", INFO, 0.798),
        ("german_credit/neural_net", "v1.4", "2024-04-01", "Data augment", INFO, 0.809),
        ("german_credit/neural_net", "v1.5", "2024-05-14", "Scheduled retrain", INFO, 0.816),
    ]

    model_y = {
        "adult/xgboost": 3,
        "wine_quality/xgboost": 2,
        "bike_sharing/lightgbm": 1,
        "german_credit/neural_net": 0,
    }

    fig = go.Figure()
    for model, y_pos in model_y.items():
        model_events = [e for e in events if e[0] == model]
        x_vals = [pd.to_datetime(e[2]) for e in model_events]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=[y_pos] * len(model_events),
                mode="lines",
                line=dict(color=model_events[0][4], width=1.5, dash="dot"),
                showlegend=False,
            )
        )
        for ev in model_events:
            ev_date = pd.to_datetime(ev[2])
            fig.add_trace(
                go.Scatter(
                    x=[ev_date],
                    y=[y_pos],
                    mode="markers+text",
                    marker=dict(
                        color=ev[4], size=12, symbol="circle", line=dict(color=BG_CARD, width=2)
                    ),
                    text=[ev[1]],
                    textposition="top center",
                    textfont=dict(color=ev[4], size=8, family=FONT),
                    name=f"{ev[0]} {ev[1]}",
                    showlegend=False,
                    hovertemplate=f"<b>{ev[0]}</b><br>Version: {ev[1]}<br>Date: {ev[2]}<br>Reason: {ev[3]}<br>Accuracy: {ev[5]:.3f}<extra></extra>",
                )
            )

    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ17 — Model Lineage Timeline", font=dict(color=ACCENT_GRN, size=12)
    )
    layout["height"] = 340
    layout["yaxis"] = dict(
        ticktext=["german_credit", "bike_sharing", "wine_quality", "adult"],
        tickvals=[0, 1, 2, 3],
        gridcolor=BORDER,
        linecolor=BORDER,
        tickfont=dict(color=TEXT, size=9, family=FONT),
    )
    layout["xaxis"] = {**PLOT_LAYOUT.get("xaxis", {}), "title": "Date"}
    fig.update_layout(**layout)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div(
    style={"backgroundColor": BG_PRIMARY, "padding": "12px", "minHeight": "100vh"},
    children=[
        html.Div(
            "◈ Registry Hub",
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
        html.Div(
            [
                html.Div(build_viz14(), style={"flex": "1.2", "minWidth": "500px"}),
                html.Div(
                    [
                        build_viz15(),
                        build_viz16(),
                    ],
                    style={"flex": "1", "minWidth": "400px"},
                ),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
        build_viz17(),
    ],
)
