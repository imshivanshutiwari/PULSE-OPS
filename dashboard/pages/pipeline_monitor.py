"""
PAGE 4 – Pipeline Monitor
VIZ18: Prefect flow dependency graph (cytoscape)
VIZ19: Pipeline success rate (stacked bar)
VIZ20: DVC data version tree (cytoscape)
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
)

cyto.load_extra_layouts()


# ---------------------------------------------------------------------------
# VIZ18 – Prefect flow dependency graph
# ---------------------------------------------------------------------------
def build_viz18():
    elements = [
        # Flow nodes
        {
            "data": {
                "id": "flow-ingest",
                "label": "data_ingestion",
                "type": "flow",
                "status": "success",
            }
        },
        {
            "data": {
                "id": "flow-feat",
                "label": "feature_engineering",
                "type": "flow",
                "status": "success",
            }
        },
        {
            "data": {
                "id": "flow-train",
                "label": "model_training",
                "type": "flow",
                "status": "running",
            }
        },
        {
            "data": {
                "id": "flow-eval",
                "label": "model_evaluation",
                "type": "flow",
                "status": "pending",
            }
        },
        {
            "data": {
                "id": "flow-drift",
                "label": "drift_detection",
                "type": "flow",
                "status": "success",
            }
        },
        {
            "data": {
                "id": "flow-reg",
                "label": "registry_promotion",
                "type": "flow",
                "status": "pending",
            }
        },
        {
            "data": {
                "id": "flow-notify",
                "label": "notification_dispatch",
                "type": "flow",
                "status": "pending",
            }
        },
        # Task sub-nodes for data_ingestion
        {
            "data": {
                "id": "t-fetch",
                "label": "fetch_raw",
                "type": "task",
                "status": "success",
                "parent": "flow-ingest",
            }
        },
        {
            "data": {
                "id": "t-valid",
                "label": "validate_schema",
                "type": "task",
                "status": "success",
                "parent": "flow-ingest",
            }
        },
        {
            "data": {
                "id": "t-store",
                "label": "store_dvc",
                "type": "task",
                "status": "success",
                "parent": "flow-ingest",
            }
        },
        # Task sub-nodes for feature_engineering
        {
            "data": {
                "id": "t-impute",
                "label": "impute_missing",
                "type": "task",
                "status": "success",
                "parent": "flow-feat",
            }
        },
        {
            "data": {
                "id": "t-encode",
                "label": "encode_cats",
                "type": "task",
                "status": "success",
                "parent": "flow-feat",
            }
        },
        {
            "data": {
                "id": "t-scale",
                "label": "scale_feats",
                "type": "task",
                "status": "success",
                "parent": "flow-feat",
            }
        },
        {
            "data": {
                "id": "t-feast",
                "label": "feast_push",
                "type": "task",
                "status": "success",
                "parent": "flow-feat",
            }
        },
        # Task sub-nodes for model_training
        {
            "data": {
                "id": "t-optuna",
                "label": "optuna_sweep",
                "type": "task",
                "status": "running",
                "parent": "flow-train",
            }
        },
        {
            "data": {
                "id": "t-fit",
                "label": "model_fit",
                "type": "task",
                "status": "pending",
                "parent": "flow-train",
            }
        },
        {
            "data": {
                "id": "t-mlflow",
                "label": "mlflow_log",
                "type": "task",
                "status": "pending",
                "parent": "flow-train",
            }
        },
        # Flow-level edges
        {"data": {"source": "flow-ingest", "target": "flow-feat", "label": ""}},
        {"data": {"source": "flow-feat", "target": "flow-train", "label": ""}},
        {"data": {"source": "flow-feat", "target": "flow-drift", "label": ""}},
        {"data": {"source": "flow-train", "target": "flow-eval", "label": ""}},
        {"data": {"source": "flow-eval", "target": "flow-reg", "label": ""}},
        {"data": {"source": "flow-reg", "target": "flow-notify", "label": ""}},
        # Task edges
        {"data": {"source": "t-fetch", "target": "t-valid", "label": ""}},
        {"data": {"source": "t-valid", "target": "t-store", "label": ""}},
        {"data": {"source": "t-impute", "target": "t-encode", "label": ""}},
        {"data": {"source": "t-encode", "target": "t-scale", "label": ""}},
        {"data": {"source": "t-scale", "target": "t-feast", "label": ""}},
        {"data": {"source": "t-optuna", "target": "t-fit", "label": ""}},
        {"data": {"source": "t-fit", "target": "t-mlflow", "label": ""}},
    ]

    status_colors = {
        "success": ACCENT_GRN,
        "running": INFO,
        "pending": TEXT_DIM,
        "failed": CRITICAL,
    }

    stylesheet = [
        {
            "selector": 'node[type="flow"]',
            "style": {
                "content": "data(label)",
                "text-wrap": "wrap",
                "text-valign": "center",
                "text-halign": "center",
                "background-color": BG_PANEL,
                "border-width": 2,
                "font-family": "JetBrains Mono, monospace",
                "font-size": "9px",
                "width": 120,
                "height": 40,
                "shape": "roundrectangle",
            },
        },
        {
            "selector": 'node[type="task"]',
            "style": {
                "content": "data(label)",
                "text-valign": "center",
                "text-halign": "center",
                "background-color": BG_CARD,
                "border-width": 1.5,
                "font-family": "JetBrains Mono, monospace",
                "font-size": "8px",
                "width": 80,
                "height": 28,
                "shape": "rectangle",
            },
        },
        *[
            {
                "selector": f'node[status="{status}"]',
                "style": {
                    "border-color": color,
                    "color": color,
                },
            }
            for status, color in status_colors.items()
        ],
        {
            "selector": 'node[status="running"]',
            "style": {
                "border-width": 3,
                "border-color": INFO,
                "background-color": "#051a2e",
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "line-color": BORDER,
                "target-arrow-color": ACCENT_GRN,
                "width": 1.5,
                "arrow-scale": 1.0,
            },
        },
    ]

    legend = html.Div(
        [
            html.Span(
                [
                    html.Span("● ", style={"color": color}),
                    html.Span(
                        label + "  ",
                        style={
                            "color": color,
                            "marginRight": "10px",
                            "fontFamily": FONT,
                            "fontSize": "10px",
                        },
                    ),
                ]
            )
            for label, color in [
                ("Success", ACCENT_GRN),
                ("Running", INFO),
                ("Pending", TEXT_DIM),
                ("Failed", CRITICAL),
            ]
        ],
        style={"marginBottom": "6px"},
    )

    return html.Div(
        [
            html.Div(
                "VIZ18 — Prefect Flow Dependency Graph",
                style={
                    "color": ACCENT_GRN,
                    "fontSize": "12px",
                    "marginBottom": "6px",
                    "fontFamily": FONT,
                },
            ),
            legend,
            cyto.Cytoscape(
                id="viz18-cyto",
                elements=elements,
                stylesheet=stylesheet,
                layout={
                    "name": "breadthfirst",
                    "directed": True,
                    "padding": 20,
                    "spacingFactor": 1.4,
                },
                style={
                    "width": "100%",
                    "height": "480px",
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
# VIZ19 – Pipeline success rate stacked bar
# ---------------------------------------------------------------------------
def build_viz19():
    rng = np.random.default_rng(55)
    days = pd.date_range(end=datetime.date.today(), periods=30, freq="D")
    total = rng.integers(18, 28, 30)
    failed = rng.integers(0, 3, 30)
    retried = rng.integers(0, 2, 30)
    success = total - failed - retried

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Success",
            x=days,
            y=success,
            marker_color=ACCENT_GRN,
            marker_line_width=0,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Retried",
            x=days,
            y=retried,
            marker_color=WARNING,
            marker_line_width=0,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Failed",
            x=days,
            y=failed,
            marker_color=CRITICAL,
            marker_line_width=0,
        )
    )
    layout = {**PLOT_LAYOUT}
    layout["title"] = dict(
        text="VIZ19 — Pipeline Run Outcomes (30d)", font=dict(color=ACCENT_GRN, size=12)
    )
    layout["barmode"] = "stack"
    layout["height"] = 300
    layout["xaxis"] = {**PLOT_LAYOUT.get("xaxis", {}), "title": "Date"}
    layout["yaxis"] = {**PLOT_LAYOUT.get("yaxis", {}), "title": "Run count"}
    fig.update_layout(**layout)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style=CARD_STYLE)


# ---------------------------------------------------------------------------
# VIZ20 – DVC data version tree
# ---------------------------------------------------------------------------
def build_viz20():
    elements = [
        # Root dataset
        {
            "data": {
                "id": "raw-v1",
                "label": "raw_data\nv1.0\n2024-01-01",
                "type": "root",
                "status": "archived",
            }
        },
        # Processed splits
        {
            "data": {
                "id": "proc-v1",
                "label": "processed\nv1.0",
                "type": "processed",
                "status": "archived",
            }
        },
        {
            "data": {
                "id": "proc-v2",
                "label": "processed\nv2.0",
                "type": "processed",
                "status": "archived",
            }
        },
        {
            "data": {
                "id": "proc-v3",
                "label": "processed\nv3.0",
                "type": "processed",
                "status": "current",
            }
        },
        # Feature sets
        {
            "data": {
                "id": "feat-v1",
                "label": "features\nv1.0\n2024-01",
                "type": "features",
                "status": "archived",
            }
        },
        {
            "data": {
                "id": "feat-v2",
                "label": "features\nv2.0\n2024-03",
                "type": "features",
                "status": "archived",
            }
        },
        {
            "data": {
                "id": "feat-v3",
                "label": "features\nv3.0\n2024-05",
                "type": "features",
                "status": "current",
            }
        },
        # Train/val/test splits from latest
        {
            "data": {
                "id": "train-latest",
                "label": "train_set\n80%\n2024-05",
                "type": "split",
                "status": "current",
            }
        },
        {
            "data": {
                "id": "val-latest",
                "label": "val_set\n10%\n2024-05",
                "type": "split",
                "status": "current",
            }
        },
        {
            "data": {
                "id": "test-latest",
                "label": "test_set\n10%\n2024-05",
                "type": "split",
                "status": "current",
            }
        },
        # Raw refreshes
        {
            "data": {
                "id": "raw-v2",
                "label": "raw_data\nv2.0\n2024-03-01",
                "type": "root",
                "status": "archived",
            }
        },
        {
            "data": {
                "id": "raw-v3",
                "label": "raw_data\nv3.0\n2024-05-01",
                "type": "root",
                "status": "current",
            }
        },
        # Edges
        {"data": {"source": "raw-v1", "target": "proc-v1"}},
        {"data": {"source": "raw-v2", "target": "proc-v2"}},
        {"data": {"source": "raw-v3", "target": "proc-v3"}},
        {"data": {"source": "raw-v1", "target": "raw-v2"}},
        {"data": {"source": "raw-v2", "target": "raw-v3"}},
        {"data": {"source": "proc-v1", "target": "feat-v1"}},
        {"data": {"source": "proc-v2", "target": "feat-v2"}},
        {"data": {"source": "proc-v3", "target": "feat-v3"}},
        {"data": {"source": "feat-v3", "target": "train-latest"}},
        {"data": {"source": "feat-v3", "target": "val-latest"}},
        {"data": {"source": "feat-v3", "target": "test-latest"}},
    ]

    type_shapes = {
        "root": "diamond",
        "processed": "roundrectangle",
        "features": "hexagon",
        "split": "rectangle",
    }
    type_colors = {
        "root": "#1a2a0a",
        "processed": BG_PANEL,
        "features": "#0a1a1a",
        "split": "#0a0a1a",
    }
    stylesheet = [
        {
            "selector": "node",
            "style": {
                "content": "data(label)",
                "text-wrap": "wrap",
                "text-valign": "center",
                "text-halign": "center",
                "border-width": 1.5,
                "border-color": BORDER,
                "color": TEXT_DIM,
                "font-family": "JetBrains Mono, monospace",
                "font-size": "8px",
                "width": 80,
                "height": 55,
            },
        },
        *[
            {
                "selector": f'node[type="{t}"]',
                "style": {"shape": shape, "background-color": type_colors[t]},
            }
            for t, shape in type_shapes.items()
        ],
        {
            "selector": 'node[status="current"]',
            "style": {
                "border-color": ACCENT_GRN,
                "border-width": 2.5,
                "color": ACCENT_GRN,
            },
        },
        {
            "selector": 'node[status="archived"]',
            "style": {"opacity": 0.6},
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "line-color": BORDER,
                "target-arrow-color": ACCENT_DIM,
                "width": 1.5,
                "arrow-scale": 1.0,
            },
        },
    ]

    return html.Div(
        [
            html.Div(
                "VIZ20 — DVC Data Version Tree",
                style={
                    "color": ACCENT_GRN,
                    "fontSize": "12px",
                    "marginBottom": "6px",
                    "fontFamily": FONT,
                },
            ),
            html.Div(
                [
                    html.Span(
                        "◆ Raw  ",
                        style={"color": ACCENT_GRN, "fontFamily": FONT, "fontSize": "10px"},
                    ),
                    html.Span(
                        "⬡ Features  ",
                        style={"color": INFO, "fontFamily": FONT, "fontSize": "10px"},
                    ),
                    html.Span(
                        "▭ Processed  ",
                        style={"color": WARNING, "fontFamily": FONT, "fontSize": "10px"},
                    ),
                    html.Span(
                        "■ Split  ",
                        style={"color": TEXT_DIM, "fontFamily": FONT, "fontSize": "10px"},
                    ),
                    html.Span(
                        "(bright border = current)",
                        style={"color": TEXT_DIM, "fontFamily": FONT, "fontSize": "9px"},
                    ),
                ],
                style={"marginBottom": "6px"},
            ),
            cyto.Cytoscape(
                id="viz20-cyto",
                elements=elements,
                stylesheet=stylesheet,
                layout={
                    "name": "breadthfirst",
                    "directed": True,
                    "padding": 20,
                    "spacingFactor": 1.5,
                },
                style={
                    "width": "100%",
                    "height": "400px",
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
# Layout
# ---------------------------------------------------------------------------
layout = html.Div(
    style={"backgroundColor": BG_PRIMARY, "padding": "12px", "minHeight": "100vh"},
    children=[
        html.Div(
            "◈ Pipeline Monitor",
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
                html.Div(build_viz18(), style={"flex": "1.4", "minWidth": "500px"}),
                html.Div(build_viz19(), style={"flex": "1", "minWidth": "380px"}),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
        build_viz20(),
    ],
)
