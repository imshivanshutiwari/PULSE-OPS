"""
Drift Lab callbacks – interval-driven report reload and distribution update.
"""

import datetime
import numpy as np
from dash import Input, Output, html
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import (
    BG_CARD,
    ACCENT_GRN,
    INFO,
    WARNING,
    CRITICAL,
    SUCCESS,
    TEXT_DIM,
    FONT,
    PLOT_LAYOUT,
    CARD_STYLE,
)


def register_drift_callbacks(app):
    """Register all drift-lab callbacks on the given Dash app instance."""

    @app.callback(
        Output("drift-last-updated", "children"),
        Input("main-interval", "n_intervals"),
    )
    def update_drift_timestamp(n_intervals):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Last refreshed: {ts}"

    @app.callback(
        Output("drift-score-display", "children"),
        Input("main-interval", "n_intervals"),
        Input("drift-model-selector", "value"),
    )
    def update_drift_scores(n_intervals, selected_model):
        rng = np.random.default_rng(
            int(datetime.datetime.now().timestamp()) % 1000 + (n_intervals or 0)
        )
        model_baselines = {
            "adult/xgboost": {"data": 0.042, "perf": 0.012, "target": 0.031},
            "wine_quality/xgboost": {"data": 0.063, "perf": 0.019, "target": 0.044},
            "bike_sharing/lightgbm": {"data": 0.187, "perf": 0.063, "target": 0.121},
            "german_credit/neural_net": {"data": 0.031, "perf": 0.008, "target": 0.022},
        }
        if selected_model not in model_baselines:
            selected_model = "adult/xgboost"
        base = model_baselines[selected_model]
        scores = {
            "Data Drift": np.clip(base["data"] + rng.normal(0, 0.005), 0, 1),
            "Perf Drop": np.clip(base["perf"] + rng.normal(0, 0.003), 0, 1),
            "Target Drift": np.clip(base["target"] + rng.normal(0, 0.004), 0, 1),
        }
        items = []
        for metric, score in scores.items():
            color = SUCCESS if score < 0.05 else (WARNING if score < 0.15 else CRITICAL)
            items.append(
                html.Div(
                    [
                        html.Span(
                            f"{metric}: ",
                            style={"color": TEXT_DIM, "fontFamily": FONT, "fontSize": "11px"},
                        ),
                        html.Span(
                            f"{score:.4f}",
                            style={
                                "color": color,
                                "fontFamily": FONT,
                                "fontSize": "11px",
                                "fontWeight": "bold",
                            },
                        ),
                    ],
                    style={"marginBottom": "4px"},
                )
            )
        return items

    @app.callback(
        Output("drift-feature-chart", "figure"),
        Input("main-interval", "n_intervals"),
        Input("drift-feature-selector", "value"),
        Input("drift-model-selector", "value"),
    )
    def update_feature_distribution(n_intervals, selected_feature, selected_model):
        rng = np.random.default_rng(
            hash(str(selected_feature) + str(selected_model)) % 10000 + ((n_intervals or 0) % 100)
        )
        feature_params = {
            "age": (42, 12, 38, 13),
            "income": (52000, 18000, 54000, 21000),
            "credit_score": (680, 60, 645, 70),
            "debt_ratio": (28, 12, 32, 14),
            "num_accounts": (4.5, 1.8, 5.5, 2.1),
            "years_employed": (5.1, 3.2, 5.3, 3.4),
        }
        if selected_feature not in feature_params:
            selected_feature = "age"
        ref_mu, ref_sd, cur_mu, cur_sd = feature_params[selected_feature]
        ref_vals = rng.normal(ref_mu, ref_sd, 600)
        cur_vals = rng.normal(cur_mu, cur_sd, 600)

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=ref_vals,
                name="Reference",
                opacity=0.65,
                marker_color=INFO,
                nbinsx=35,
                histnorm="probability density",
            )
        )
        fig.add_trace(
            go.Histogram(
                x=cur_vals,
                name="Current",
                opacity=0.65,
                marker_color=WARNING,
                nbinsx=35,
                histnorm="probability density",
            )
        )
        layout = {**PLOT_LAYOUT}
        layout["title"] = dict(
            text=f"Distribution: {selected_feature} ({selected_model})",
            font=dict(color=ACCENT_GRN, size=11),
        )
        layout["barmode"] = "overlay"
        layout["height"] = 280
        layout["xaxis"] = {**PLOT_LAYOUT.get("xaxis", {}), "title": selected_feature}
        layout["yaxis"] = {**PLOT_LAYOUT.get("yaxis", {}), "title": "Density"}
        fig.update_layout(**layout)
        return fig

    @app.callback(
        Output("drift-alert-banner", "children"),
        Output("drift-alert-banner", "style"),
        Input("main-interval", "n_intervals"),
    )
    def update_drift_alert(n_intervals):
        rng = np.random.default_rng((n_intervals or 0) % 500)
        drift_score = 0.187 + rng.normal(0, 0.01)
        if drift_score > 0.20:
            msg = (
                f"⚠ HIGH DRIFT DETECTED  bike_sharing/lightgbm"
                f"  score={drift_score:.3f}  → retraining recommended"
            )
            style = {
                "backgroundColor": "#2a1000",
                "border": f"1px solid {WARNING}",
                "color": WARNING,
                "fontFamily": FONT,
                "fontSize": "11px",
                "padding": "8px 12px",
                "borderRadius": "3px",
                "marginBottom": "8px",
            }
        elif drift_score > 0.15:
            msg = (
                f"● DRIFT WARNING  bike_sharing/lightgbm"
                f"  score={drift_score:.3f}  → monitoring closely"
            )
            style = {
                "backgroundColor": "#1a1000",
                "border": f"1px solid {WARNING}",
                "color": WARNING,
                "fontFamily": FONT,
                "fontSize": "11px",
                "padding": "8px 12px",
                "borderRadius": "3px",
                "marginBottom": "8px",
            }
        else:
            msg = "✓ All models within drift thresholds"
            style = {
                "backgroundColor": CARD_STYLE.get("backgroundColor", BG_CARD),
                "border": f"1px solid {SUCCESS}",
                "color": SUCCESS,
                "fontFamily": FONT,
                "fontSize": "11px",
                "padding": "8px 12px",
                "borderRadius": "3px",
                "marginBottom": "8px",
            }
        return msg, style
