"""
PULSE-OPS — Autonomous MLOps Platform Dashboard
Main Dash application entry point.
"""

import os
import sys
import datetime
import numpy as np

import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# Ensure dashboard package is importable when run from any working directory
_DASH_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_DASH_DIR)
for _path in (_DASH_DIR, _ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from theme import (
    BG_PRIMARY, BG_PANEL, BG_CARD, BORDER, ACCENT_GRN, ACCENT_DIM,
    SUCCESS, WARNING, CRITICAL, INFO, TEXT, TEXT_DIM, FONT,
    HEADER_STYLE, TAB_STYLE, TAB_SELECTED_STYLE,
)

# Page modules
from pages import mlops_ops, drift_lab, registry_hub, pipeline_monitor, system_health

# Callback registrations
from callbacks.drift_callbacks    import register_drift_callbacks
from callbacks.registry_callbacks import register_registry_callbacks

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title='PULSE-OPS MLOps Platform',
    update_title=None,
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
)
server = app.server

# ---------------------------------------------------------------------------
# Inline CSS injected as a custom stylesheet string
# ---------------------------------------------------------------------------
_CSS = f"""
body {{
    background-color: {BG_PRIMARY};
    color: {TEXT};
    font-family: {FONT};
    margin: 0;
    padding: 0;
}}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG_PANEL}; }}
::-webkit-scrollbar-thumb {{ background: {ACCENT_DIM}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_GRN}; }}
.dash-tab {{ transition: all 0.2s ease; }}
.tab-content {{ background-color: {BG_PRIMARY}; }}
.Select-control {{ background-color: {BG_CARD} !important; color: {TEXT} !important; border-color: {BORDER} !important; font-family: {FONT}; }}
.Select-menu-outer {{ background-color: {BG_PANEL} !important; border-color: {BORDER} !important; }}
.Select-option {{ background-color: {BG_PANEL} !important; color: {TEXT} !important; font-family: {FONT}; }}
.Select-option.is-selected {{ background-color: {ACCENT_DIM} !important; color: {ACCENT_GRN} !important; }}
.Select-option.is-focused {{ background-color: {ACCENT_DIM} !important; }}
.Select-value-label {{ color: {TEXT} !important; }}
"""

# ---------------------------------------------------------------------------
# Header component
# ---------------------------------------------------------------------------
def _service_dot(name, color):
    return html.Span(
        [html.Span('● ', style={'color': color, 'fontSize': '10px'}),
         html.Span(name, style={'color': TEXT_DIM, 'fontSize': '9px', 'marginRight': '10px'})],
    )


_HEADER = html.Div(
    style=HEADER_STYLE,
    children=[
        html.Div([
            html.Span('◈ ', style={'color': ACCENT_GRN, 'fontSize': '20px'}),
            html.Span('PULSE', style={'color': ACCENT_GRN, 'fontSize': '18px', 'fontWeight': 'bold', 'letterSpacing': '3px'}),
            html.Span('-OPS', style={'color': SUCCESS, 'fontSize': '18px', 'fontWeight': 'bold', 'letterSpacing': '1px'}),
            html.Span('  [MLOPS PLATFORM v1.0]', style={'color': TEXT_DIM, 'fontSize': '11px', 'marginLeft': '8px'}),
        ]),
        html.Div(
            id='header-service-bar',
            children=[
                _service_dot('MLflow',   SUCCESS),
                _service_dot('Prefect',  SUCCESS),
                _service_dot('FeatStore', SUCCESS),
                _service_dot('ModelAPI', SUCCESS),
                _service_dot('Prometheus', WARNING),
                _service_dot('Grafana',  WARNING),
                _service_dot('Redis',    SUCCESS),
                _service_dot('DVC',      SUCCESS),
            ],
            style={'display': 'flex', 'alignItems': 'center'},
        ),
        html.Div(
            id='header-clock',
            style={'color': TEXT_DIM, 'fontSize': '11px', 'fontFamily': FONT},
            children=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ),
    ],
)

# ---------------------------------------------------------------------------
# Registry Hub interactive controls
# ---------------------------------------------------------------------------
_REGISTRY_CONTROLS = html.Div([
    html.Div([
        html.Span('Stage Filter: ', style={'color': TEXT_DIM, 'fontFamily': FONT, 'fontSize': '11px', 'marginRight': '8px'}),
        dcc.Checklist(
            id='registry-stage-filter',
            options=[
                {'label': html.Span('Production', style={'color': ACCENT_GRN, 'fontFamily': FONT, 'fontSize': '11px'}), 'value': 'production'},
                {'label': html.Span('Staging',    style={'color': INFO,       'fontFamily': FONT, 'fontSize': '11px'}), 'value': 'staging'},
                {'label': html.Span('Archived',   style={'color': TEXT_DIM,   'fontFamily': FONT, 'fontSize': '11px'}), 'value': 'archived'},
            ],
            value=['production', 'staging'],
            inline=True,
            inputStyle={'marginRight': '4px', 'marginLeft': '10px'},
            labelStyle={'marginRight': '4px'},
        ),
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
    html.Div(id='registry-node-detail', style={'minHeight': '40px'}),
    dcc.Graph(id='registry-filtered-table', config={'displayModeBar': False},
              style={'backgroundColor': BG_CARD, 'border': f'1px solid {BORDER}',
                     'borderRadius': '4px', 'marginTop': '8px'}),
], style={
    'backgroundColor': BG_CARD, 'border': f'1px solid {BORDER}',
    'borderRadius': '4px', 'padding': '12px', 'marginBottom': '12px',
})

# ---------------------------------------------------------------------------
# Drift Lab interactive controls
# ---------------------------------------------------------------------------
_DRIFT_CONTROLS = html.Div([
    html.Div(id='drift-alert-banner', children='', style={}),
    html.Div([
        html.Div([
            html.Span('Model: ', style={'color': TEXT_DIM, 'fontFamily': FONT, 'fontSize': '11px'}),
            dcc.Dropdown(
                id='drift-model-selector',
                options=[
                    {'label': 'adult/xgboost',            'value': 'adult/xgboost'},
                    {'label': 'wine_quality/xgboost',     'value': 'wine_quality/xgboost'},
                    {'label': 'bike_sharing/lightgbm',    'value': 'bike_sharing/lightgbm'},
                    {'label': 'german_credit/neural_net', 'value': 'german_credit/neural_net'},
                ],
                value='bike_sharing/lightgbm',
                clearable=False,
                style={
                    'backgroundColor': BG_CARD, 'color': TEXT,
                    'border': f'1px solid {BORDER}', 'fontFamily': FONT,
                    'fontSize': '11px', 'width': '240px',
                },
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'marginRight': '20px'}),
        html.Div([
            html.Span('Feature: ', style={'color': TEXT_DIM, 'fontFamily': FONT, 'fontSize': '11px'}),
            dcc.Dropdown(
                id='drift-feature-selector',
                options=[{'label': f, 'value': f} for f in
                         ['age', 'income', 'credit_score', 'debt_ratio', 'num_accounts', 'years_employed']],
                value='credit_score',
                clearable=False,
                style={
                    'backgroundColor': BG_CARD, 'color': TEXT,
                    'border': f'1px solid {BORDER}', 'fontFamily': FONT,
                    'fontSize': '11px', 'width': '180px',
                },
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'marginBottom': '8px'}),
    html.Div(id='drift-score-display', style={'marginBottom': '8px'}),
    dcc.Graph(id='drift-feature-chart', config={'displayModeBar': False}),
    html.Div(id='drift-last-updated',
             style={'color': TEXT_DIM, 'fontFamily': FONT, 'fontSize': '9px', 'marginTop': '4px'}),
], style={
    'backgroundColor': BG_CARD, 'border': f'1px solid {BORDER}',
    'borderRadius': '4px', 'padding': '12px', 'marginBottom': '12px',
})

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={'backgroundColor': BG_PRIMARY, 'minHeight': '100vh'},
    children=[
        # Inject CSS via assets or index_string; use a hidden Div as style comment
        html.Div(id='_css-placeholder', style={'display': 'none'}),

        # Live update interval (30 seconds)
        dcc.Interval(id='main-interval', interval=30_000, n_intervals=0),

        # Header
        _HEADER,

        # Tab navigation
        dcc.Tabs(
            id='main-tabs',
            value='tab-mlops',
            style={
                'backgroundColor': BG_PANEL,
                'borderBottom': f'1px solid {BORDER}',
                'fontFamily': FONT,
            },
            children=[
                dcc.Tab(
                    label='⬡ MLOps Ops Center',
                    value='tab-mlops',
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
                dcc.Tab(
                    label='≋ Drift Analysis Lab',
                    value='tab-drift',
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
                dcc.Tab(
                    label='◫ Registry Hub',
                    value='tab-registry',
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
                dcc.Tab(
                    label='⌬ Pipeline Monitor',
                    value='tab-pipeline',
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
                dcc.Tab(
                    label='◉ System Health',
                    value='tab-health',
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
            ],
        ),

        # Page content
        html.Div(id='tab-content', style={'backgroundColor': BG_PRIMARY}),
    ],
)

# ---------------------------------------------------------------------------
# Tab routing callback
# ---------------------------------------------------------------------------
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value'),
)
def render_tab(tab):
    if tab == 'tab-mlops':
        return mlops_ops.layout
    if tab == 'tab-drift':
        return html.Div([
            _DRIFT_CONTROLS,
            drift_lab.layout,
        ], style={'backgroundColor': BG_PRIMARY, 'padding': '0'})
    if tab == 'tab-registry':
        return html.Div([
            html.Div(_REGISTRY_CONTROLS, style={'padding': '12px 12px 0 12px'}),
            registry_hub.layout,
        ], style={'backgroundColor': BG_PRIMARY})
    if tab == 'tab-pipeline':
        return pipeline_monitor.layout
    if tab == 'tab-health':
        return system_health.layout
    return html.Div('Select a tab', style={'color': TEXT_DIM, 'fontFamily': FONT, 'padding': '20px'})


# ---------------------------------------------------------------------------
# Clock update callback
# ---------------------------------------------------------------------------
@app.callback(
    Output('header-clock', 'children'),
    Input('main-interval', 'n_intervals'),
)
def update_clock(n):
    return datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')


# ---------------------------------------------------------------------------
# Register feature-rich callbacks from callback modules
# ---------------------------------------------------------------------------
register_drift_callbacks(app)
register_registry_callbacks(app)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import webbrowser
    import threading

    def _open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open('http://localhost:8050')

    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(debug=False, port=8050, host='0.0.0.0')
