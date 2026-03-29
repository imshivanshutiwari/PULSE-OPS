"""
Registry Hub callbacks – model version click, stage filter.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html, no_update
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import (
    BG_CARD, BG_PANEL, BORDER, ACCENT_GRN, ACCENT_DIM, INFO, WARNING,
    CRITICAL, SUCCESS, TEXT, TEXT_DIM, FONT, PLOT_LAYOUT, CARD_STYLE
)

_ALL_VERSIONS = {
    'a-v1': {'model': 'adult/xgboost',            'version': 'v1.0', 'stage': 'archived',   'accuracy': 0.831, 'f1': 0.818, 'auc': 0.879, 'created': '2024-01-10'},
    'a-v2': {'model': 'adult/xgboost',            'version': 'v2.0', 'stage': 'archived',   'accuracy': 0.858, 'f1': 0.845, 'auc': 0.901, 'created': '2024-02-18'},
    'a-v3': {'model': 'adult/xgboost',            'version': 'v3.2', 'stage': 'production', 'accuracy': 0.873, 'f1': 0.861, 'auc': 0.921, 'created': '2024-05-12'},
    'a-v4': {'model': 'adult/xgboost',            'version': 'v3.3', 'stage': 'staging',    'accuracy': 0.881, 'f1': 0.872, 'auc': 0.934, 'created': '2024-05-15'},
    'w-v1': {'model': 'wine_quality/xgboost',     'version': 'v1.0', 'stage': 'archived',   'accuracy': 0.771, 'f1': 0.758, 'auc': 0.822, 'created': '2024-01-15'},
    'w-v2': {'model': 'wine_quality/xgboost',     'version': 'v2.1', 'stage': 'production', 'accuracy': 0.791, 'f1': 0.778, 'auc': 0.843, 'created': '2024-05-10'},
    'w-v3': {'model': 'wine_quality/xgboost',     'version': 'v2.2', 'stage': 'staging',    'accuracy': 0.798, 'f1': 0.784, 'auc': 0.851, 'created': '2024-05-16'},
    'b-v1': {'model': 'bike_sharing/lightgbm',    'version': 'v3.8', 'stage': 'archived',   'accuracy': 0.888, 'f1': 0.881, 'auc': 0.934, 'created': '2024-01-20'},
    'b-v2': {'model': 'bike_sharing/lightgbm',    'version': 'v3.9', 'stage': 'archived',   'accuracy': 0.901, 'f1': 0.895, 'auc': 0.947, 'created': '2024-03-10'},
    'b-v3': {'model': 'bike_sharing/lightgbm',    'version': 'v4.0', 'stage': 'production', 'accuracy': 0.912, 'f1': 0.905, 'auc': 0.958, 'created': '2024-05-08'},
    'b-v4': {'model': 'bike_sharing/lightgbm',    'version': 'v4.1', 'stage': 'staging',    'accuracy': 0.918, 'f1': 0.911, 'auc': 0.963, 'created': '2024-05-17'},
    'g-v1': {'model': 'german_credit/neural_net', 'version': 'v1.3', 'stage': 'archived',   'accuracy': 0.798, 'f1': 0.782, 'auc': 0.841, 'created': '2024-02-05'},
    'g-v2': {'model': 'german_credit/neural_net', 'version': 'v1.4', 'stage': 'archived',   'accuracy': 0.809, 'f1': 0.794, 'auc': 0.856, 'created': '2024-04-01'},
    'g-v3': {'model': 'german_credit/neural_net', 'version': 'v1.5', 'stage': 'production', 'accuracy': 0.816, 'f1': 0.803, 'auc': 0.867, 'created': '2024-05-14'},
}

_STAGE_COLORS = {
    'production': ACCENT_GRN,
    'staging':    INFO,
    'archived':   TEXT_DIM,
}


def register_registry_callbacks(app):
    """Register all registry-hub callbacks on the given Dash app instance."""

    @app.callback(
        Output('registry-node-detail', 'children'),
        Input('viz14-cyto', 'tapNodeData'),
        prevent_initial_call=True,
    )
    def display_node_detail(node_data):
        if not node_data:
            return no_update
        node_id = node_data.get('id', '')
        info    = _ALL_VERSIONS.get(node_id)
        if not info:
            return no_update

        stage_color = _STAGE_COLORS.get(info['stage'], TEXT_DIM)
        rows = [
            ('Model',    info['model']),
            ('Version',  info['version']),
            ('Stage',    info['stage']),
            ('Accuracy', f"{info['accuracy']:.4f}"),
            ('F1 Score', f"{info['f1']:.4f}"),
            ('AUC-ROC',  f"{info['auc']:.4f}"),
            ('Created',  info['created']),
        ]
        row_els = []
        for label, value in rows:
            v_color = stage_color if label == 'Stage' else TEXT
            row_els.append(html.Div([
                html.Span(f'{label}: ', style={'color': TEXT_DIM, 'fontFamily': FONT, 'fontSize': '10px', 'width': '70px', 'display': 'inline-block'}),
                html.Span(value, style={'color': v_color, 'fontFamily': FONT, 'fontSize': '10px'}),
            ], style={'marginBottom': '3px'}))
        return html.Div([
            html.Div(f'{info["model"]} — {info["version"]}',
                     style={'color': ACCENT_GRN, 'fontFamily': FONT, 'fontSize': '11px',
                            'fontWeight': 'bold', 'marginBottom': '8px',
                            'borderBottom': f'1px solid {BORDER}', 'paddingBottom': '4px'}),
            *row_els,
        ], style={
            'backgroundColor': BG_PANEL, 'border': f'1px solid {stage_color}',
            'borderRadius': '3px', 'padding': '10px',
        })

    @app.callback(
        Output('registry-filtered-table', 'figure'),
        Input('registry-stage-filter', 'value'),
    )
    def filter_registry_table(selected_stages):
        if not selected_stages:
            selected_stages = ['production', 'staging', 'archived']
        rows = [v for v in _ALL_VERSIONS.values() if v['stage'] in selected_stages]
        if not rows:
            rows = list(_ALL_VERSIONS.values())

        fill_colors_stage = [_STAGE_COLORS.get(r['stage'], TEXT_DIM) for r in rows]
        cell_fill_stage   = [BG_CARD if r['stage'] != 'production' else ACCENT_DIM for r in rows]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Model</b>', '<b>Version</b>', '<b>Stage</b>',
                        '<b>Accuracy</b>', '<b>F1</b>', '<b>AUC</b>', '<b>Created</b>'],
                fill_color=ACCENT_DIM,
                font=dict(color=ACCENT_GRN, size=11, family=FONT),
                align='left',
                line_color=BORDER,
            ),
            cells=dict(
                values=[
                    [r['model']           for r in rows],
                    [r['version']         for r in rows],
                    [r['stage']           for r in rows],
                    [f"{r['accuracy']:.4f}" for r in rows],
                    [f"{r['f1']:.4f}"     for r in rows],
                    [f"{r['auc']:.4f}"    for r in rows],
                    [r['created']         for r in rows],
                ],
                fill_color=[
                    cell_fill_stage,
                    [BG_CARD] * len(rows),
                    cell_fill_stage,
                    [BG_CARD] * len(rows),
                    [BG_CARD] * len(rows),
                    [BG_CARD] * len(rows),
                    [BG_CARD] * len(rows),
                ],
                font=dict(
                    color=[
                        [ACCENT_GRN if r['stage'] == 'production' else TEXT for r in rows],
                        [TEXT] * len(rows),
                        fill_colors_stage,
                        [TEXT] * len(rows),
                        [TEXT] * len(rows),
                        [TEXT] * len(rows),
                        [TEXT_DIM] * len(rows),
                    ],
                    size=10,
                    family=FONT,
                ),
                align='left',
                line_color=BORDER,
            ),
        )])
        layout = {**PLOT_LAYOUT}
        stage_label = ', '.join(selected_stages) if selected_stages else 'all'
        layout['title'] = dict(
            text=f'Filtered Registry — Stage: {stage_label}',
            font=dict(color=ACCENT_GRN, size=12),
        )
        layout['height'] = 320
        layout.pop('xaxis', None)
        layout.pop('yaxis', None)
        fig.update_layout(**layout)
        return fig
