"""Dash web application for the smvis Model Visualizer."""
from __future__ import annotations
import hashlib
import os
import json
import logging
import time
import traceback

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_cytoscape as cyto

from smvis.smv_parser import parse_smv
from smvis.smv_model import SmvModel, get_domain, expr_to_str
from smvis.explicit_engine import explore, ExplicitResult
from smvis.bdd_engine import build_from_explicit, BddResult, get_bdd_structure, build_truth_table
from smvis.graph_builder import (
    build_elements, compute_concentric_positions,
    CYTO_STYLESHEET, BDD_STYLESHEET, BUCHI_STYLESHEET, get_state_detail,
    apply_trace_overlay, apply_repeatable_overlay,
    build_buchi_elements, apply_lasso_overlay,
)
from smvis.cycle_analysis import analyze_cycles, CycleAnalysisResult
from smvis.bdd_visualizer import get_bdd_summary, get_reduction_stats
from smvis.ltl_buchi import build_buchi_for_spec, negate_ltl, simplify_ltl, UnsupportedLTLPattern
from smvis.product_model import compose as compose_product
from smvis.accepting_cycles import find_accepting_cycles, project_trace
from smvis.nuxmv_runner import (
    run_batch_check, nuxmv_available, NuxmvSession, write_temp_model,
    _NUXMV_PATH,
)

log = logging.getLogger("smvis")

# Module-level nuXmv interactive session (one per server instance)
_nuxmv_session: NuxmvSession | None = None
_nuxmv_temp_path: str | None = None

# Result cache keyed by SHA-256 of SMV text to avoid recomputation
_compute_cache: dict[str, tuple[ExplicitResult, BddResult, CycleAnalysisResult]] = {}
_CACHE_MAX = 10


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _get_cached(text: str) -> tuple[ExplicitResult, BddResult, CycleAnalysisResult] | None:
    return _compute_cache.get(_cache_key(text))


def _put_cached(text: str, explicit: ExplicitResult, bdd: BddResult,
                cycle: CycleAnalysisResult):
    key = _cache_key(text)
    _compute_cache[key] = (explicit, bdd, cycle)
    if len(_compute_cache) > _CACHE_MAX:
        oldest = next(iter(_compute_cache))
        del _compute_cache[oldest]

# Callback log for debugging GUI interactions server-side
_callback_log: list[dict] = []
_CALLBACK_LOG_MAX = 200


def _log_callback(name: str, inputs: dict, error: str | None = None):
    entry = {"time": time.time(), "callback": name, "inputs": inputs}
    if error:
        entry["error"] = error
    _callback_log.append(entry)
    if len(_callback_log) > _CALLBACK_LOG_MAX:
        _callback_log.pop(0)


# Path to bundled .smv model files (smvis/examples/)
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "examples")

def _find_model_files():
    """Find all .smv files in the models directory."""
    files = []
    if os.path.isdir(_MODELS_DIR):
        for f in sorted(os.listdir(_MODELS_DIR)):
            if f.endswith(".smv"):
                files.append(f)
    return files


def _load_model_text(filename: str) -> str:
    """Load the text of a model file."""
    filepath = os.path.join(_MODELS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return f.read()
    return ""


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"),
    )

    model_files = _find_model_files()
    default_model = model_files[0] if model_files else ""
    default_text = _load_model_text(default_model) if default_model else ""

    app.layout = html.Div([
        # ---- Header ----
        html.Div([
            html.H2("smvis", style={"margin": "0", "flex": "1"}),
            html.Div([
                html.Label("Model:", style={"marginRight": "8px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id="model-selector",
                    options=[{"label": f, "value": f} for f in model_files],
                    value=default_model,
                    style={"width": "200px"},
                    clearable=False,
                ),
                html.Button("Debug Log", id="btn-debug", n_clicks=0, style={
                    "marginLeft": "16px", "fontSize": "11px", "padding": "4px 10px",
                    "backgroundColor": "#7f8c8d", "color": "#fff", "border": "none",
                    "borderRadius": "3px", "cursor": "pointer",
                }),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "padding": "10px 20px",
            "backgroundColor": "#2c3e50", "color": "#ecf0f1",
        }),

        # ---- Debug Log Panel (hidden by default) ----
        html.Div(id="debug-panel", style={
            "display": "none", "padding": "8px 20px",
            "backgroundColor": "#1e1e1e", "color": "#d4d4d4",
            "maxHeight": "200px", "overflowY": "auto",
            "fontSize": "11px", "fontFamily": "Consolas, monospace",
        }),

        # ---- Main Content (resizable panels) ----
        html.Div([
            # ---- Left Column: Editor + Controls ----
            html.Div([
                # Editor
                html.Div([
                    html.H4("SMV Editor", style={"marginTop": "0"}),
                    dcc.Textarea(
                        id="smv-editor",
                        value=default_text,
                        style={
                            "width": "100%", "height": "350px",
                            "fontFamily": "Consolas, monospace",
                            "fontSize": "12px", "resize": "vertical",
                            "backgroundColor": "#1e1e1e", "color": "#d4d4d4",
                            "border": "1px solid #555", "padding": "8px",
                        },
                    ),
                    html.Div([
                        html.Button("Parse", id="btn-parse", n_clicks=0,
                                    style=_btn_style("#3498db")),
                        html.Button("Compute All", id="btn-compute", n_clicks=0,
                                    style=_btn_style("#27ae60")),
                        dcc.Download(id="download-smv"),
                        html.Button("Save", id="btn-save", n_clicks=0,
                                    style=_btn_style("#8e44ad")),
                    ], style={"display": "flex", "gap": "8px", "marginTop": "8px"}),
                    html.Div(id="parse-status", style={
                        "marginTop": "8px", "padding": "6px",
                        "fontSize": "12px", "maxHeight": "80px", "overflow": "auto",
                    }),
                ], style={"marginBottom": "16px"}),

                # Controls
                html.Div([
                    html.H4("Statistics", style={"marginTop": "0"}),
                    html.Div(id="stats-panel", style={"fontSize": "12px"}),
                ]),

                # Verification
                html.Div([
                    html.H4("Verification", style={"marginTop": "12px"}),
                    html.Div([
                        html.Button("Check All Specs", id="btn-check-specs",
                                    n_clicks=0, style=_btn_style("#e67e22")),
                        html.Button("Clear Trace", id="btn-clear-trace",
                                    n_clicks=0, style={
                                        **_btn_style("#95a5a6"),
                                        "fontSize": "11px", "padding": "4px 10px",
                                    }),
                    ], style={"display": "flex", "gap": "8px"}),
                    dcc.Dropdown(
                        id="trace-selector", options=[], value=None,
                        placeholder="Select trace to visualize...",
                        style={"fontSize": "11px", "marginTop": "6px"},
                        clearable=True,
                    ),
                    html.Div(id="spec-results", style={
                        "fontSize": "11px", "marginTop": "8px",
                        "maxHeight": "300px", "overflowY": "auto",
                    }),
                ]),

                # LTL Buchi Analysis
                html.Div([
                    html.H4("LTL Buchi Analysis", style={"marginTop": "12px"}),
                    dcc.Dropdown(
                        id="ltl-spec-selector",
                        options=[], value=None,
                        placeholder="Select LTLSPEC...",
                        style={"fontSize": "11px"},
                        clearable=True,
                    ),
                    html.Div([
                        html.Button("Compose & Analyze", id="btn-compose-buchi",
                                    n_clicks=0, style=_btn_style("#c0392b")),
                        html.Button("Clear", id="btn-clear-buchi",
                                    n_clicks=0, style={
                                        **_btn_style("#95a5a6"),
                                        "fontSize": "11px", "padding": "4px 10px",
                                    }),
                    ], style={"display": "flex", "gap": "8px", "marginTop": "6px"}),
                    html.Div(id="buchi-info", style={
                        "fontSize": "11px", "marginTop": "8px",
                        "maxHeight": "350px", "overflowY": "auto",
                    }),
                ]),
            ], className="left-panel"),

            # ---- Resize Handle ----
            html.Div(className="resize-handle"),

            # ---- Right Column: Visualizations ----
            html.Div([
                # Model Summary
                html.Div([
                    html.H4("Model Summary", style={"marginTop": "0"}),
                    html.Div(id="model-summary", style={
                        "fontSize": "12px", "maxHeight": "180px", "overflowY": "auto",
                    }),
                ], style={"marginBottom": "12px"}),

                # State Graph
                html.Div([
                    html.Div([
                        html.H4("Transition System", style={"margin": "0", "flex": "1"}),
                        html.Div([
                            html.Button("Export PNG", id="btn-export-graph",
                                        n_clicks=0, style={
                                            "fontSize": "10px", "padding": "3px 8px",
                                            "backgroundColor": "#95a5a6", "color": "#fff",
                                            "border": "none", "borderRadius": "3px",
                                            "cursor": "pointer", "marginRight": "8px",
                                        }),
                            dcc.Checklist(
                                id="graph-options",
                                options=[
                                    {"label": " Reachable only", "value": "reachable_only"},
                                    {"label": " Repeatable states", "value": "show_repeatable"},
                                ],
                                value=["reachable_only"],
                                inline=True,
                                style={"fontSize": "12px"},
                            ),
                            dcc.Input(
                                id="state-filter", type="text",
                                placeholder="Filter: e.g., mode=on",
                                style={"width": "160px", "fontSize": "12px",
                                       "marginLeft": "8px"},
                            ),
                            dcc.Dropdown(
                                id="layout-selector",
                                options=[
                                    {"label": "Force-directed", "value": "cose"},
                                    {"label": "Breadthfirst", "value": "breadthfirst"},
                                    {"label": "Grid", "value": "grid"},
                                    {"label": "Circle", "value": "circle"},
                                    {"label": "Concentric", "value": "concentric"},
                                ],
                                value="cose",
                                style={"width": "140px", "fontSize": "12px",
                                       "marginLeft": "8px"},
                                clearable=False,
                            ),
                            dcc.Input(
                                id="max-nodes", type="number",
                                value=500, min=10, max=10000,
                                style={"width": "70px", "fontSize": "12px",
                                       "marginLeft": "8px"},
                            ),
                            html.Label("max nodes", style={"fontSize": "11px",
                                                            "marginLeft": "4px"}),
                        ], style={"display": "flex", "alignItems": "center",
                                  "flexWrap": "wrap", "gap": "4px"}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "alignItems": "center", "marginBottom": "4px"}),
                    # Repeatable states controls (hidden until checkbox checked)
                    html.Div(id="repeatable-controls", children=[
                        html.Label("Mode:", style={"fontSize": "11px", "marginRight": "4px"}),
                        dcc.Dropdown(
                            id="repeatable-mode",
                            options=[
                                {"label": "R* (all repeatable)", "value": "r_star"},
                                {"label": "R(n) layers", "value": "r_n"},
                                {"label": "SCC coloring", "value": "scc"},
                            ],
                            value="r_star",
                            clearable=False,
                            style={"width": "160px", "fontSize": "11px"},
                        ),
                        html.Label("Step:", style={
                            "fontSize": "11px", "marginLeft": "12px", "marginRight": "4px",
                        }),
                        html.Div(
                            dcc.Slider(
                                id="repeatable-step", min=1, max=10, step=1, value=1,
                                marks={1: "1", 10: "10"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            style={"flex": "1", "minWidth": "150px"},
                        ),
                        html.Span(id="repeatable-step-label", children="",
                                  style={"fontSize": "11px", "marginLeft": "8px",
                                         "whiteSpace": "nowrap"}),
                    ], style={"display": "none", "alignItems": "center", "gap": "4px",
                              "padding": "4px 0", "flexWrap": "wrap"}),
                    cyto.Cytoscape(
                        id="state-graph",
                        clearOnUnhover=True,
                        layout={"name": "cose", "animate": False,
                                "nodeRepulsion": 8000, "idealEdgeLength": 50},
                        style={"width": "100%", "height": "380px",
                               "border": "1px solid #ddd"},
                        stylesheet=CYTO_STYLESHEET,
                        elements=[],
                    ),
                    html.Div(id="state-detail", style={
                        "fontSize": "11px", "padding": "4px",
                        "backgroundColor": "#f8f9fa", "marginTop": "4px",
                        "minHeight": "20px",
                    }),
                    # Hover tooltip (positioned absolutely via clientside callback)
                    html.Div(id="hover-tooltip", style={
                        "display": "none",
                        "position": "fixed",
                        "backgroundColor": "rgba(44, 62, 80, 0.95)",
                        "color": "#ecf0f1",
                        "padding": "10px 14px",
                        "borderRadius": "6px",
                        "fontSize": "14px",
                        "fontFamily": "Consolas, monospace",
                        "lineHeight": "1.6",
                        "pointerEvents": "none",
                        "zIndex": "9999",
                        "boxShadow": "0 4px 12px rgba(0,0,0,0.3)",
                        "maxWidth": "400px",
                        "whiteSpace": "pre-line",
                    }),
                ], style={"marginBottom": "12px", "position": "relative"}),

                # Buchi Automaton Graph (hidden until analysis triggered)
                html.Div(id="buchi-panel", children=[
                    html.H4("Buchi Automaton for \u00acφ", style={"marginTop": "0"}),
                    cyto.Cytoscape(
                        id="buchi-graph",
                        layout={"name": "preset", "animate": False},
                        style={"width": "100%", "height": "180px",
                               "border": "1px solid #ddd"},
                        stylesheet=BUCHI_STYLESHEET,
                        elements=[],
                    ),
                ], style={"display": "none", "marginBottom": "12px"}),

                # BDD Section (full-width, stacked layout)
                html.Div([
                    html.Div([
                        html.H4("BDD Visualization", style={"margin": "0", "flex": "1"}),
                        html.Button("Export PNG", id="btn-export-bdd",
                                    n_clicks=0, style={
                                        "fontSize": "10px", "padding": "3px 8px",
                                        "backgroundColor": "#95a5a6", "color": "#fff",
                                        "border": "none", "borderRadius": "3px",
                                        "cursor": "pointer", "marginRight": "8px",
                                    }),
                        dcc.Dropdown(
                            id="bdd-selector",
                            options=[
                                {"label": "Initial States", "value": "init"},
                                {"label": "Reachable States", "value": "reached"},
                                {"label": "Transition Relation", "value": "trans"},
                                {"label": "Domain Constraint", "value": "domain"},
                            ],
                            value="reached",
                            style={"width": "200px", "fontSize": "12px"},
                            clearable=False,
                        ),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "alignItems": "center", "marginBottom": "4px"}),
                    cyto.Cytoscape(
                        id="bdd-graph",
                        layout={"name": "preset", "animate": False},
                        style={"width": "100%", "height": "350px",
                               "border": "1px solid #ddd"},
                        stylesheet=BDD_STYLESHEET,
                        elements=[],
                    ),
                    html.Div(id="bdd-info", style={
                        "fontSize": "12px", "marginTop": "8px",
                        "maxHeight": "250px", "overflowY": "auto",
                    }),
                ]),
            ], className="right-panel"),
        ], className="main-container"),

        # ---- nuXmv Terminal Panel (bottom, collapsible) ----
        html.Div([
            # Terminal header bar (always visible)
            html.Div([
                html.Button("\u25bc Terminal", id="btn-toggle-terminal",
                            n_clicks=0, style={
                                "background": "none", "border": "none",
                                "color": "#ecf0f1", "fontWeight": "bold",
                                "fontSize": "13px", "cursor": "pointer",
                                "padding": "0", "flex": "1", "textAlign": "left",
                            }),
                html.Div([
                    html.Button("go", id="btn-cmd-go", n_clicks=0,
                                style=_term_btn_style()),
                    html.Button("check_ctlspec", id="btn-cmd-ctl", n_clicks=0,
                                style=_term_btn_style()),
                    html.Button("check_ltlspec", id="btn-cmd-ltl", n_clicks=0,
                                style=_term_btn_style()),
                    html.Button("check_invar", id="btn-cmd-invar", n_clicks=0,
                                style=_term_btn_style()),
                    html.Button("show_traces", id="btn-cmd-traces", n_clicks=0,
                                style=_term_btn_style()),
                ], id="terminal-quick-cmds",
                   style={"display": "flex", "gap": "4px", "marginRight": "8px"}),
                html.Button("Start", id="btn-terminal-start", n_clicks=0,
                            style={**_btn_style("#27ae60"), "fontSize": "11px",
                                   "padding": "3px 10px"}),
                html.Button("Stop", id="btn-terminal-stop", n_clicks=0,
                            style={**_btn_style("#e74c3c"), "fontSize": "11px",
                                   "padding": "3px 10px", "marginLeft": "4px"}),
            ], style={"display": "flex", "alignItems": "center",
                      "padding": "6px 12px", "backgroundColor": "#2c3e50"}),
            # Collapsible terminal body (hidden by default)
            html.Div(id="terminal-body", children=[
                html.Pre(id="terminal-output", children="",
                         className="terminal-output"),
                html.Div([
                    html.Span("nuXmv > ", style={"color": "#3498db",
                                                   "flexShrink": "0"}),
                    dcc.Input(id="terminal-input", type="text",
                              placeholder="Type command and press Enter...",
                              debounce=True,
                              style={"flex": "1", "backgroundColor": "#2d2d2d",
                                     "color": "#d4d4d4", "border": "1px solid #555",
                                     "fontFamily": "Consolas, monospace",
                                     "fontSize": "12px", "padding": "4px 8px"}),
                    html.Button("Send", id="btn-terminal-send", n_clicks=0,
                                style={**_btn_style("#3498db"), "fontSize": "11px",
                                       "padding": "3px 10px", "marginLeft": "4px"}),
                ], className="terminal-input"),
            ], style={"display": "none"}),
            dcc.Interval(id="terminal-poll", interval=500, disabled=True),
        ], className="terminal-panel", id="terminal-panel"),

        # ---- Hidden Stores ----
        dcc.Store(id="parsed-model-store", data=None),
        dcc.Store(id="explicit-result-store", data=None),
        dcc.Store(id="bdd-result-store", data=None),
        dcc.Store(id="traces-store", data=None),
        dcc.Store(id="active-trace-store", data=None),
        dcc.Store(id="cycle-result-store", data=None),
        dcc.Store(id="ltl-specs-store", data=None),
    ], style={"fontFamily": "Segoe UI, Arial, sans-serif"})

    # ================================================================
    # CALLBACKS
    # ================================================================

    @app.callback(
        Output("smv-editor", "value"),
        Input("model-selector", "value"),
    )
    def load_model(filename):
        if not filename:
            return no_update
        return _load_model_text(filename)

    @app.callback(
        Output("download-smv", "data"),
        Input("btn-save", "n_clicks"),
        State("smv-editor", "value"),
        State("model-selector", "value"),
        prevent_initial_call=True,
    )
    def save_model(n, text, filename):
        if not n:
            return no_update
        name = filename or "model.smv"
        # Also save to disk
        filepath = os.path.join(_MODELS_DIR, name)
        try:
            with open(filepath, "w") as f:
                f.write(text)
        except Exception as e:
            log.exception("Error saving model to %s", filepath)
        return dcc.send_string(text, name)

    @app.callback(
        Output("parsed-model-store", "data"),
        Output("parse-status", "children"),
        Output("parse-status", "style"),
        Output("model-summary", "children"),
        Input("btn-parse", "n_clicks"),
        Input("btn-compute", "n_clicks"),
        State("smv-editor", "value"),
        prevent_initial_call=True,
    )
    def parse_model(n_parse, n_compute, text):
        if not text:
            return None, "No text to parse.", _status_style(False), ""
        try:
            model = parse_smv(text)
            summary = _build_summary(model)
            # Serialize model info for other callbacks
            model_data = _serialize_model(model)
            return model_data, "Parsed successfully.", _status_style(True), summary
        except Exception as e:
            err = str(e)
            return None, f"Parse error: {err}", _status_style(False), ""

    @app.callback(
        Output("explicit-result-store", "data"),
        Output("bdd-result-store", "data"),
        Output("stats-panel", "children"),
        Output("state-graph", "elements"),
        Output("bdd-graph", "elements"),
        Output("bdd-graph", "layout"),
        Output("bdd-info", "children"),
        Output("bdd-selector", "options"),
        Output("cycle-result-store", "data"),
        Output("repeatable-step", "max"),
        Output("repeatable-step", "marks"),
        Input("btn-compute", "n_clicks"),
        State("smv-editor", "value"),
        State("graph-options", "value"),
        State("state-filter", "value"),
        State("layout-selector", "value"),
        State("max-nodes", "value"),
        State("bdd-selector", "value"),
        prevent_initial_call=True,
    )
    def compute_all(n, text, graph_opts, filter_expr, layout, max_nodes, bdd_sel):
        _log_callback("compute_all", {"graph_opts": graph_opts, "layout": layout,
                                       "max_nodes": max_nodes, "bdd_sel": bdd_sel})
        if not n or not text:
            return (no_update,) * 11
        try:
            model = parse_smv(text)
            # Explicit exploration
            explicit_result = explore(model)
            # BDD computation
            bdd_result = build_from_explicit(
                model,
                explicit_result.initial_states,
                explicit_result.transitions,
                explicit_result.var_names,
                explicit_result.state_to_dict,
            )
            # Cycle analysis
            cycle_result = analyze_cycles(explicit_result)

            # Build visualizations
            reachable_only = "reachable_only" in (graph_opts or [])
            show_repeatable = "show_repeatable" in (graph_opts or [])
            graph_elements = build_elements(
                explicit_result,
                reachable_only=reachable_only,
                max_nodes=max_nodes or 500,
                filter_expr=filter_expr or "",
            )
            if show_repeatable:
                graph_elements = apply_repeatable_overlay(
                    graph_elements, cycle_result, mode="r_star"
                )

            stats = _build_stats(explicit_result, bdd_result, cycle_result)
            bdd_elements, bdd_info = _build_bdd_view(bdd_result, bdd_sel or "reached")
            bdd_layout = {"name": "preset", "animate": False}

            # Build dynamic BDD selector options including iteration views
            bdd_options = [
                {"label": "Initial States", "value": "init"},
                {"label": "Reachable States", "value": "reached"},
                {"label": "Transition Relation", "value": "trans"},
                {"label": "Domain Constraint", "value": "domain"},
            ]
            for i in range(len(bdd_result.iteration_bdds)):
                bdd_options.append({
                    "label": f"Iteration {i}" if i > 0 else "Iteration 0 (Init)",
                    "value": f"iter_{i}",
                })

            # Cache results for other callbacks
            _put_cached(text, explicit_result, bdd_result, cycle_result)

            # Serialize for store (lightweight)
            explicit_data = _serialize_explicit(explicit_result)
            bdd_data = {"total_reachable": bdd_result.total_reachable}

            # Cycle analysis store data
            conv = cycle_result.convergence_step if cycle_result.convergence_step > 0 else 1
            cycle_data = {"convergence_step": conv}
            slider_max = max(conv, 1)
            slider_marks = _build_slider_marks(slider_max)

            return (explicit_data, bdd_data, stats, graph_elements,
                    bdd_elements, bdd_layout, bdd_info, bdd_options,
                    cycle_data, slider_max, slider_marks)
        except Exception as e:
            _log_callback("compute_all", {"bdd_sel": bdd_sel}, error=str(e))
            log.exception("Error in compute_all callback")
            err_msg = html.Div([
                html.Span(f"Error: {e}", style={"color": "#e74c3c"}),
                html.Pre(traceback.format_exc(), style={"fontSize": "10px"}),
            ])
            return (None, None, err_msg, [], [], no_update, str(e), no_update,
                    None, no_update, no_update)

    @app.callback(
        Output("state-graph", "elements", allow_duplicate=True),
        Output("state-graph", "layout"),
        Input("graph-options", "value"),
        Input("state-filter", "value"),
        Input("layout-selector", "value"),
        Input("max-nodes", "value"),
        Input("repeatable-mode", "value"),
        Input("repeatable-step", "value"),
        State("explicit-result-store", "data"),
        State("smv-editor", "value"),
        prevent_initial_call=True,
    )
    def update_graph(graph_opts, filter_expr, layout, max_nodes,
                     rep_mode, rep_step, explicit_data, text):
        _log_callback("update_graph", {"graph_opts": graph_opts, "layout": layout,
                                        "max_nodes": max_nodes, "filter": filter_expr,
                                        "rep_mode": rep_mode, "rep_step": rep_step})
        if not explicit_data or not text:
            return no_update, no_update
        try:
            cached = _get_cached(text)
            if cached:
                explicit_result, _, cycle_result = cached
            else:
                model = parse_smv(text)
                explicit_result = explore(model)
                cycle_result = analyze_cycles(explicit_result)
            reachable_only = "reachable_only" in (graph_opts or [])
            show_repeatable = "show_repeatable" in (graph_opts or [])
            elements = build_elements(
                explicit_result,
                reachable_only=reachable_only,
                max_nodes=max_nodes or 500,
                filter_expr=filter_expr or "",
            )
            if show_repeatable:
                elements = apply_repeatable_overlay(
                    elements, cycle_result,
                    mode=rep_mode or "r_star",
                    step_n=rep_step or 1,
                )
            layout_dict = {"name": layout or "cose", "animate": False}
            if layout == "cose":
                layout_dict["nodeRepulsion"] = 8000
                layout_dict["idealEdgeLength"] = 50
            elif layout == "breadthfirst":
                layout_dict["directed"] = True
            elif layout == "concentric":
                elements = compute_concentric_positions(elements)
                layout_dict = {"name": "preset", "animate": False}
            return elements, layout_dict
        except Exception as e:
            _log_callback("update_graph", {"layout": layout}, error=str(e))
            log.exception("Error in update_graph callback")
            return [], {"name": "cose", "animate": False}

    @app.callback(
        Output("bdd-graph", "elements", allow_duplicate=True),
        Output("bdd-graph", "layout", allow_duplicate=True),
        Output("bdd-info", "children", allow_duplicate=True),
        Input("bdd-selector", "value"),
        State("bdd-result-store", "data"),
        State("smv-editor", "value"),
        State("explicit-result-store", "data"),
        prevent_initial_call=True,
    )
    def update_bdd_view(bdd_sel, bdd_data, text, explicit_data):
        _log_callback("update_bdd_view", {"bdd_sel": bdd_sel})
        if not bdd_data or not text:
            return no_update, no_update, no_update
        try:
            cached = _get_cached(text)
            if cached:
                _, bdd_result, _ = cached
            else:
                model = parse_smv(text)
                explicit_result = explore(model)
                bdd_result = build_from_explicit(
                    model,
                    explicit_result.initial_states,
                    explicit_result.transitions,
                    explicit_result.var_names,
                    explicit_result.state_to_dict,
                )
            elements, info = _build_bdd_view(bdd_result, bdd_sel or "reached")
            bdd_layout = {"name": "preset", "animate": False}
            return elements, bdd_layout, info
        except Exception as e:
            _log_callback("update_bdd_view", {"bdd_sel": bdd_sel}, error=str(e))
            log.exception("Error in update_bdd_view callback")
            return [], no_update, html.Span(f"Error: {e}", style={"color": "#e74c3c"})

    @app.callback(
        Output("state-detail", "children"),
        Input("state-graph", "tapNodeData"),
        State("smv-editor", "value"),
    )
    def show_state_detail(node_data, text):
        _log_callback("show_state_detail", {"node_id": node_data.get("id") if node_data else None})
        if not node_data or not text:
            return ""
        try:
            cached = _get_cached(text)
            if cached:
                result, _, _ = cached
            else:
                model = parse_smv(text)
                result = explore(model)
            detail = get_state_detail(result, node_data.get("id", ""))
            if detail:
                sd = detail["state"]
                parts = [f"{k}={v}" for k, v in sd.items()]
                flags = []
                if detail["is_initial"]:
                    flags.append("INITIAL")
                if detail["is_reachable"]:
                    flags.append("REACHABLE")
                return html.Span([
                    html.B("State: "),
                    ", ".join(parts),
                    " | ",
                    " ".join(flags),
                    f" | Successors: {detail['successor_count']}",
                ])
            return f"Node: {node_data.get('label', '?')}"
        except Exception as e:
            _log_callback("show_state_detail", {}, error=str(e))
            log.exception("Error in show_state_detail callback")
            return html.Span(f"Error: {e}", style={"color": "#e74c3c", "fontSize": "11px"})

    # ---- Debug log panel toggle ----
    @app.callback(
        Output("debug-panel", "children"),
        Output("debug-panel", "style"),
        Input("btn-debug", "n_clicks"),
        State("debug-panel", "style"),
        prevent_initial_call=True,
    )
    def toggle_debug_panel(n_clicks, current_style):
        if not n_clicks:
            return no_update, no_update
        visible = current_style.get("display", "none") != "none"
        new_style = {**current_style, "display": "none" if visible else "block"}
        if visible:
            return no_update, new_style
        # Render the log entries
        from datetime import datetime
        lines = []
        for entry in reversed(_callback_log):
            ts = datetime.fromtimestamp(entry["time"]).strftime("%H:%M:%S")
            cb = entry["callback"]
            inputs_str = json.dumps(entry["inputs"], default=str)[:120]
            line = f"[{ts}] {cb}: {inputs_str}"
            if "error" in entry:
                line += f"  ERROR: {entry['error']}"
            lines.append(line)
        content = html.Pre("\n".join(lines) if lines else "(no callbacks logged yet)")
        return content, new_style

    # ---- Hover tooltip via clientside callback (runs in browser) ----
    # Uses clearOnUnhover=True on the Cytoscape component so that
    # mouseoverNodeData is cleared to None when the mouse leaves a node.
    app.clientside_callback(
        """
        function(hoverData, modelData) {
            // clearOnUnhover=True sets hoverData to null on mouseout
            if (!hoverData) {
                return ['', {display: 'none'}];
            }

            // Build content lines from variable names
            var lines = [];
            var varNames = [];
            if (modelData && modelData.variables) {
                varNames = Object.keys(modelData.variables);
            }
            for (var i = 0; i < varNames.length; i++) {
                var v = varNames[i];
                if (v in hoverData) {
                    lines.push(v + ' = ' + hoverData[v]);
                }
            }
            // Fallback: show all data keys except id/label/depth
            if (lines.length === 0) {
                var keys = Object.keys(hoverData);
                for (var j = 0; j < keys.length; j++) {
                    var k = keys[j];
                    if (k !== 'id' && k !== 'label' && k !== 'depth') {
                        lines.push(k + ' = ' + hoverData[k]);
                    }
                }
            }
            if (lines.length === 0) {
                lines.push(hoverData.label || '?');
            }

            // Position near cursor
            var x = (window._smvisMouseX || 300) + 15;
            var y = (window._smvisMouseY || 200) + 15;
            // Keep tooltip on screen
            if (x + 300 > window.innerWidth) { x = x - 330; }
            if (y + 200 > window.innerHeight) { y = y - 220; }

            var style = {
                display: 'block',
                position: 'fixed',
                left: x + 'px',
                top: y + 'px',
                backgroundColor: 'rgba(44, 62, 80, 0.95)',
                color: '#ecf0f1',
                padding: '10px 14px',
                borderRadius: '6px',
                fontSize: '14px',
                fontFamily: 'Consolas, monospace',
                lineHeight: '1.6',
                pointerEvents: 'none',
                zIndex: '9999',
                boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                maxWidth: '400px',
                whiteSpace: 'pre-line',
            };

            return [lines.join('\\n'), style];
        }
        """,
        Output("hover-tooltip", "children"),
        Output("hover-tooltip", "style"),
        Input("state-graph", "mouseoverNodeData"),
        State("parsed-model-store", "data"),
    )

    # ---- PNG Export callbacks (clientside) ----
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) { return window.dash_clientside.no_update; }
            return {type: 'png', action: 'download', options: {bg: '#ffffff', full: true}};
        }
        """,
        Output("state-graph", "generateImage"),
        Input("btn-export-graph", "n_clicks"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) { return window.dash_clientside.no_update; }
            return {type: 'png', action: 'download', options: {bg: '#ffffff', full: true}};
        }
        """,
        Output("bdd-graph", "generateImage"),
        Input("btn-export-bdd", "n_clicks"),
        prevent_initial_call=True,
    )

    # ---- Verification: Check All Specs ----
    @app.callback(
        Output("spec-results", "children"),
        Output("traces-store", "data"),
        Output("trace-selector", "options"),
        Output("trace-selector", "value"),
        Input("btn-check-specs", "n_clicks"),
        State("smv-editor", "value"),
        prevent_initial_call=True,
    )
    def check_all_specs(n, text):
        _log_callback("check_all_specs", {"n": n})
        if not n or not text:
            return no_update, no_update, no_update, no_update
        result = run_batch_check(text, _NUXMV_PATH)
        if result.error:
            return (
                html.Div(f"Error: {result.error}", style={"color": "#e74c3c"}),
                None, [], None,
            )
        rows = []
        trace_options = []
        traces_data = []
        for spec in result.specs:
            icon = "\u2713" if spec.passed else "\u2717"
            color = "#27ae60" if spec.passed else "#e74c3c"
            row = html.Div([
                html.Span(icon, style={"color": color, "fontWeight": "bold",
                                       "marginRight": "6px"}),
                html.Span(spec.spec_kind, style={
                    "color": "#7f8c8d", "marginRight": "6px",
                    "fontSize": "10px", "fontStyle": "italic",
                }),
                html.Span(spec.spec_text),
            ], style={"padding": "2px 0", "borderBottom": "1px solid #eee"})
            rows.append(row)
            if spec.trace:
                idx = len(traces_data)
                label_text = spec.spec_text
                if len(label_text) > 50:
                    label_text = label_text[:47] + "..."
                trace_options.append({
                    "label": f"[{spec.spec_kind}] {label_text}",
                    "value": idx,
                })
                traces_data.append({
                    "states": spec.trace.states,
                    "loop_start": spec.trace.loop_start,
                    "description": spec.trace.description,
                    "spec_text": spec.spec_text,
                    "spec_kind": spec.spec_kind,
                })
        n_passed = sum(1 for s in result.specs if s.passed)
        n_total = len(result.specs)
        summary_color = "#27ae60" if n_passed == n_total else "#e67e22"
        header = html.Div(
            f"{n_passed}/{n_total} specifications passed",
            style={"fontWeight": "bold", "marginBottom": "6px", "color": summary_color},
        )
        return html.Div([header] + rows), traces_data, trace_options, None

    # ---- Verification: Show Trace on Graph ----
    @app.callback(
        Output("state-graph", "elements", allow_duplicate=True),
        Input("trace-selector", "value"),
        State("traces-store", "data"),
        State("smv-editor", "value"),
        State("graph-options", "value"),
        State("state-filter", "value"),
        State("layout-selector", "value"),
        State("max-nodes", "value"),
        prevent_initial_call=True,
    )
    def show_trace_on_graph(trace_idx, traces_data, text, graph_opts,
                            filter_expr, layout, max_nodes):
        _log_callback("show_trace_on_graph", {"trace_idx": trace_idx})
        if not text:
            return no_update
        cached = _get_cached(text)
        if not cached:
            return no_update
        explicit_result, _, _ = cached
        reachable_only = "reachable_only" in (graph_opts or [])
        elements = build_elements(
            explicit_result,
            reachable_only=reachable_only,
            max_nodes=max_nodes or 500,
            filter_expr=filter_expr or "",
        )
        if layout == "concentric":
            elements = compute_concentric_positions(elements)
        if (trace_idx is not None and traces_data
                and 0 <= trace_idx < len(traces_data)):
            trace = traces_data[trace_idx]
            elements = apply_trace_overlay(
                elements,
                trace["states"],
                explicit_result.state_to_dict,
                explicit_result.var_names,
                loop_start=trace.get("loop_start"),
            )
        return elements

    # ---- Verification: Clear Trace ----
    @app.callback(
        Output("trace-selector", "value", allow_duplicate=True),
        Input("btn-clear-trace", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_trace(n):
        if not n:
            return no_update
        return None

    # ---- Repeatable controls visibility ----
    @app.callback(
        Output("repeatable-controls", "style"),
        Input("graph-options", "value"),
    )
    def toggle_repeatable_controls(graph_opts):
        show = "show_repeatable" in (graph_opts or [])
        return {"display": "flex" if show else "none",
                "alignItems": "center", "gap": "4px",
                "padding": "4px 0", "flexWrap": "wrap"}

    # ---- Repeatable step label ----
    @app.callback(
        Output("repeatable-step-label", "children"),
        Input("repeatable-step", "value"),
        State("cycle-result-store", "data"),
    )
    def update_step_label(step_n, cycle_data):
        if not cycle_data:
            return ""
        conv = cycle_data.get("convergence_step", 0)
        return f"R({step_n}) / R({conv})"

    # ---- LTL Buchi: Populate LTLSPEC dropdown ----
    @app.callback(
        Output("ltl-spec-selector", "options"),
        Output("ltl-specs-store", "data"),
        Input("parsed-model-store", "data"),
        State("smv-editor", "value"),
        prevent_initial_call=True,
    )
    def populate_ltl_specs(model_data, text):
        if not model_data or not text:
            return [], None
        try:
            model = parse_smv(text)
            options = []
            specs_data = []
            for i, spec in enumerate(model.specs):
                if spec.kind == "LTLSPEC":
                    label = expr_to_str(spec.expr)
                    if len(label) > 60:
                        label = label[:57] + "..."
                    options.append({"label": f"[{i}] {label}", "value": i})
                    specs_data.append({"index": i, "text": expr_to_str(spec.expr)})
            return options, specs_data
        except Exception:
            return [], None

    # ---- LTL Buchi: Compose & Analyze ----
    @app.callback(
        Output("buchi-info", "children"),
        Output("buchi-graph", "elements"),
        Output("buchi-panel", "style"),
        Output("state-graph", "elements", allow_duplicate=True),
        Input("btn-compose-buchi", "n_clicks"),
        State("ltl-spec-selector", "value"),
        State("smv-editor", "value"),
        State("graph-options", "value"),
        State("state-filter", "value"),
        State("max-nodes", "value"),
        prevent_initial_call=True,
    )
    def compose_and_analyze(n, spec_idx, text, graph_opts, filter_expr, max_nodes):
        _log_callback("compose_and_analyze", {"spec_idx": spec_idx})
        if not n or spec_idx is None or not text:
            return no_update, no_update, no_update, no_update
        try:
            return _run_buchi_analysis(
                spec_idx, text, graph_opts, filter_expr, max_nodes,
            )
        except Exception as e:
            _log_callback("compose_and_analyze", {"spec_idx": spec_idx}, error=str(e))
            log.exception("Error in compose_and_analyze")
            err = html.Div([
                html.Span(f"Error: {e}", style={"color": "#e74c3c"}),
                html.Pre(traceback.format_exc(),
                         style={"fontSize": "10px", "maxHeight": "150px",
                                "overflow": "auto"}),
            ])
            return err, no_update, no_update, no_update

    # ---- LTL Buchi: Clear ----
    @app.callback(
        Output("buchi-info", "children", allow_duplicate=True),
        Output("buchi-graph", "elements", allow_duplicate=True),
        Output("buchi-panel", "style", allow_duplicate=True),
        Input("btn-clear-buchi", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_buchi(n):
        if not n:
            return no_update, no_update, no_update
        return "", [], {"display": "none", "marginBottom": "12px"}

    # ---- Terminal: Toggle visibility ----
    @app.callback(
        Output("terminal-body", "style"),
        Output("btn-toggle-terminal", "children"),
        Output("terminal-quick-cmds", "style"),
        Input("btn-toggle-terminal", "n_clicks"),
        State("terminal-body", "style"),
        prevent_initial_call=True,
    )
    def toggle_terminal(n, current_style):
        if not n:
            return no_update, no_update, no_update
        visible = current_style.get("display", "none") != "none"
        if visible:
            return ({"display": "none"}, "\u25b6 Terminal",
                    {"display": "none", "gap": "4px", "marginRight": "8px"})
        return ({"display": "block"}, "\u25bc Terminal",
                {"display": "flex", "gap": "4px", "marginRight": "8px"})

    # ---- Terminal: Start Session ----
    @app.callback(
        Output("terminal-output", "children", allow_duplicate=True),
        Output("terminal-poll", "disabled"),
        Input("btn-terminal-start", "n_clicks"),
        State("smv-editor", "value"),
        prevent_initial_call=True,
    )
    def start_terminal(n, text):
        global _nuxmv_session, _nuxmv_temp_path
        _log_callback("start_terminal", {"n": n})
        if not n:
            return no_update, no_update
        # Stop existing session
        if _nuxmv_session:
            _nuxmv_session.stop()
        if _nuxmv_temp_path:
            try:
                os.unlink(_nuxmv_temp_path)
            except OSError:
                pass
            _nuxmv_temp_path = None
        _nuxmv_session = NuxmvSession(_NUXMV_PATH)
        temp_path = None
        if text and text.strip():
            temp_path = write_temp_model(text)
            _nuxmv_temp_path = temp_path
        if _nuxmv_session.start(temp_path):
            return "Session started. Waiting for nuXmv...\n", False
        else:
            return "Error: Could not start nuXmv session.\n", True

    # ---- Terminal: Stop Session ----
    @app.callback(
        Output("terminal-output", "children", allow_duplicate=True),
        Output("terminal-poll", "disabled", allow_duplicate=True),
        Input("btn-terminal-stop", "n_clicks"),
        State("terminal-output", "children"),
        prevent_initial_call=True,
    )
    def stop_terminal(n, current):
        global _nuxmv_session, _nuxmv_temp_path
        _log_callback("stop_terminal", {"n": n})
        if not n:
            return no_update, no_update
        if _nuxmv_session:
            _nuxmv_session.stop()
            _nuxmv_session = None
        if _nuxmv_temp_path:
            try:
                os.unlink(_nuxmv_temp_path)
            except OSError:
                pass
            _nuxmv_temp_path = None
        return (current or "") + "\n[Session stopped]\n", True

    # ---- Terminal: Send Command ----
    @app.callback(
        Output("terminal-input", "value"),
        Input("btn-terminal-send", "n_clicks"),
        Input("terminal-input", "n_submit"),
        State("terminal-input", "value"),
        prevent_initial_call=True,
    )
    def send_terminal_command(n_click, n_submit, cmd):
        _log_callback("send_terminal_command", {"cmd": cmd})
        if _nuxmv_session and cmd and cmd.strip():
            _nuxmv_session.send_command(cmd.strip())
        return ""

    # ---- Terminal: Poll Output ----
    @app.callback(
        Output("terminal-output", "children", allow_duplicate=True),
        Input("terminal-poll", "n_intervals"),
        State("terminal-output", "children"),
        prevent_initial_call=True,
    )
    def poll_terminal(n, current):
        if not _nuxmv_session:
            return no_update
        new = _nuxmv_session.get_new_output()
        if not new:
            return no_update
        return (current or "") + new

    # ---- Terminal: Quick Commands ----
    @app.callback(
        Output("terminal-output", "children", allow_duplicate=True),
        Input("btn-cmd-go", "n_clicks"),
        Input("btn-cmd-ctl", "n_clicks"),
        Input("btn-cmd-ltl", "n_clicks"),
        Input("btn-cmd-invar", "n_clicks"),
        Input("btn-cmd-traces", "n_clicks"),
        State("terminal-output", "children"),
        prevent_initial_call=True,
    )
    def quick_command(n1, n2, n3, n4, n5, current):
        commands = {
            "btn-cmd-go": "go",
            "btn-cmd-ctl": "check_ctlspec",
            "btn-cmd-ltl": "check_ltlspec",
            "btn-cmd-invar": "check_invar",
            "btn-cmd-traces": "show_traces",
        }
        triggered = dash.callback_context.triggered
        if not triggered:
            return no_update
        btn_id = triggered[0]["prop_id"].split(".")[0]
        cmd = commands.get(btn_id)
        if cmd and _nuxmv_session:
            _nuxmv_session.send_command(cmd)
        return no_update

    return app


# ================================================================
# HELPERS
# ================================================================

def _btn_style(color: str) -> dict:
    return {
        "backgroundColor": color, "color": "#fff", "border": "none",
        "padding": "6px 16px", "borderRadius": "4px", "cursor": "pointer",
        "fontSize": "13px",
    }


def _term_btn_style() -> dict:
    return {
        "backgroundColor": "#34495e", "color": "#d4d4d4", "border": "1px solid #555",
        "padding": "2px 8px", "borderRadius": "3px", "cursor": "pointer",
        "fontSize": "10px", "fontFamily": "Consolas, monospace",
    }


def _status_style(ok: bool) -> dict:
    return {
        "marginTop": "8px", "padding": "6px", "fontSize": "12px",
        "maxHeight": "80px", "overflow": "auto",
        "backgroundColor": "#d4edda" if ok else "#f8d7da",
        "color": "#155724" if ok else "#721c24",
        "borderRadius": "4px",
    }


def _build_summary(model: SmvModel) -> html.Div:
    """Build the model summary panel."""
    rows = []

    # Variables
    rows.append(html.H5("Variables", style={"margin": "4px 0"}))
    total = 1
    for name, vd in model.variables.items():
        domain = get_domain(vd)
        total *= len(domain)
        domain_str = str(domain) if len(domain) <= 8 else f"{len(domain)} values"
        rows.append(html.Div(f"{name}: {domain_str}", style={"paddingLeft": "12px"}))
    rows.append(html.Div(f"Total state space: {total}", style={
        "fontWeight": "bold", "marginTop": "4px"}))

    # Defines
    if model.defines:
        rows.append(html.H5("Defines", style={"margin": "4px 0"}))
        for name, expr in model.defines.items():
            rows.append(html.Div(f"{name} := {expr_to_str(expr)}",
                                 style={"paddingLeft": "12px"}))

    # Init conditions
    if model.inits:
        rows.append(html.H5("Initial Conditions", style={"margin": "4px 0"}))
        for name, expr in model.inits.items():
            rows.append(html.Div(f"init({name}) := {expr_to_str(expr)}",
                                 style={"paddingLeft": "12px"}))
        # Note unconstrained vars
        for name in model.variables:
            if name not in model.inits:
                rows.append(html.Div(f"init({name}) := * (unconstrained)",
                                     style={"paddingLeft": "12px", "color": "#888"}))

    # Specs count
    spec_counts = {}
    for sp in model.specs:
        spec_counts[sp.kind] = spec_counts.get(sp.kind, 0) + 1
    if spec_counts:
        rows.append(html.H5("Specifications", style={"margin": "4px 0"}))
        for kind, count in spec_counts.items():
            rows.append(html.Div(f"{kind}: {count}", style={"paddingLeft": "12px"}))

    return html.Div(rows)


def _build_stats(explicit: ExplicitResult, bdd: BddResult,
                  cycle: CycleAnalysisResult | None = None) -> html.Div:
    """Build the statistics comparison panel."""
    items = [
        html.H5("Explicit-State", style={"margin": "4px 0"}),
        html.Div(f"Total states: {explicit.total_states}"),
        html.Div(f"Initial states: {len(explicit.initial_states)}"),
        html.Div(f"Reachable states: {len(explicit.reachable_states)}"),
        html.Div(f"Transitions: {len(set(explicit.transitions))}"),
        html.Div(f"BFS depth: {len(explicit.bfs_layers)}"),

        html.Hr(style={"margin": "8px 0"}),

        html.H5("Symbolic (BDD)", style={"margin": "4px 0"}),
        html.Div(f"Reachable states: {bdd.total_reachable}"),
        html.Div(f"BDD vars (current): {sum(e.n_bits for e in bdd.encoding.values())}"),
        html.Div(f"BDD vars (total): {2 * sum(e.n_bits for e in bdd.encoding.values())}"),
        html.Div(f"Init BDD nodes: {bdd.init_node_count}"),
        html.Div(f"Trans BDD nodes: {bdd.trans_node_count}"),
        html.Div(f"Reached BDD nodes: {bdd.reached_node_count}"),
        html.Div(f"Fixpoint iterations: {len(bdd.iterations)}"),

        html.Hr(style={"margin": "8px 0"}),

        html.H5("Verification", style={"margin": "4px 0"}),
        html.Div(
            f"Match: {'YES' if len(explicit.reachable_states) == bdd.total_reachable else 'NO'}",
            style={"color": "#27ae60" if len(explicit.reachable_states) == bdd.total_reachable
                   else "#e74c3c",
                   "fontWeight": "bold"}),
    ]

    if cycle:
        items.extend([
            html.Hr(style={"margin": "8px 0"}),
            html.H5("Cycle Analysis", style={"margin": "4px 0"}),
            html.Div(f"Non-trivial SCCs: {len(cycle.nontrivial_sccs)}"),
            html.Div(f"Repeatable (R*): {len(cycle.r_star)} / "
                      f"{len(cycle.r_star) + len(cycle.transient_states)}"),
            html.Div(f"Transient: {len(cycle.transient_states)}"),
            html.Div(f"R(1) self-loops: {len(cycle.r_sets.get(1, set()))}"),
            html.Div(f"Convergence: R({cycle.convergence_step})"),
        ])

    return html.Div(items)


def _build_bdd_view(bdd_result: BddResult, selector: str) -> tuple[list, html.Div]:
    """Build BDD visualization elements and info panel."""
    if selector == "init":
        node = bdd_result.init_bdd
        label = "Initial States"
        node_count = bdd_result.init_node_count
    elif selector == "trans":
        node = bdd_result.trans_bdd
        label = "Transition Relation"
        node_count = bdd_result.trans_node_count
    elif selector == "reached":
        node = bdd_result.reached_bdd
        label = "Reachable States"
        node_count = bdd_result.reached_node_count
    elif selector == "domain":
        node = bdd_result.domain_constraint
        label = "Domain Constraint"
        node_count = len(node)
    elif selector.startswith("iter_"):
        idx = int(selector.split("_")[1])
        if idx < len(bdd_result.iteration_bdds):
            node = bdd_result.iteration_bdds[idx]
            label = f"Reached after Iteration {idx}"
            node_count = len(node)
        else:
            return [], html.Div("Iteration not available.")
    else:
        return [], ""

    elements = get_bdd_structure(node, bdd_result.bdd, max_nodes=150)

    # Info panel
    summary = get_bdd_summary(bdd_result)
    info_items = [
        html.H5(label, style={"margin": "4px 0"}),
        html.Div(f"BDD nodes: {node_count}"),
    ]

    # Reduction stats
    reduction = get_reduction_stats(node, bdd_result.bdd, bdd_result.encoding)
    info_items.append(html.Div(
        f"Full tree: {reduction['full_tree_nodes']:,} nodes → "
        f"ROBDD: {reduction['robdd_nodes']} nodes "
        f"({reduction['reduction_pct']}% reduction)",
        style={"fontSize": "11px", "color": "#7f8c8d", "marginTop": "4px"},
    ))

    if selector == "reached":
        info_items.append(html.Div(f"States represented: {bdd_result.total_reachable}"))
        info_items.append(html.H6("Fixpoint Iterations:", style={"margin": "8px 0 4px"}))
        for it in summary["iterations"]:
            info_items.append(html.Div(
                f"  Iter {it['iteration']}: +{it['new_states']} new, "
                f"{it['total_reachable']} total, {it['bdd_nodes']} BDD nodes",
                style={"fontSize": "11px", "paddingLeft": "8px"},
            ))

    if selector == "domain":
        info_items.append(html.Div(
            "Constrains BDD bit patterns to valid domain values. "
            "Invalid codes (e.g., code 5 for a 3-value enum using 3 bits) are excluded.",
            style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
        ))

    # Encoding table
    info_items.append(html.H6("Binary Encoding:", style={"margin": "8px 0 4px"}))
    for enc_info in summary["encoding"]:
        info_items.append(html.Div(
            f"  {enc_info['variable']}: {enc_info['domain_size']} values, "
            f"{enc_info['bits']} bits ({enc_info['bdd_vars']})",
            style={"fontSize": "11px", "paddingLeft": "8px"},
        ))

    # Truth table for small BDDs (not for transition relation which uses primed vars)
    if selector not in ("trans",):
        var_names = list(bdd_result.encoding.keys())
        truth_table = build_truth_table(
            node, bdd_result.bdd, bdd_result.encoding, var_names, max_rows=64
        )
        if truth_table is not None and len(truth_table) <= 64:
            info_items.append(html.H6("Satisfying Assignments:", style={"margin": "8px 0 4px"}))
            # Build HTML table
            header = html.Tr([html.Th(v, style={"padding": "2px 6px", "fontSize": "10px",
                                                  "borderBottom": "1px solid #ccc"})
                              for v in var_names])
            rows = []
            for row in truth_table:
                cells = [html.Td(str(row.get(v, "?")),
                                 style={"padding": "2px 6px", "fontSize": "10px"})
                         for v in var_names]
                rows.append(html.Tr(cells))
            info_items.append(html.Div(
                html.Table([html.Thead(header), html.Tbody(rows)],
                           style={"borderCollapse": "collapse"}),
                style={"maxHeight": "150px", "overflowY": "auto", "marginTop": "4px"},
            ))

    # ROBDD explanation
    info_items.append(html.H6("ROBDD Properties:", style={"margin": "8px 0 4px"}))
    info_items.append(html.Div([
        html.Div("• Green (solid) edges = high/then (variable = 1)",
                 style={"fontSize": "10px", "color": "#27ae60"}),
        html.Div("• Red (dashed) edges = low/else (variable = 0)",
                 style={"fontSize": "10px", "color": "#e74c3c"}),
        html.Div("• Redundant test removal: nodes where high = low are eliminated",
                 style={"fontSize": "10px", "color": "#666"}),
        html.Div("• Isomorphic subgraph merging: identical sub-BDDs share one node",
                 style={"fontSize": "10px", "color": "#666"}),
    ]))

    return elements, html.Div(info_items)


def _serialize_model(model: SmvModel) -> dict:
    """Serialize model info for dcc.Store."""
    return {
        "variables": {
            name: {"name": name, "domain_size": len(get_domain(vd))}
            for name, vd in model.variables.items()
        },
        "n_defines": len(model.defines),
        "n_specs": len(model.specs),
    }


def _build_slider_marks(max_val: int) -> dict:
    """Build slider marks for the R(n) step slider."""
    marks = {1: "1"}
    if max_val <= 10:
        for i in range(1, max_val + 1):
            marks[i] = str(i)
    else:
        # Show key marks only to avoid crowding
        marks[max_val] = str(max_val)
        step = max(1, max_val // 5)
        for i in range(step, max_val, step):
            marks[i] = str(i)
    return marks


def _serialize_explicit(result: ExplicitResult) -> dict:
    """Serialize explicit result summary for dcc.Store."""
    return {
        "var_names": result.var_names,
        "total_states": result.total_states,
        "n_initial": len(result.initial_states),
        "n_reachable": len(result.reachable_states),
        "n_transitions": len(result.transitions),
        "n_bfs_layers": len(result.bfs_layers),
    }


def _run_buchi_analysis(spec_idx, text, graph_opts, filter_expr, max_nodes):
    """Run the full LTL Buchi analysis pipeline.

    Returns (buchi_info, buchi_elements, buchi_panel_style, graph_elements).
    """
    model = parse_smv(text)

    # 1. Get the selected LTLSPEC
    ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
    spec = None
    for s in model.specs:
        if s.kind == "LTLSPEC" and model.specs.index(s) == spec_idx:
            spec = s
            break
    if spec is None:
        return (html.Div("No LTLSPEC found at that index.",
                         style={"color": "#e74c3c"}),
                no_update, no_update, no_update)

    info_items = []
    original_text = expr_to_str(spec.expr)
    info_items.append(html.Div([
        html.B("1. Original: "), html.Span(f"\u03c6 = {original_text}"),
    ], style={"marginBottom": "4px"}))

    # 2. Negate and build Buchi
    negated = simplify_ltl(negate_ltl(spec.expr))
    negated_text = expr_to_str(negated)
    info_items.append(html.Div([
        html.B("2. Negated: "), html.Span(f"\u00ac\u03c6 = {negated_text}"),
    ], style={"marginBottom": "4px"}))

    try:
        buchi = build_buchi_for_spec(spec)
    except UnsupportedLTLPattern as e:
        info_items.append(html.Div([
            html.B("Error: "),
            html.Span(f"Unsupported LTL pattern: {e}",
                      style={"color": "#e74c3c"}),
        ]))
        return (html.Div(info_items), [], {"display": "none", "marginBottom": "12px"},
                no_update)

    n_states = len(buchi.states)
    n_trans = len(buchi.transitions)
    acc_names = ", ".join(buchi.accepting)
    info_items.append(html.Div([
        html.B("3. Buchi for \u00ac\u03c6: "),
        html.Span(f"{n_states} states, {n_trans} transitions, "
                   f"accepting: {{{acc_names}}}"),
    ], style={"marginBottom": "4px"}))

    # Build Buchi graph elements
    buchi_elements = build_buchi_elements(buchi)

    # 3. Compose product
    product_info = compose_product(model, buchi)
    product_model = product_info.product_model

    # 4. Explore product
    product_result = explore(product_model)
    n_reachable = len(product_result.reachable_states)
    n_total = product_result.total_states
    info_items.append(html.Div([
        html.B("4. Product: "),
        html.Span(f"{n_total} total, {n_reachable} reachable"),
    ], style={"marginBottom": "4px"}))

    # 5. Cycle analysis on product
    product_cycles = analyze_cycles(product_result)

    # 6. Find accepting cycles
    accepting = find_accepting_cycles(product_result, product_cycles, product_info)

    if accepting.has_accepting_cycle:
        n_acc_sccs = len(accepting.accepting_sccs)
        acc_scc_sizes = ", ".join(str(len(s)) for s in accepting.accepting_sccs)
        info_items.append(html.Div([
            html.B("5. Accepting cycles: "),
            html.Span("FOUND ", style={"color": "#e74c3c", "fontWeight": "bold"}),
            html.Span(f"(property VIOLATED)"),
        ], style={"marginBottom": "2px"}))
        info_items.append(html.Div(
            f"   Accepting SCCs: {n_acc_sccs} ({acc_scc_sizes} states)",
            style={"paddingLeft": "16px", "marginBottom": "4px"},
        ))
    else:
        info_items.append(html.Div([
            html.B("5. Accepting cycles: "),
            html.Span("NONE ", style={"color": "#27ae60", "fontWeight": "bold"}),
            html.Span(f"(property HOLDS)"),
        ], style={"marginBottom": "4px"}))

    # 7. Lasso extraction + visualization on ORIGINAL graph
    graph_elements = no_update
    if accepting.lasso:
        prefix, cycle = accepting.lasso
        info_items.append(html.Div([
            html.B("6. Lasso: "),
            html.Span(f"prefix {len(prefix)} states + cycle {len(cycle)} states"),
        ], style={"marginBottom": "4px"}))

        # Project lasso to original model variables
        proj_prefix, proj_cycle = project_trace(accepting.lasso, product_info)

        # Build original model graph and overlay the projected lasso
        cached = _get_cached(text)
        if cached:
            orig_explicit, _, _ = cached
        else:
            orig_explicit = explore(model)
        reachable_only = "reachable_only" in (graph_opts or [])
        graph_elements = build_elements(
            orig_explicit,
            reachable_only=reachable_only,
            max_nodes=max_nodes or 500,
            filter_expr=filter_expr or "",
        )
        # Build projected lasso as state tuples for overlay
        proj_prefix_tuples = _dicts_to_tuples(proj_prefix, orig_explicit.var_names)
        proj_cycle_tuples = _dicts_to_tuples(proj_cycle, orig_explicit.var_names)
        projected_lasso = (proj_prefix_tuples, proj_cycle_tuples)
        graph_elements = apply_lasso_overlay(
            graph_elements, projected_lasso, orig_explicit,
        )

        # Show projected lasso trace details
        info_items.append(html.Div([
            html.B("Counterexample (projected):"),
        ], style={"marginTop": "6px", "marginBottom": "2px"}))

        # Prefix
        if proj_prefix:
            info_items.append(html.Div("Prefix:", style={
                "fontWeight": "bold", "paddingLeft": "8px", "fontSize": "10px",
            }))
            for i, sd in enumerate(proj_prefix):
                vals = ", ".join(f"{k}={v}" for k, v in sd.items())
                info_items.append(html.Div(
                    f"  [{i}] {vals}",
                    style={"paddingLeft": "16px", "fontSize": "10px",
                           "fontFamily": "Consolas, monospace"},
                ))

        # Cycle
        if proj_cycle:
            info_items.append(html.Div("Cycle:", style={
                "fontWeight": "bold", "paddingLeft": "8px", "fontSize": "10px",
            }))
            for i, sd in enumerate(proj_cycle):
                vals = ", ".join(f"{k}={v}" for k, v in sd.items())
                info_items.append(html.Div(
                    f"  [{i}] {vals}",
                    style={"paddingLeft": "16px", "fontSize": "10px",
                           "fontFamily": "Consolas, monospace"},
                ))

    # Result summary box
    if accepting.property_holds:
        result_style = {
            "backgroundColor": "#d4edda", "color": "#155724",
            "padding": "6px 10px", "borderRadius": "4px",
            "marginTop": "8px", "fontWeight": "bold",
        }
        result_text = f"\u2713 {original_text}"
    else:
        result_style = {
            "backgroundColor": "#f8d7da", "color": "#721c24",
            "padding": "6px 10px", "borderRadius": "4px",
            "marginTop": "8px", "fontWeight": "bold",
        }
        result_text = f"\u2717 {original_text}"
    info_items.append(html.Div(result_text, style=result_style))

    buchi_panel_style = {"display": "block", "marginBottom": "12px"}

    return (html.Div(info_items), buchi_elements, buchi_panel_style, graph_elements)


def _dicts_to_tuples(state_dicts: list[dict], var_names: list[str]) -> list[tuple]:
    """Convert list of state dicts to list of state tuples for graph overlay."""
    tuples = []
    for sd in state_dicts:
        tuples.append(tuple(sd.get(v) for v in var_names))
    return tuples
