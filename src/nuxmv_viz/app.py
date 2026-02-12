"""Dash web application for the nuXmv Model Visualizer."""
from __future__ import annotations
import os
import json
import traceback

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_cytoscape as cyto

from nuxmv_viz.smv_parser import parse_smv
from nuxmv_viz.smv_model import SmvModel, get_domain, expr_to_str
from nuxmv_viz.explicit_engine import explore, ExplicitResult
from nuxmv_viz.bdd_engine import build_from_explicit, BddResult, get_bdd_structure
from nuxmv_viz.graph_builder import build_elements, CYTO_STYLESHEET, BDD_STYLESHEET, get_state_detail
from nuxmv_viz.bdd_visualizer import get_bdd_summary

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
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    model_files = _find_model_files()
    default_model = model_files[0] if model_files else ""
    default_text = _load_model_text(default_model) if default_model else ""

    app.layout = html.Div([
        # ---- Header ----
        html.Div([
            html.H2("nuXmv Model Visualizer", style={"margin": "0", "flex": "1"}),
            html.Div([
                html.Label("Model:", style={"marginRight": "8px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id="model-selector",
                    options=[{"label": f, "value": f} for f in model_files],
                    value=default_model,
                    style={"width": "200px"},
                    clearable=False,
                ),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "padding": "10px 20px",
            "backgroundColor": "#2c3e50", "color": "#ecf0f1",
        }),

        # ---- Main Content ----
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
            ], style={"width": "30%", "padding": "12px", "overflowY": "auto"}),

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
                            dcc.Checklist(
                                id="graph-options",
                                options=[{"label": " Reachable only", "value": "reachable_only"}],
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
                        ], style={"display": "flex", "alignItems": "center"}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "alignItems": "center", "marginBottom": "4px"}),
                    cyto.Cytoscape(
                        id="state-graph",
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
                ], style={"marginBottom": "12px"}),

                # BDD Section
                html.Div([
                    html.Div([
                        html.H4("BDD Visualization", style={"margin": "0", "flex": "1"}),
                        dcc.Dropdown(
                            id="bdd-selector",
                            options=[
                                {"label": "Initial States", "value": "init"},
                                {"label": "Reachable States", "value": "reached"},
                                {"label": "Transition Relation", "value": "trans"},
                            ],
                            value="reached",
                            style={"width": "180px", "fontSize": "12px"},
                            clearable=False,
                        ),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "alignItems": "center", "marginBottom": "4px"}),
                    html.Div([
                        cyto.Cytoscape(
                            id="bdd-graph",
                            layout={"name": "breadthfirst", "directed": True,
                                    "animate": False},
                            style={"width": "50%", "height": "250px",
                                   "border": "1px solid #ddd",
                                   "display": "inline-block"},
                            stylesheet=BDD_STYLESHEET,
                            elements=[],
                        ),
                        html.Div(id="bdd-info", style={
                            "width": "48%", "display": "inline-block",
                            "verticalAlign": "top", "paddingLeft": "12px",
                            "fontSize": "12px",
                        }),
                    ]),
                ]),
            ], style={"width": "70%", "padding": "12px", "overflowY": "auto"}),
        ], style={"display": "flex", "height": "calc(100vh - 60px)"}),

        # ---- Hidden Stores ----
        dcc.Store(id="parsed-model-store", data=None),
        dcc.Store(id="explicit-result-store", data=None),
        dcc.Store(id="bdd-result-store", data=None),
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
        except Exception:
            pass
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
        Output("bdd-info", "children"),
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
        if not n or not text:
            return (no_update,) * 6
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
            # Build visualizations
            reachable_only = "reachable_only" in (graph_opts or [])
            graph_elements = build_elements(
                explicit_result,
                reachable_only=reachable_only,
                max_nodes=max_nodes or 500,
                filter_expr=filter_expr or "",
            )
            stats = _build_stats(explicit_result, bdd_result)
            bdd_summary = get_bdd_summary(bdd_result)
            bdd_elements, bdd_info = _build_bdd_view(bdd_result, bdd_sel or "reached")

            # Serialize for store (lightweight)
            explicit_data = _serialize_explicit(explicit_result)
            bdd_data = {"total_reachable": bdd_result.total_reachable}

            return explicit_data, bdd_data, stats, graph_elements, bdd_elements, bdd_info
        except Exception as e:
            err_msg = html.Div([
                html.Span(f"Error: {e}", style={"color": "#e74c3c"}),
                html.Pre(traceback.format_exc(), style={"fontSize": "10px"}),
            ])
            return None, None, err_msg, [], [], str(e)

    @app.callback(
        Output("state-graph", "elements", allow_duplicate=True),
        Output("state-graph", "layout"),
        Input("graph-options", "value"),
        Input("state-filter", "value"),
        Input("layout-selector", "value"),
        Input("max-nodes", "value"),
        State("explicit-result-store", "data"),
        State("smv-editor", "value"),
        prevent_initial_call=True,
    )
    def update_graph(graph_opts, filter_expr, layout, max_nodes, explicit_data, text):
        if not explicit_data or not text:
            return no_update, no_update
        try:
            model = parse_smv(text)
            explicit_result = explore(model)
            reachable_only = "reachable_only" in (graph_opts or [])
            elements = build_elements(
                explicit_result,
                reachable_only=reachable_only,
                max_nodes=max_nodes or 500,
                filter_expr=filter_expr or "",
            )
            layout_dict = {"name": layout or "cose", "animate": False}
            if layout == "cose":
                layout_dict["nodeRepulsion"] = 8000
                layout_dict["idealEdgeLength"] = 50
            elif layout == "breadthfirst":
                layout_dict["directed"] = True
            elif layout == "concentric":
                layout_dict["concentric"] = "function(n) { return n.data('depth') || 0; }"
            return elements, layout_dict
        except Exception:
            return no_update, no_update

    @app.callback(
        Output("bdd-graph", "elements", allow_duplicate=True),
        Output("bdd-info", "children", allow_duplicate=True),
        Input("bdd-selector", "value"),
        State("bdd-result-store", "data"),
        State("smv-editor", "value"),
        State("explicit-result-store", "data"),
        prevent_initial_call=True,
    )
    def update_bdd_view(bdd_sel, bdd_data, text, explicit_data):
        if not bdd_data or not text:
            return no_update, no_update
        try:
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
            return elements, info
        except Exception:
            return no_update, no_update

    @app.callback(
        Output("state-detail", "children"),
        Input("state-graph", "tapNodeData"),
        State("smv-editor", "value"),
    )
    def show_state_detail(node_data, text):
        if not node_data or not text:
            return ""
        try:
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
        except Exception:
            return ""

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


def _build_stats(explicit: ExplicitResult, bdd: BddResult) -> html.Div:
    """Build the statistics comparison panel."""
    return html.Div([
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
    ])


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
    else:
        return [], ""

    elements = get_bdd_structure(node, bdd_result.bdd, max_nodes=150)

    # Info panel
    summary = get_bdd_summary(bdd_result)
    info_items = [
        html.H5(label, style={"margin": "4px 0"}),
        html.Div(f"BDD nodes: {node_count}"),
    ]

    if selector == "reached":
        info_items.append(html.Div(f"States represented: {bdd_result.total_reachable}"))
        info_items.append(html.H6("Fixpoint Iterations:", style={"margin": "8px 0 4px"}))
        for it in summary["iterations"]:
            info_items.append(html.Div(
                f"  Iter {it['iteration']}: +{it['new_states']} new, "
                f"{it['total_reachable']} total, {it['bdd_nodes']} BDD nodes",
                style={"fontSize": "11px", "paddingLeft": "8px"},
            ))

    # Encoding table
    info_items.append(html.H6("Binary Encoding:", style={"margin": "8px 0 4px"}))
    for enc_info in summary["encoding"]:
        info_items.append(html.Div(
            f"  {enc_info['variable']}: {enc_info['domain_size']} values, "
            f"{enc_info['bits']} bits ({enc_info['bdd_vars']})",
            style={"fontSize": "11px", "paddingLeft": "8px"},
        ))

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
