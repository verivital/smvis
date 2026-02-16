"""Build Cytoscape.js elements from explicit-state exploration results."""
from __future__ import annotations
import math
from smvis.explicit_engine import ExplicitResult, State
from smvis.cycle_analysis import CycleAnalysisResult
from smvis.smv_model import expr_to_str


def build_elements(result: ExplicitResult,
                   reachable_only: bool = True,
                   max_nodes: int = 500,
                   filter_expr: str = "") -> list[dict]:
    """Build Cytoscape elements (nodes + edges) from explicit results.

    Args:
        result: Explicit exploration result.
        reachable_only: If True, only show reachable states.
        max_nodes: Maximum number of nodes to display.
        filter_expr: Optional filter like "mode=on" or "pc=l1".

    Returns:
        List of Cytoscape element dicts.
    """
    var_names = result.var_names
    initial_set = set(result.initial_states)
    reachable_set = result.reachable_states

    # Determine which states to show
    if reachable_only:
        show_states = reachable_set
    else:
        show_states = set(result.state_to_dict.keys())

    # Apply filter
    if filter_expr.strip():
        show_states = _apply_filter(show_states, result.state_to_dict,
                                    var_names, filter_expr)

    # Compute BFS depth for each state
    depth_map = {}
    for depth, layer in enumerate(result.bfs_layers):
        for sk in layer:
            if sk not in depth_map:
                depth_map[sk] = depth

    # Limit number of nodes (keep initial states + BFS order)
    if len(show_states) > max_nodes:
        sorted_states = sorted(show_states, key=lambda s: depth_map.get(s, 999999))
        show_states = set(sorted_states[:max_nodes])
        # Always include initial states
        show_states |= (initial_set & reachable_set)

    # Build node elements
    elements = []
    node_ids = set()
    for sk in show_states:
        sd = result.state_to_dict.get(sk)
        if sd is None:
            continue
        node_id = _state_id(sk)
        node_ids.add(node_id)

        label = _state_label(sd, var_names)
        classes = []
        if sk in initial_set:
            classes.append("initial")
        if sk in reachable_set:
            classes.append("reachable")
        else:
            classes.append("unreachable")

        depth = depth_map.get(sk, -1)
        elements.append({
            "data": {
                "id": node_id,
                "label": label,
                "depth": depth,
                **{v: str(sd[v]) for v in var_names},
            },
            "classes": " ".join(classes),
        })

    # Build edge elements
    seen_edges = set()
    for src, dst in result.transitions:
        src_id = _state_id(src)
        dst_id = _state_id(dst)
        if src_id in node_ids and dst_id in node_ids:
            edge_key = (src_id, dst_id)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                is_self = (src == dst)
                elements.append({
                    "data": {
                        "id": f"e_{src_id}_{dst_id}",
                        "source": src_id,
                        "target": dst_id,
                    },
                    "classes": "self-loop" if is_self else "",
                })

    return elements


def _state_id(state: State) -> str:
    return "s_" + "_".join(str(v) for v in state)


def _state_label(sd: dict, var_names: list[str]) -> str:
    parts = []
    for v in var_names:
        val = sd[v]
        if isinstance(val, bool):
            parts.append("T" if val else "F")
        else:
            parts.append(str(val))
    return ",".join(parts)


def _apply_filter(states: set[State], state_map: dict,
                  var_names: list[str], filter_expr: str) -> set[State]:
    """Apply a simple filter expression like 'mode=on' or 'pc=l1,a=3'."""
    conditions = []
    for part in filter_expr.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        conditions.append((key.strip(), val.strip()))

    if not conditions:
        return states

    filtered = set()
    for sk in states:
        sd = state_map.get(sk)
        if sd is None:
            continue
        match = True
        for key, val in conditions:
            if key in sd:
                sv = str(sd[key])
                if sv != val:
                    match = False
                    break
        if match:
            filtered.add(sk)
    return filtered


def get_state_detail(result: ExplicitResult, state_id: str) -> dict | None:
    """Get detailed info about a state from its Cytoscape node ID."""
    for sk, sd in result.state_to_dict.items():
        if _state_id(sk) == state_id:
            # Find successors and predecessors
            succs = set()
            preds = set()
            for src, dst in result.transitions:
                if src == sk:
                    succs.add(dst)
                if dst == sk:
                    preds.add(src)
            return {
                "state": sd,
                "is_initial": sk in result.initial_states,
                "is_reachable": sk in result.reachable_states,
                "successor_count": len(succs),
                "predecessor_count": len(preds),
            }
    return None


def compute_concentric_positions(elements: list[dict],
                                  spacing: int = 120) -> list[dict]:
    """Compute concentric ring positions based on BFS depth.

    Nodes at the same BFS depth are arranged in a ring. Returns the
    same elements list with ``position`` dicts added to node elements.
    Use with ``layout={"name": "preset"}``.
    """
    nodes = [e for e in elements if "source" not in e.get("data", {})]
    edges = [e for e in elements if "source" in e.get("data", {})]

    # Group nodes by depth
    depth_groups: dict[int, list[dict]] = {}
    for node in nodes:
        d = node["data"].get("depth", -1)
        depth_groups.setdefault(d, []).append(node)

    for d in sorted(depth_groups.keys()):
        group = depth_groups[d]
        n = len(group)
        if d <= 0 and n == 1:
            radius = 0
        else:
            radius = max(d, 1) * spacing
        for i, node in enumerate(group):
            angle = 2 * math.pi * i / max(n, 1)
            node["position"] = {
                "x": round(radius * math.cos(angle)),
                "y": round(radius * math.sin(angle)),
            }

    return nodes + edges


def apply_trace_overlay(elements: list[dict],
                        trace_states: list[dict[str, str]],
                        state_to_dict: dict,
                        var_names: list[str],
                        loop_start: int | None = None) -> list[dict]:
    """Add trace highlighting CSS classes to nodes/edges in a counterexample path.

    Args:
        elements: Existing Cytoscape elements.
        trace_states: List of state dicts from a nuXmv trace.
        state_to_dict: Map from state tuples to variable dicts.
        var_names: Variable names (to exclude DEFINEs from matching).
        loop_start: Index of loop-back state for lasso traces.

    Returns:
        Modified elements with trace-node/trace-edge classes added.
    """
    # Map each trace state to a node ID
    var_set = set(var_names)
    trace_node_ids: list[str] = []

    for ts in trace_states:
        # Only compare actual variables (skip DEFINEs)
        trace_vars = {k: v for k, v in ts.items() if k in var_set}
        matched_id = None
        for state_tuple, sd in state_to_dict.items():
            if all(_normalize_val(sd.get(k)) == _normalize_val(v)
                   for k, v in trace_vars.items()):
                matched_id = _state_id(state_tuple)
                break
        if matched_id:
            trace_node_ids.append(matched_id)

    if not trace_node_ids:
        return elements

    # Build set of trace edges (consecutive pairs)
    trace_edges = set()
    for i in range(len(trace_node_ids) - 1):
        trace_edges.add((trace_node_ids[i], trace_node_ids[i + 1]))
    # Loop-back edge
    loop_edge = None
    if loop_start is not None and 0 <= loop_start < len(trace_node_ids):
        loop_edge = (trace_node_ids[-1], trace_node_ids[loop_start])

    trace_set = set(trace_node_ids)

    for elem in elements:
        d = elem["data"]
        classes = elem.get("classes", "")

        if "source" not in d:
            # Node element
            if d["id"] in trace_set:
                classes += " trace-node"
                # Add step numbers
                steps = [i for i, nid in enumerate(trace_node_ids) if nid == d["id"]]
                elem["data"]["trace_step"] = steps[0]
                elem["data"]["trace_label"] = ",".join(f"[{s}]" for s in steps)
        else:
            # Edge element
            edge_key = (d["source"], d["target"])
            if loop_edge and edge_key == loop_edge:
                classes += " trace-loop"
            elif edge_key in trace_edges:
                classes += " trace-edge"

        elem["classes"] = classes.strip()

    return elements


def apply_repeatable_overlay(
    elements: list[dict],
    cycle_result: CycleAnalysisResult,
    mode: str = "r_star",
    step_n: int = 1,
) -> list[dict]:
    """Add repeatable-state CSS classes to Cytoscape elements.

    Args:
        elements: Existing Cytoscape elements.
        cycle_result: Result from cycle analysis.
        mode: Visualization mode.
            "r_star" — All repeatable nodes purple, transient gray, cycle edges highlighted.
            "r_n"   — Cumulative view: slider at step k shows R(1) | ... | R(k).
                       Nodes colored by min return time (heat map).
            "scc"   — Color by SCC membership (8 rotating colors).
        step_n: Current slider step for r_n mode.

    Returns:
        Modified elements with overlay classes added.
    """
    # Build lookup: state tuple -> SCC id and min return time
    # We need to map node IDs back to state tuples
    r_star = cycle_result.r_star
    transient = cycle_result.transient_states
    min_rt = cycle_result.min_return_time
    scc_id_map = cycle_result.state_to_scc_id
    cumulative = cycle_result.cumulative_r

    # Build set of node IDs for repeatable/transient states
    r_star_ids: dict[str, State] = {}
    transient_ids: set[str] = set()
    for s in r_star:
        r_star_ids[_state_id(s)] = s
    for s in transient:
        transient_ids.add(_state_id(s))

    # Build cycle edge set (as node ID pairs)
    cycle_edge_ids: set[tuple[str, str]] = set()
    for src, dst in cycle_result.cycle_edges:
        cycle_edge_ids.add((_state_id(src), _state_id(dst)))

    # Heat map colors for R(n) layers (by min return time)
    _HEAT_CLASSES = [
        "repeat-1", "repeat-2", "repeat-3", "repeat-4",
        "repeat-5", "repeat-6", "repeat-high",
    ]

    # For r_n mode, pre-compute which node IDs are "active" (in cumulative R(step_n))
    active_ids: set[str] = set()
    if mode == "r_n":
        for s in r_star:
            mrt = min_rt.get(s)
            if mrt is not None and mrt <= step_n:
                active_ids.add(_state_id(s))

    for elem in elements:
        d = elem["data"]
        classes = elem.get("classes", "")

        if "source" not in d:
            # Node element
            node_id = d["id"]

            if mode == "r_star":
                if node_id in r_star_ids:
                    classes += " repeatable"
                    s = r_star_ids[node_id]
                    d["repeat_info"] = f"R({min_rt.get(s, '?')})"
                elif node_id in transient_ids:
                    classes += " transient"

            elif mode == "r_n":
                if node_id in r_star_ids:
                    s = r_star_ids[node_id]
                    mrt = min_rt.get(s)
                    if mrt is not None and mrt <= step_n:
                        # State is in cumulative R(step_n) — color by min return
                        idx = min(mrt - 1, len(_HEAT_CLASSES) - 1)
                        classes += f" {_HEAT_CLASSES[idx]}"
                        d["repeat_info"] = f"R({mrt})"
                    else:
                        classes += " not-yet-repeatable"
                elif node_id in transient_ids:
                    classes += " transient"

            elif mode == "scc":
                if node_id in r_star_ids:
                    s = r_star_ids[node_id]
                    scc_idx = scc_id_map.get(s, 0) % 8
                    classes += f" scc-{scc_idx}"
                elif node_id in transient_ids:
                    classes += " transient"

        else:
            # Edge element
            edge_key = (d["source"], d["target"])
            if mode == "r_star" and edge_key in cycle_edge_ids:
                classes += " cycle-edge"
            elif mode == "r_n":
                if edge_key in cycle_edge_ids and d["source"] in active_ids and d["target"] in active_ids:
                    classes += " cycle-edge"
                elif d["source"] not in active_ids or d["target"] not in active_ids:
                    classes += " faded-edge"

        elem["classes"] = classes.strip()

    return elements


def _normalize_val(val) -> str:
    """Normalize values for trace comparison (nuXmv TRUE/FALSE vs Python True/False)."""
    s = str(val)
    if s in ("TRUE", "True"):
        return "True"
    if s in ("FALSE", "False"):
        return "False"
    return s


def build_buchi_elements(buchi) -> list[dict]:
    """Build Cytoscape elements for a Buchi automaton visualization.

    Args:
        buchi: BuchiAutomaton from ltl_buchi module.

    Returns:
        List of Cytoscape element dicts for the small Buchi graph.
    """
    elements = []
    n = len(buchi.states)

    for i, state in enumerate(buchi.states):
        classes = ["buchi-state"]
        if state.accepting:
            classes.append("buchi-accepting")
        if state.name == buchi.initial:
            classes.append("buchi-initial")

        # Circle layout for 1-3 states
        angle = 2 * math.pi * i / max(n, 1) - math.pi / 2
        radius = 0 if n == 1 else 80
        elements.append({
            "data": {
                "id": f"b_{state.name}",
                "label": state.name,
            },
            "classes": " ".join(classes),
            "position": {
                "x": round(radius * math.cos(angle)),
                "y": round(radius * math.sin(angle)),
            },
        })

    for t in buchi.transitions:
        guard_str = expr_to_str(t.guard)
        is_self = t.src == t.dst
        elements.append({
            "data": {
                "id": f"be_{t.src}_{t.dst}_{guard_str[:20]}",
                "source": f"b_{t.src}",
                "target": f"b_{t.dst}",
                "label": guard_str,
            },
            "classes": "buchi-edge" + (" self-loop" if is_self else ""),
        })

    return elements


def apply_lasso_overlay(
    elements: list[dict],
    lasso: tuple[list[State], list[State]],
    explicit_result: ExplicitResult,
) -> list[dict]:
    """Highlight lasso counterexample (prefix + cycle) on graph elements.

    Args:
        elements: Existing Cytoscape elements for the product graph.
        lasso: (prefix_states, cycle_states) tuples from accepting_cycles.
        explicit_result: Product exploration result.

    Returns:
        Modified elements with lasso overlay classes.
    """
    prefix, cycle = lasso

    prefix_ids = {_state_id(s) for s in prefix}
    cycle_ids = {_state_id(s) for s in cycle}

    # Build prefix edge set
    prefix_edges: set[tuple[str, str]] = set()
    for i in range(len(prefix) - 1):
        prefix_edges.add((_state_id(prefix[i]), _state_id(prefix[i + 1])))

    # Build cycle edge set
    cycle_edges: set[tuple[str, str]] = set()
    for i in range(len(cycle) - 1):
        cycle_edges.add((_state_id(cycle[i]), _state_id(cycle[i + 1])))

    for elem in elements:
        d = elem["data"]
        classes = elem.get("classes", "")

        if "source" not in d:
            # Node
            nid = d["id"]
            if nid in cycle_ids:
                classes += " lasso-cycle"
            elif nid in prefix_ids:
                classes += " lasso-prefix"
        else:
            # Edge
            edge_key = (d["source"], d["target"])
            if edge_key in cycle_edges:
                classes += " lasso-cycle-edge"
            elif edge_key in prefix_edges:
                classes += " lasso-prefix-edge"

        elem["classes"] = classes.strip()

    return elements


CYTO_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "font-size": "7px",
            "width": 25,
            "height": 25,
            "background-color": "#bbb",
            "text-valign": "center",
            "text-halign": "center",
            "color": "#333",
            "text-wrap": "wrap",
            "text-max-width": "60px",
        },
    },
    {
        "selector": "node.reachable",
        "style": {
            "background-color": "#4a90d9",
            "color": "#fff",
        },
    },
    {
        "selector": "node.initial",
        "style": {
            "border-width": 3,
            "border-color": "#2ecc71",
        },
    },
    {
        "selector": "node.unreachable",
        "style": {
            "background-color": "#eee",
            "opacity": 0.4,
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "border-width": 3,
            "border-color": "#e74c3c",
            "background-color": "#f39c12",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "width": 1,
            "line-color": "#999",
            "target-arrow-color": "#999",
            "arrow-scale": 0.8,
        },
    },
    {
        "selector": "edge.self-loop",
        "style": {
            "curve-style": "loop",
            "loop-direction": "-45deg",
            "loop-sweep": "90deg",
        },
    },
    # Repeatable states: R* mode
    {"selector": "node.repeatable", "style": {
        "background-color": "#8e44ad", "color": "#fff",
    }},
    {"selector": "node.transient", "style": {
        "background-color": "#bdc3c7", "opacity": 0.5,
    }},
    # Repeatable states: R(n) heat map (by min return time)
    {"selector": "node.repeat-1", "style": {
        "background-color": "#e74c3c", "color": "#fff",
    }},
    {"selector": "node.repeat-2", "style": {
        "background-color": "#e67e22", "color": "#fff",
    }},
    {"selector": "node.repeat-3", "style": {
        "background-color": "#f1c40f", "color": "#333",
    }},
    {"selector": "node.repeat-4", "style": {
        "background-color": "#2ecc71", "color": "#fff",
    }},
    {"selector": "node.repeat-5", "style": {
        "background-color": "#3498db", "color": "#fff",
    }},
    {"selector": "node.repeat-6", "style": {
        "background-color": "#9b59b6", "color": "#fff",
    }},
    {"selector": "node.repeat-high", "style": {
        "background-color": "#8e44ad", "color": "#fff",
    }},
    # States not yet in cumulative R(n) at current slider position
    {"selector": "node.not-yet-repeatable", "style": {
        "background-color": "#ddd", "opacity": 0.4,
    }},
    # Cycle edges (within nontrivial SCCs)
    {"selector": "edge.cycle-edge", "style": {
        "line-color": "#8e44ad", "target-arrow-color": "#8e44ad", "width": 2,
    }},
    # Faded edges (not connecting active nodes in R(n) mode)
    {"selector": "edge.faded-edge", "style": {
        "line-color": "#e0e0e0", "target-arrow-color": "#e0e0e0",
        "opacity": 0.25, "width": 1,
    }},
    # SCC membership colors (8 rotating)
    {"selector": "node.scc-0", "style": {"background-color": "#e74c3c", "color": "#fff"}},
    {"selector": "node.scc-1", "style": {"background-color": "#3498db", "color": "#fff"}},
    {"selector": "node.scc-2", "style": {"background-color": "#2ecc71", "color": "#fff"}},
    {"selector": "node.scc-3", "style": {"background-color": "#f39c12", "color": "#fff"}},
    {"selector": "node.scc-4", "style": {"background-color": "#9b59b6", "color": "#fff"}},
    {"selector": "node.scc-5", "style": {"background-color": "#1abc9c", "color": "#fff"}},
    {"selector": "node.scc-6", "style": {"background-color": "#e67e22", "color": "#fff"}},
    {"selector": "node.scc-7", "style": {"background-color": "#34495e", "color": "#fff"}},
    # Trace overlay styles (counterexample visualization)
    {
        "selector": "node.trace-node",
        "style": {
            "border-color": "#e67e22",
            "border-width": 4,
            "background-color": "#fdebd0",
            "color": "#333",
            "z-index": 999,
            "label": "data(trace_label)",
            "font-size": "9px",
            "font-weight": "bold",
        },
    },
    {
        "selector": "edge.trace-edge",
        "style": {
            "line-color": "#e67e22",
            "target-arrow-color": "#e67e22",
            "width": 3,
            "z-index": 998,
        },
    },
    {
        "selector": "edge.trace-loop",
        "style": {
            "line-color": "#e74c3c",
            "line-style": "dashed",
            "target-arrow-color": "#e74c3c",
            "width": 3,
            "z-index": 998,
        },
    },
    # Lasso counterexample overlay
    {"selector": "node.lasso-cycle", "style": {
        "background-color": "#fadbd8", "border-color": "#e74c3c",
        "border-width": 3, "color": "#c0392b",
    }},
    {"selector": "node.lasso-prefix", "style": {
        "background-color": "#fef9e7", "border-color": "#f39c12",
        "border-width": 2, "color": "#d68910",
    }},
    {"selector": "edge.lasso-cycle-edge", "style": {
        "line-color": "#e74c3c", "target-arrow-color": "#e74c3c",
        "width": 3, "z-index": 997,
    }},
    {"selector": "edge.lasso-prefix-edge", "style": {
        "line-color": "#f39c12", "target-arrow-color": "#f39c12",
        "width": 2, "z-index": 996,
    }},
]


BUCHI_STYLESHEET = [
    {"selector": "node.buchi-state", "style": {
        "label": "data(label)", "font-size": "11px",
        "width": 40, "height": 40,
        "background-color": "#d5e8d4", "color": "#333",
        "text-valign": "center", "text-halign": "center",
        "border-width": 2, "border-color": "#82b366",
    }},
    {"selector": "node.buchi-accepting", "style": {
        "border-width": 5, "border-color": "#e74c3c",
        "border-style": "double", "background-color": "#fadbd8",
    }},
    {"selector": "node.buchi-initial", "style": {
        "border-color": "#2ecc71", "border-width": 3,
    }},
    {"selector": "edge.buchi-edge", "style": {
        "label": "data(label)", "font-size": "8px",
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "width": 2, "line-color": "#666",
        "target-arrow-color": "#666", "arrow-scale": 1.0,
        "text-rotation": "autorotate",
        "text-background-color": "#fff",
        "text-background-opacity": 0.8,
        "text-background-padding": "2px",
    }},
    {"selector": "edge.buchi-edge.self-loop", "style": {
        "curve-style": "loop",
        "loop-direction": "-45deg", "loop-sweep": "90deg",
    }},
]

BDD_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "font-size": "9px",
            "width": 30,
            "height": 30,
            "background-color": "#7fb3e0",
            "text-valign": "center",
            "text-halign": "center",
            "shape": "ellipse",
        },
    },
    {
        "selector": "node.terminal-true",
        "style": {
            "background-color": "#2ecc71",
            "shape": "rectangle",
            "width": 25,
            "height": 25,
        },
    },
    {
        "selector": "node.terminal-false",
        "style": {
            "background-color": "#e74c3c",
            "shape": "rectangle",
            "width": 25,
            "height": 25,
        },
    },
    {
        "selector": "edge.high-edge",
        "style": {
            "line-color": "#2ecc71",
            "target-arrow-color": "#2ecc71",
            "target-arrow-shape": "triangle",
            "width": 2,
            "curve-style": "bezier",
        },
    },
    {
        "selector": "edge.low-edge",
        "style": {
            "line-color": "#e74c3c",
            "target-arrow-color": "#e74c3c",
            "target-arrow-shape": "triangle",
            "line-style": "dashed",
            "width": 2,
            "curve-style": "bezier",
        },
    },
    {
        "selector": "node.level-label",
        "style": {
            "background-opacity": 0,
            "border-width": 0,
            "label": "data(label)",
            "font-size": "8px",
            "color": "#888",
            "text-halign": "right",
            "text-valign": "center",
            "width": 10,
            "height": 10,
        },
    },
]
