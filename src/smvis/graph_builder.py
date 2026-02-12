"""Build Cytoscape.js elements from explicit-state exploration results."""
from __future__ import annotations
from smvis.explicit_engine import ExplicitResult, State


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
                    preds.add(dst)
            return {
                "state": sd,
                "is_initial": sk in result.initial_states,
                "is_reachable": sk in result.reachable_states,
                "successor_count": len(succs),
                "predecessor_count": len(preds),
            }
    return None


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
]
