"""GUI callback regression tests.

Exercises every callback x every model x every option permutation
to ensure no exceptions are raised. Catches issues like the BDD
dangling-edge bug without needing a browser.
"""
import os
import pytest
from smvis.smv_parser import parse_smv_file
from smvis.explicit_engine import explore
from smvis.bdd_engine import (
    build_from_explicit, get_bdd_structure, build_truth_table,
)
from smvis.graph_builder import (
    build_elements, compute_concentric_positions, get_state_detail,
)
from smvis.bdd_visualizer import get_bdd_summary, get_reduction_stats

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)

MODEL_FILES = sorted(f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".smv"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=MODEL_FILES)
def model_pipeline(request):
    """Parse -> explore -> BDD for each model."""
    path = os.path.join(EXAMPLES_DIR, request.param)
    model = parse_smv_file(path)
    explicit = explore(model)
    bdd = build_from_explicit(
        model,
        explicit.initial_states,
        explicit.transitions,
        explicit.var_names,
        explicit.state_to_dict,
    )
    return request.param, model, explicit, bdd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_cytoscape_elements(elements: list[dict]):
    """Validate that elements form a valid Cytoscape graph (no dangling edges)."""
    node_ids = set()
    for e in elements:
        d = e["data"]
        if "source" not in d:
            assert "id" in d, f"node missing 'id': {d}"
            node_ids.add(d["id"])
    for e in elements:
        d = e["data"]
        if "source" in d:
            assert d["source"] in node_ids, (
                f"edge source '{d['source']}' not in nodes"
            )
            assert d["target"] in node_ids, (
                f"edge target '{d['target']}' not in nodes"
            )


# ---------------------------------------------------------------------------
# Test: Graph builder permutations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reachable_only", [True, False])
@pytest.mark.parametrize("max_nodes", [10, 100, 500])
def test_build_elements_permutations(model_pipeline, reachable_only, max_nodes):
    """build_elements produces valid elements for all reachable/max_nodes combos."""
    _, _, explicit, _ = model_pipeline
    elems = build_elements(
        explicit, reachable_only=reachable_only, max_nodes=max_nodes,
    )
    assert isinstance(elems, list)
    _validate_cytoscape_elements(elems)
    # Node count should not exceed max_nodes (+ possible initial state extras)
    nodes = [e for e in elems if "source" not in e["data"]]
    assert len(nodes) <= max_nodes + len(explicit.initial_states)


# ---------------------------------------------------------------------------
# Test: Layout permutations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["cose", "breadthfirst", "grid", "circle", "concentric"])
def test_layouts(model_pipeline, layout):
    """All layout modes work without exception."""
    _, _, explicit, _ = model_pipeline
    elems = build_elements(explicit, reachable_only=True, max_nodes=200)
    if layout == "concentric":
        elems = compute_concentric_positions(elems)
        # Verify position dicts were added to nodes
        nodes = [e for e in elems if "source" not in e.get("data", {})]
        for node in nodes:
            assert "position" in node, f"concentric node missing position: {node['data']['id']}"
            assert "x" in node["position"]
            assert "y" in node["position"]
    _validate_cytoscape_elements(elems)


# ---------------------------------------------------------------------------
# Test: BDD selector permutations
# ---------------------------------------------------------------------------

BDD_STATIC_SELECTORS = ["init", "reached", "trans", "domain"]


def test_bdd_selectors(model_pipeline):
    """All BDD selectors produce valid elements."""
    name, _, _, bdd = model_pipeline
    selectors = list(BDD_STATIC_SELECTORS)
    # Add dynamic iteration selectors
    for i in range(len(bdd.iteration_bdds)):
        selectors.append(f"iter_{i}")

    for sel in selectors:
        # Resolve the BDD node
        if sel == "init":
            node = bdd.init_bdd
        elif sel == "reached":
            node = bdd.reached_bdd
        elif sel == "trans":
            node = bdd.trans_bdd
        elif sel == "domain":
            node = bdd.domain_constraint
        elif sel.startswith("iter_"):
            idx = int(sel.split("_")[1])
            node = bdd.iteration_bdds[idx]
        else:
            continue

        elems = get_bdd_structure(node, bdd.bdd)
        assert isinstance(elems, list), f"selector={sel} model={name}"
        _validate_cytoscape_elements(elems)


# ---------------------------------------------------------------------------
# Test: BDD edge integrity (regression for dangling-edge bug)
# ---------------------------------------------------------------------------

def test_bdd_structure_no_dangling_edges(model_pipeline):
    """Every edge target and source must reference an existing node."""
    _, _, _, bdd = model_pipeline
    all_nodes = [bdd.init_bdd, bdd.reached_bdd, bdd.trans_bdd,
                 bdd.domain_constraint] + bdd.iteration_bdds
    for bdd_node in all_nodes:
        elems = get_bdd_structure(bdd_node, bdd.bdd)
        node_ids = {e["data"]["id"] for e in elems if "source" not in e["data"]}
        for e in elems:
            if "source" in e["data"]:
                assert e["data"]["source"] in node_ids, (
                    f"dangling source: {e['data']['source']}"
                )
                assert e["data"]["target"] in node_ids, (
                    f"dangling target: {e['data']['target']}"
                )


# ---------------------------------------------------------------------------
# Test: BDD edge integrity with small max_nodes (stress test)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_nodes", [5, 20, 50])
def test_bdd_structure_small_max_nodes(model_pipeline, max_nodes):
    """BDD structure with tight max_nodes never has dangling edges."""
    _, _, _, bdd = model_pipeline
    for bdd_node in [bdd.trans_bdd, bdd.reached_bdd]:
        elems = get_bdd_structure(bdd_node, bdd.bdd, max_nodes=max_nodes)
        node_ids = {e["data"]["id"] for e in elems if "source" not in e["data"]}
        for e in elems:
            if "source" in e["data"]:
                assert e["data"]["source"] in node_ids
                assert e["data"]["target"] in node_ids


# ---------------------------------------------------------------------------
# Test: Truth table permutations
# ---------------------------------------------------------------------------

def test_truth_tables(model_pipeline):
    """Truth table returns None (too many vars) or a valid list of dicts."""
    _, _, _, bdd = model_pipeline
    var_names = list(bdd.encoding.keys())
    for label, node in [("init", bdd.init_bdd), ("reached", bdd.reached_bdd),
                        ("domain", bdd.domain_constraint)]:
        result = build_truth_table(node, bdd.bdd, bdd.encoding, var_names)
        if result is not None:
            assert isinstance(result, list)
            for row in result:
                assert isinstance(row, dict)
                for vn in var_names:
                    assert vn in row, f"truth table row missing var {vn} for {label}"


# ---------------------------------------------------------------------------
# Test: Reduction stats
# ---------------------------------------------------------------------------

def test_reduction_stats(model_pipeline):
    """Reduction stats compute without error for all BDD nodes."""
    _, _, _, bdd = model_pipeline
    for node in [bdd.init_bdd, bdd.reached_bdd, bdd.trans_bdd]:
        stats = get_reduction_stats(node, bdd.bdd, bdd.encoding)
        assert "full_tree_nodes" in stats
        assert "robdd_nodes" in stats
        assert "reduction_pct" in stats
        assert stats["reduction_pct"] >= 0


# ---------------------------------------------------------------------------
# Test: BDD summary
# ---------------------------------------------------------------------------

def test_bdd_summary(model_pipeline):
    """BDD summary computes without error."""
    _, _, _, bdd = model_pipeline
    summary = get_bdd_summary(bdd)
    assert "encoding" in summary
    assert "total_reachable" in summary
    assert "iterations" in summary
    assert summary["total_reachable"] > 0


# ---------------------------------------------------------------------------
# Test: State detail for all reachable states
# ---------------------------------------------------------------------------

def test_state_detail_all_reachable(model_pipeline):
    """get_state_detail returns data for every reachable node."""
    _, _, explicit, _ = model_pipeline
    elems = build_elements(explicit, reachable_only=True, max_nodes=500)
    nodes = [e for e in elems if "source" not in e["data"]]
    for node in nodes:
        detail = get_state_detail(explicit, node["data"]["id"])
        assert detail is not None, f"no detail for node {node['data']['id']}"
        assert "state" in detail
        assert "is_initial" in detail
        assert "is_reachable" in detail
        assert detail["is_reachable"]


# ---------------------------------------------------------------------------
# Test: Filter expressions
# ---------------------------------------------------------------------------

def test_filter_expressions(model_pipeline):
    """Filter expressions don't crash even with nonsense values."""
    _, _, explicit, _ = model_pipeline
    # Valid filter (pick first var and first value)
    if explicit.var_names:
        var = explicit.var_names[0]
        first_state = next(iter(explicit.reachable_states), None)
        if first_state is not None:
            sd = explicit.state_to_dict[first_state]
            val = sd[var]
            elems = build_elements(
                explicit, reachable_only=True, filter_expr=f"{var}={val}",
            )
            _validate_cytoscape_elements(elems)
            assert len(elems) > 0

    # Nonsense filter (should return empty, not crash)
    elems = build_elements(
        explicit, reachable_only=True, filter_expr="nonexistent=999",
    )
    assert isinstance(elems, list)


# ---------------------------------------------------------------------------
# Test: App creation still works
# ---------------------------------------------------------------------------

def test_app_creation():
    """create_app succeeds after all changes."""
    from smvis.app import create_app
    app = create_app()
    assert app is not None
    assert app.layout is not None
