"""Tests for graph element generation."""
from smvis.graph_builder import build_elements, get_state_detail, compute_concentric_positions


class TestBuildElements:
    def test_counter_node_count(self, counter_explicit):
        elements = build_elements(counter_explicit, reachable_only=True)
        nodes = [e for e in elements if "source" not in e["data"]]
        assert len(nodes) == 24

    def test_counter_has_edges(self, counter_explicit):
        elements = build_elements(counter_explicit, reachable_only=True)
        edges = [e for e in elements if "source" in e["data"]]
        assert len(edges) > 0

    def test_initial_nodes_have_class(self, counter_explicit):
        elements = build_elements(counter_explicit, reachable_only=True)
        initial = [e for e in elements
                   if "source" not in e["data"] and "initial" in e.get("classes", "")]
        assert len(initial) == 2

    def test_max_nodes_limit(self, gcd_explicit):
        elements = build_elements(gcd_explicit, reachable_only=True, max_nodes=10)
        nodes = [e for e in elements if "source" not in e["data"]]
        # max_nodes + forced initial states
        assert len(nodes) <= 10 + len(gcd_explicit.initial_states)


class TestGetStateDetail:
    def test_predecessor_fix(self, counter_explicit):
        """Regression: predecessors should use src, not dst."""
        # Find a state that is a successor of an initial state
        init_key = counter_explicit.initial_states[0]
        succs = {dst for src, dst in counter_explicit.transitions if src == init_key}
        if succs:
            succ_key = next(iter(succs))
            from smvis.graph_builder import _state_id
            detail = get_state_detail(counter_explicit, _state_id(succ_key))
            if detail:
                assert detail["predecessor_count"] >= 1


class TestConcentricPositions:
    def test_adds_positions(self, counter_explicit):
        elements = build_elements(counter_explicit, reachable_only=True)
        result = compute_concentric_positions(elements)
        nodes = [e for e in result if "source" not in e.get("data", {})]
        for n in nodes:
            assert "position" in n
            assert "x" in n["position"]
            assert "y" in n["position"]
