"""Tests for cycle_analysis module: SCC decomposition and R(n) computation."""
from __future__ import annotations
import os
import pytest
from smvis.cycle_analysis import (
    CycleAnalysisResult,
    compute_sccs,
    analyze_cycles,
)
from smvis.explicit_engine import ExplicitResult, State, explore
from smvis.smv_parser import parse_smv_file
from smvis.graph_builder import build_elements, apply_repeatable_overlay

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)


# ======================== Hand-crafted graph helpers ========================

def _make_explicit(states, initial, transitions) -> ExplicitResult:
    """Build a minimal ExplicitResult from hand-crafted data."""
    reachable = set(states)
    var_names = [f"v{i}" for i in range(len(states[0]))] if states else []
    state_to_dict = {s: dict(zip(var_names, s)) for s in states}
    return ExplicitResult(
        var_names=var_names,
        total_states=len(states),
        initial_states=list(initial),
        reachable_states=reachable,
        transitions=transitions,
        bfs_layers=[set(initial)],
        state_to_dict=state_to_dict,
    )


# ======================== SCC Tests ========================

class TestSCC:
    def test_simple_cycle(self):
        """a->b->c->a + d->a: SCC {a,b,c} nontrivial, {d} trivial."""
        a, b, c, d = (0,), (1,), (2,), (3,)
        states = [a, b, c, d]
        transitions = [(a, b), (b, c), (c, a), (d, a)]
        er = _make_explicit(states, [d], transitions)
        result = analyze_cycles(er)

        # Should have 2 SCCs: {a,b,c} and {d}
        assert len(result.sccs) == 2
        assert len(result.nontrivial_sccs) == 1
        assert result.nontrivial_sccs[0] == frozenset({a, b, c})
        assert d in result.transient_states

    def test_self_loop_scc(self):
        """Single state with self-loop is nontrivial."""
        s = (0,)
        er = _make_explicit([s], [s], [(s, s)])
        result = analyze_cycles(er)

        assert len(result.nontrivial_sccs) == 1
        assert s in result.r_star
        assert len(result.transient_states) == 0

    def test_no_edges(self):
        """Isolated states: all trivial SCCs, all transient."""
        a, b = (0,), (1,)
        er = _make_explicit([a, b], [a], [])
        result = analyze_cycles(er)

        assert len(result.nontrivial_sccs) == 0
        assert result.r_star == set()
        assert result.transient_states == {a, b}

    def test_scc_partition(self):
        """SCCs partition all reachable states."""
        a, b, c, d = (0,), (1,), (2,), (3,)
        transitions = [(a, b), (b, a), (c, d), (d, c), (a, c)]
        er = _make_explicit([a, b, c, d], [a], transitions)
        result = analyze_cycles(er)

        all_in_sccs: set[State] = set()
        for scc in result.sccs:
            # No overlap between SCCs
            assert len(all_in_sccs & scc) == 0
            all_in_sccs |= scc
        assert all_in_sccs == er.reachable_states

    def test_two_separate_sccs(self):
        """Two disconnected cycles."""
        a, b, c, d = (0,), (1,), (2,), (3,)
        transitions = [(a, b), (b, a), (c, d), (d, c)]
        er = _make_explicit([a, b, c, d], [a, c], transitions)
        result = analyze_cycles(er)

        assert len(result.nontrivial_sccs) == 2
        assert result.r_star == {a, b, c, d}


class TestSCCOnModels:
    def test_counter_sccs(self, counter_explicit):
        """counter.smv: all 24 reachable states in one SCC."""
        result = analyze_cycles(counter_explicit)
        assert len(result.nontrivial_sccs) == 1
        assert result.r_star == counter_explicit.reachable_states
        assert len(result.transient_states) == 0

    def test_mutex_sccs(self, mutex_explicit):
        """mutex.smv: all 16 reachable states should be repeatable."""
        result = analyze_cycles(mutex_explicit)
        assert result.r_star == mutex_explicit.reachable_states
        assert len(result.transient_states) == 0


# ======================== R(n) Tests ========================

class TestRSets:
    def test_r1_equals_self_loops(self):
        """R(1) = states with self-loops."""
        a, b, c = (0,), (1,), (2,)
        transitions = [(a, a), (a, b), (b, c), (c, b)]
        er = _make_explicit([a, b, c], [a], transitions)
        result = analyze_cycles(er)

        assert result.r_sets.get(1, set()) == {a}
        assert result.min_return_time[a] == 1

    def test_pure_3_cycle(self):
        """a->b->c->a: R(1)=empty, R(2)=empty, R(3)={a,b,c}."""
        a, b, c = (0,), (1,), (2,)
        transitions = [(a, b), (b, c), (c, a)]
        er = _make_explicit([a, b, c], [a], transitions)
        result = analyze_cycles(er)

        assert result.r_sets.get(1, set()) == set()
        assert result.r_sets.get(2, set()) == set()
        assert result.r_sets.get(3, set()) == {a, b, c}
        assert result.convergence_step == 3
        for s in [a, b, c]:
            assert result.min_return_time[s] == 3

    def test_pure_2_cycle(self):
        """a->b->a: R(1)=empty, R(2)={a,b}."""
        a, b = (0,), (1,)
        transitions = [(a, b), (b, a)]
        er = _make_explicit([a, b], [a], transitions)
        result = analyze_cycles(er)

        assert result.r_sets.get(1, set()) == set()
        assert result.r_sets.get(2, set()) == {a, b}
        assert result.convergence_step == 2

    def test_self_loop_plus_cycle(self):
        """a->a, a->b, b->a: a has min return 1, b has min return 2."""
        a, b = (0,), (1,)
        transitions = [(a, a), (a, b), (b, a)]
        er = _make_explicit([a, b], [a], transitions)
        result = analyze_cycles(er)

        assert result.min_return_time[a] == 1
        assert result.min_return_time[b] == 2
        assert result.convergence_step == 2

    def test_r_star_equals_nontrivial_union(self):
        """R* = union of all nontrivial SCCs."""
        a, b, c, d = (0,), (1,), (2,), (3,)
        transitions = [(a, b), (b, a), (c, d)]
        er = _make_explicit([a, b, c, d], [a, c], transitions)
        result = analyze_cycles(er)

        nontrivial_union: set[State] = set()
        for scc in result.nontrivial_sccs:
            nontrivial_union |= scc
        assert result.r_star == nontrivial_union

    def test_transient_plus_repeatable(self):
        """R* union transient = reachable, and they don't overlap."""
        a, b, c, d = (0,), (1,), (2,), (3,)
        transitions = [(a, b), (b, c), (c, b), (a, d)]
        er = _make_explicit([a, b, c, d], [a], transitions)
        result = analyze_cycles(er)

        assert result.r_star | result.transient_states == er.reachable_states
        assert result.r_star & result.transient_states == set()

    def test_convergence_bounded(self):
        """convergence_step <= |reachable states|."""
        a, b, c = (0,), (1,), (2,)
        transitions = [(a, b), (b, c), (c, a)]
        er = _make_explicit([a, b, c], [a], transitions)
        result = analyze_cycles(er)

        assert result.convergence_step <= len(er.reachable_states)

    def test_cumulative_monotone(self):
        """cumulative_r[k] is subset of cumulative_r[k+1]."""
        a, b, c = (0,), (1,), (2,)
        transitions = [(a, a), (a, b), (b, c), (c, a)]
        er = _make_explicit([a, b, c], [a], transitions)
        result = analyze_cycles(er)

        keys = sorted(result.cumulative_r.keys())
        for i in range(len(keys) - 1):
            assert result.cumulative_r[keys[i]] <= result.cumulative_r[keys[i + 1]]

    def test_mixed_cycle_lengths(self):
        """Graph with both 2-cycles and 3-cycles: different min return times."""
        # a->b->a (2-cycle), c->d->e->c (3-cycle), a->c (connecting)
        a, b, c, d, e = (0,), (1,), (2,), (3,), (4,)
        transitions = [(a, b), (b, a), (c, d), (d, e), (e, c), (a, c)]
        er = _make_explicit([a, b, c, d, e], [a], transitions)
        result = analyze_cycles(er)

        assert result.min_return_time[a] == 2
        assert result.min_return_time[b] == 2
        assert result.min_return_time[c] == 3
        assert result.min_return_time[d] == 3
        assert result.min_return_time[e] == 3


class TestRSetsOnModels:
    def test_counter_r1(self, counter_explicit):
        """counter.smv: R(1) should contain only (off, F, 0) self-loop."""
        result = analyze_cycles(counter_explicit)
        # (off, F, 0) = ('off', False, 0) in our state representation
        r1 = result.r_sets.get(1, set())
        assert len(r1) == 1
        # The self-loop state: mode=off, press=False, x=0
        self_loop_state = next(iter(r1))
        sd = counter_explicit.state_to_dict[self_loop_state]
        assert sd["mode"] == "off"
        assert sd["press"] == False
        assert sd["x"] == 0

    def test_counter_convergence(self, counter_explicit):
        """counter.smv: convergence should be at step 12.

        The last states to converge are (on,F,10), (on,F,9), and (on,T,10),
        all at min return time 12. The full-loop path is:
        reset→off (1 step) + re-enter on (1 step) + count 0→10 (10 steps) = 12.
        """
        result = analyze_cycles(counter_explicit)
        assert result.convergence_step == 12

    def test_counter_all_repeatable(self, counter_explicit):
        """counter.smv: all 24 reachable states are repeatable."""
        result = analyze_cycles(counter_explicit)
        assert result.r_star == counter_explicit.reachable_states
        assert len(result.r_star) == 24

    def test_counter_min_return_times(self, counter_explicit):
        """counter.smv: verify min return times for key states."""
        result = analyze_cycles(counter_explicit)
        sd_map = counter_explicit.state_to_dict

        for state, sd in sd_map.items():
            if state not in counter_explicit.reachable_states:
                continue
            mrt = result.min_return_time.get(state)
            assert mrt is not None, f"State {sd} has no min return time"

            mode, press, x = sd["mode"], sd["press"], sd["x"]
            if mode == "off" and press == False and x == 0:
                assert mrt == 1, f"(off,F,0) should have min return 1, got {mrt}"
            elif mode == "off" and press == True and x == 0:
                assert mrt == 2, f"(off,T,0) should have min return 2, got {mrt}"
            elif mode == "on" and press == True:
                # (on, T, k) should have min return k+2
                expected = x + 2
                assert mrt == expected, (
                    f"(on,T,{x}) should have min return {expected}, got {mrt}"
                )
            elif mode == "on" and press == False:
                # (on, F, k): min return k+3 for k<10, 12 for k=10
                # k=10 can't go through (on,T,11) — goes directly to off
                if x < 10:
                    expected = x + 3
                else:
                    expected = 12  # 1 step to off + 1 to on + 10 counting = 12
                assert mrt == expected, (
                    f"(on,F,{x}) should have min return {expected}, got {mrt}"
                )

    def test_counter_cumulative_growth(self, counter_explicit):
        """counter.smv: cumulative grows by ~2 per step."""
        result = analyze_cycles(counter_explicit)
        prev_size = 0
        for k in range(1, result.convergence_step + 1):
            cur_size = len(result.cumulative_r[k])
            assert cur_size >= prev_size
            prev_size = cur_size
        assert prev_size == 24


# ======================== Cycle Edges Tests ========================

class TestCycleEdges:
    def test_cycle_edges_within_scc(self):
        """Cycle edges are edges within the same nontrivial SCC."""
        a, b, c, d = (0,), (1,), (2,), (3,)
        transitions = [(a, b), (b, a), (a, c), (c, d)]
        er = _make_explicit([a, b, c, d], [a], transitions)
        result = analyze_cycles(er)

        assert (a, b) in result.cycle_edges
        assert (b, a) in result.cycle_edges
        # a->c is cross-SCC, not a cycle edge
        assert (a, c) not in result.cycle_edges
        # c->d is between trivial SCCs
        assert (c, d) not in result.cycle_edges

    def test_self_loop_is_cycle_edge(self):
        """Self-loop edge should be a cycle edge."""
        a = (0,)
        er = _make_explicit([a], [a], [(a, a)])
        result = analyze_cycles(er)
        assert (a, a) in result.cycle_edges


# ======================== Empty / Edge Cases ========================

class TestEdgeCases:
    def test_empty_graph(self):
        """Empty graph: no states, no SCCs."""
        er = ExplicitResult(
            var_names=[],
            total_states=0,
            initial_states=[],
            reachable_states=set(),
            transitions=[],
            bfs_layers=[],
        )
        result = analyze_cycles(er)
        assert len(result.sccs) == 0
        assert len(result.r_star) == 0
        assert result.convergence_step == 0

    def test_single_state_no_edges(self):
        """Single state with no edges: trivial SCC, transient."""
        a = (0,)
        er = _make_explicit([a], [a], [])
        result = analyze_cycles(er)
        assert a in result.transient_states
        assert a not in result.r_star

    def test_linear_chain(self):
        """a->b->c: no cycles, all transient."""
        a, b, c = (0,), (1,), (2,)
        transitions = [(a, b), (b, c)]
        er = _make_explicit([a, b, c], [a], transitions)
        result = analyze_cycles(er)
        assert result.r_star == set()
        assert result.transient_states == {a, b, c}
        assert result.convergence_step == 0


# ======================== Regression Tests: All 9 Models ========================

_ALL_MODELS = [
    "counter.smv", "mutex.smv", "gcd_01.smv", "mult.smv",
    "bubble_sort3.smv", "traffic_light.smv", "swap.smv",
    "abs_diff.smv", "fibonacci.smv",
]

# ======================== Overlay Tests ========================

class TestOverlay:
    def test_overlay_r_star_classes(self, counter_explicit):
        """R* mode: every reachable node gets 'repeatable' or 'transient'."""
        result = analyze_cycles(counter_explicit)
        elements = build_elements(counter_explicit, reachable_only=True)
        elements = apply_repeatable_overlay(elements, result, mode="r_star")

        for elem in elements:
            if "source" not in elem["data"]:
                classes = elem.get("classes", "")
                assert "repeatable" in classes or "transient" in classes, (
                    f"Node {elem['data']['id']} missing repeatable/transient class"
                )

    def test_overlay_scc_classes(self, counter_explicit):
        """SCC mode: repeatable nodes get scc-N, transient nodes get transient."""
        result = analyze_cycles(counter_explicit)
        elements = build_elements(counter_explicit, reachable_only=True)
        elements = apply_repeatable_overlay(elements, result, mode="scc")

        for elem in elements:
            if "source" not in elem["data"]:
                classes = elem.get("classes", "")
                has_scc = any(f"scc-{i}" in classes for i in range(8))
                has_transient = "transient" in classes
                assert has_scc or has_transient, (
                    f"Node {elem['data']['id']} missing scc-N/transient class"
                )

    def test_overlay_r_n_slider(self, counter_explicit):
        """R(n) mode: slider=1 shows 1 state, slider=12 shows all 24."""
        result = analyze_cycles(counter_explicit)

        # Slider=1: only (off,F,0) colored
        elements = build_elements(counter_explicit, reachable_only=True)
        elements = apply_repeatable_overlay(elements, result, mode="r_n", step_n=1)
        colored = [e for e in elements if "source" not in e["data"]
                   and "repeat-" in e.get("classes", "")]
        assert len(colored) == 1

        # Slider=12: all 24 states colored
        elements = build_elements(counter_explicit, reachable_only=True)
        elements = apply_repeatable_overlay(elements, result, mode="r_n", step_n=12)
        colored = [e for e in elements if "source" not in e["data"]
                   and "repeat-" in e.get("classes", "")]
        assert len(colored) == 24

    def test_overlay_cycle_edges(self, counter_explicit):
        """R* mode: cycle edges get cycle-edge class."""
        result = analyze_cycles(counter_explicit)
        elements = build_elements(counter_explicit, reachable_only=True)
        elements = apply_repeatable_overlay(elements, result, mode="r_star")

        cycle_edge_count = sum(
            1 for e in elements
            if "source" in e.get("data", {}) and "cycle-edge" in e.get("classes", "")
        )
        assert cycle_edge_count > 0

    def test_overlay_preserves_initial_class(self, counter_explicit):
        """Overlay should not remove existing 'initial' class."""
        result = analyze_cycles(counter_explicit)
        elements = build_elements(counter_explicit, reachable_only=True)
        elements = apply_repeatable_overlay(elements, result, mode="r_star")

        initial_nodes = [e for e in elements if "source" not in e["data"]
                         and "initial" in e.get("classes", "")]
        assert len(initial_nodes) > 0


# ======================== Regression Tests: All 9 Models ========================

@pytest.mark.parametrize("model_name", _ALL_MODELS)
def test_cycle_analysis_invariants(model_name):
    """For every model: R* + transient = reachable, convergence bounded."""
    model_path = os.path.join(EXAMPLES_DIR, model_name)
    if not os.path.exists(model_path):
        pytest.skip(f"{model_name} not found")

    model = parse_smv_file(model_path)
    explicit = explore(model)
    result = analyze_cycles(explicit)

    # R* + transient = reachable
    assert result.r_star | result.transient_states == explicit.reachable_states
    # No overlap
    assert result.r_star & result.transient_states == set()
    # Convergence bounded
    if result.r_star:
        assert result.convergence_step <= len(explicit.reachable_states)
    # Every repeatable state has a min return time
    for s in result.r_star:
        assert s in result.min_return_time
    # Every transient state does NOT have a min return time
    for s in result.transient_states:
        assert s not in result.min_return_time
    # SCCs partition reachable states
    all_scc: set[State] = set()
    for scc in result.sccs:
        assert len(all_scc & scc) == 0
        all_scc |= scc
    assert all_scc == explicit.reachable_states


@pytest.mark.parametrize("model_name", _ALL_MODELS)
def test_cycle_edges_valid(model_name):
    """Every cycle edge connects states in the same nontrivial SCC."""
    model_path = os.path.join(EXAMPLES_DIR, model_name)
    if not os.path.exists(model_path):
        pytest.skip(f"{model_name} not found")

    model = parse_smv_file(model_path)
    explicit = explore(model)
    result = analyze_cycles(explicit)

    for src, dst in result.cycle_edges:
        assert src in result.r_star
        assert dst in result.r_star
        assert result.state_to_scc_id[src] == result.state_to_scc_id[dst]
