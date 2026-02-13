"""Tests for nuXmv semantic edge cases."""


class TestUnconstrainedInit:
    def test_gcd_a_b_unconstrained(self, gcd_model, gcd_explicit):
        """a and b have no init() -> all 11*11=121 combos with pc=l1."""
        assert "a" not in gcd_model.inits
        assert "b" not in gcd_model.inits
        assert len(gcd_explicit.initial_states) == 121

    def test_mult_a_b_unconstrained(self, mult_model, mult_explicit):
        """a and b have no init() -> 11*11=121 initial states."""
        assert "a" not in mult_model.inits
        assert "b" not in mult_model.inits
        assert len(mult_explicit.initial_states) == 121


class TestUnconstrainedNext:
    def test_counter_press_no_next(self, counter_model):
        """press has no next() -> non-deterministic over full domain."""
        assert "press" not in counter_model.nexts

    def test_press_causes_branching(self, counter_explicit):
        """Each reachable state has at least 2 successors (press=T/F)."""
        from collections import Counter
        src_counts = Counter(src for src, _ in counter_explicit.transitions)
        for sk in counter_explicit.reachable_states:
            assert src_counts[sk] >= 2, f"State {sk} has fewer than 2 successors"


class TestNextRefDependencies:
    def test_mutex_topological_order(self, mutex_model):
        """process1/process2 must precede flag1/flag2 in dep order."""
        from smvis.explicit_engine import compute_dep_order
        order = compute_dep_order(mutex_model)
        assert order.index("process1") < order.index("flag1")
        assert order.index("process2") < order.index("flag2")
