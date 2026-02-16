"""Tests for product_model and accepting_cycles modules.

Validates: composition structure, product exploration, accepting cycle detection,
lasso extraction, and cross-validation with known results.
"""
from __future__ import annotations
import os
import pytest
from smvis.smv_model import (
    SmvModel, VarDecl, BoolType, EnumType, RangeType,
    Expr, BinOp, UnaryOp, BoolLit, VarRef, IntLit,
    TemporalUnary, TemporalBinary, SpecDecl, CaseExpr, SetExpr,
    expr_to_str, get_domain,
)
from smvis.ltl_buchi import build_buchi_for_spec, BuchiAutomaton
from smvis.product_model import compose, ProductInfo, BUCHI_VAR, DEAD_STATE
from smvis.explicit_engine import explore, ExplicitResult
from smvis.cycle_analysis import analyze_cycles, CycleAnalysisResult
from smvis.accepting_cycles import (
    find_accepting_cycles, AcceptingCycleResult, extract_lasso, project_trace,
)
from smvis.smv_parser import parse_smv_file

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)


# ======================== Helpers ========================

def _spec(expr: Expr) -> SpecDecl:
    return SpecDecl(kind="LTLSPEC", expr=expr)


def _full_pipeline(model: SmvModel, spec: SpecDecl):
    """Run the full pipeline: build Buchi, compose, explore, analyze, find accepting."""
    buchi = build_buchi_for_spec(spec)
    pi = compose(model, buchi)
    result = explore(pi.product_model)
    cycles = analyze_cycles(result)
    acc = find_accepting_cycles(result, cycles, pi)
    return pi, result, cycles, acc


# ======================== Composition Structure ========================

class TestCompositionStructure:
    """Test that compose() produces correct product model structure."""

    def test_variables_include_buchi(self):
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        buchi = build_buchi_for_spec(spec)
        pi = compose(model, buchi)
        assert BUCHI_VAR in pi.product_model.variables
        assert set(model.variables.keys()) < set(pi.product_model.variables.keys())

    def test_buchi_var_is_enum(self):
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        buchi = build_buchi_for_spec(spec)
        pi = compose(model, buchi)
        vd = pi.product_model.variables[BUCHI_VAR]
        assert isinstance(vd.var_type, EnumType)
        assert "q0" in vd.var_type.values

    def test_init_buchi_state(self):
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        buchi = build_buchi_for_spec(spec)
        pi = compose(model, buchi)
        assert BUCHI_VAR in pi.product_model.inits

    def test_next_buchi_state(self):
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        buchi = build_buchi_for_spec(spec)
        pi = compose(model, buchi)
        assert BUCHI_VAR in pi.product_model.nexts

    def test_original_inits_preserved(self):
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        buchi = build_buchi_for_spec(spec)
        pi = compose(model, buchi)
        for name in model.inits:
            assert name in pi.product_model.inits

    def test_original_nexts_preserved(self):
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        buchi = build_buchi_for_spec(spec)
        pi = compose(model, buchi)
        for name in model.nexts:
            assert name in pi.product_model.nexts

    def test_ap_defines_added(self):
        """AP expressions should be added as DEFINEs."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        buchi = build_buchi_for_spec(spec)
        pi = compose(model, buchi)
        for ap_name in buchi.ap_exprs:
            assert ap_name in pi.product_model.defines

    def test_fairness_includes_acceptance(self):
        """Product fairness should include Buchi acceptance constraint."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        buchi = build_buchi_for_spec(spec)
        pi = compose(model, buchi)
        # Original mutex has 0 fairness, Buchi adds 1
        assert len(pi.product_model.fairness) == len(buchi.accepting)

    def test_counter_fairness_preserved(self):
        """Counter has FAIRNESS mode != off; product should include it."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        buchi = build_buchi_for_spec(spec)
        pi = compose(model, buchi)
        assert len(pi.product_model.fairness) == len(model.fairness) + len(buchi.accepting)

    def test_dead_state_present_when_needed(self):
        """G(p) Buchi has conditional-only self-loop; needs _dead state."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        # F(p1 = critical) → negated G(!crit) → 1-state Buchi with conditional self-loop
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]
        buchi = build_buchi_for_spec(spec)
        pi = compose(model, buchi)
        assert DEAD_STATE in pi.buchi_domain

    def test_no_dead_state_when_unnecessary(self):
        """F(p) Buchi has TRUE self-loops everywhere; no _dead needed."""
        p = VarRef("p")
        model = SmvModel()
        model.variables["v"] = VarDecl("v", BoolType())
        model.inits["v"] = BoolLit(True)
        model.nexts["v"] = UnaryOp("!", VarRef("v"))
        spec = _spec(TemporalUnary("G", p))  # neg -> F(!p), all TRUE self-loops
        buchi = build_buchi_for_spec(spec)
        pi = compose(model, buchi)
        assert DEAD_STATE not in pi.buchi_domain


# ======================== Product Exploration ========================

class TestProductExploration:
    """Test that product exploration produces correct state counts."""

    def test_mutex_2state_buchi(self):
        """mutex.smv × 2-state Buchi + _dead → up to 48 reachable."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        pi, result, _, _ = _full_pipeline(model, spec)
        # F(p & G(q)) pattern: q0, q1, _dead → 16 × 3 = 48 max
        assert 16 <= len(result.reachable_states) <= 48

    def test_mutex_1state_buchi(self):
        """mutex.smv × 1-state Buchi (+ _dead) → 16 live + some dead."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]  # F(p1=critical)
        pi, result, _, _ = _full_pipeline(model, spec)
        # 1-state Buchi + _dead = 2 states in domain → ≤ 32 product states
        assert len(result.reachable_states) <= 32

    def test_counter_product(self):
        """counter.smv × 2-state Buchi."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        pi, result, _, _ = _full_pipeline(model, spec)
        # 24 reachable × 2 Buchi + potentially some _dead
        assert len(result.reachable_states) > 0

    def test_product_initial_states(self):
        """Product should have correct initial states."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        pi, result, _, _ = _full_pipeline(model, spec)
        # All initial states should have _buchi_q = q0
        var_names = result.var_names
        buchi_idx = var_names.index(BUCHI_VAR)
        for s in result.initial_states:
            assert s[buchi_idx] == "q0"


# ======================== Accepting Cycle Detection ========================

class TestAcceptingCycles:
    """Cross-validate property results with known correct answers."""

    # --- mutex.smv (no fairness) ---

    def test_mutex_response_holds(self):
        """G(p1=waiting -> F(p1=critical)) should HOLD (Peterson guarantees progress)."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.property_holds is True

    def test_mutex_response_2_holds(self):
        """G(p2=waiting -> F(p2=critical)) should HOLD."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][1]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.property_holds is True

    def test_mutex_eventually_critical_violated(self):
        """F(p1=critical) should be VIOLATED (p1 might never enter waiting)."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.property_holds is False
        assert acc.has_accepting_cycle is True

    # --- counter.smv (FAIRNESS mode != off) ---

    def test_counter_g_x0_violated(self):
        """G(x=0) should be VIOLATED (counter counts up)."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.property_holds is False

    def test_counter_gf_off_x0_holds(self):
        """GF(mode=off & x=0) should HOLD (counter always returns to off/0)."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][5]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.property_holds is True

    def test_counter_fg_off_x0_violated(self):
        """FG(mode=off & x=0) should be VIOLATED (counter keeps cycling)."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][6]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.property_holds is False

    def test_counter_g_x_bounded_holds(self):
        """G(x <= count_max) should HOLD (counter bounded)."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][15]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.property_holds is True

    def test_counter_f_x_exceeds_violated(self):
        """F(x > count_max) should be VIOLATED (x never exceeds count_max)."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][16]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.property_holds is False

    def test_counter_gf_conjunction_holds(self):
        """GF(mode=off) & GF(x=0) should HOLD."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][9]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.property_holds is True

    def test_counter_fg_on_violated(self):
        """FG(mode=on) should be VIOLATED (mode cycles)."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][12]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.property_holds is False

    # --- swap.smv ---

    def test_swap_conditional_holds(self):
        """(a=3 & b=1) -> F(a=1 & b=3) should HOLD (swap completes)."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "swap.smv"))
        ltl = [s for s in model.specs if s.kind == "LTLSPEC"]
        _, _, _, acc = _full_pipeline(model, ltl[0])
        assert acc.property_holds is True

    def test_swap_persistence_holds(self):
        """G(pc=done -> G(pc=done)) should HOLD (done is absorbing)."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "swap.smv"))
        ltl = [s for s in model.specs if s.kind == "LTLSPEC"]
        _, _, _, acc = _full_pipeline(model, ltl[2])
        assert acc.property_holds is True


# ======================== Lasso Extraction ========================

class TestLassoExtraction:
    """Test lasso counterexample extraction."""

    def test_lasso_exists_when_violated(self):
        """VIOLATED properties should have a lasso."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.lasso is not None
        prefix, cycle = acc.lasso
        assert len(prefix) >= 1
        assert len(cycle) >= 2  # at least entry + return

    def test_lasso_none_when_holds(self):
        """Properties that HOLD should have no lasso."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][0]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.lasso is None

    def test_lasso_prefix_starts_at_initial(self):
        """Lasso prefix should start from an initial state."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]  # G(x=0) → violated
        _, result, _, acc = _full_pipeline(model, spec)
        assert acc.lasso is not None
        prefix, _ = acc.lasso
        assert prefix[0] in result.initial_states

    def test_lasso_cycle_in_accepting_scc(self):
        """Cycle states should be within an accepting SCC."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]
        _, _, _, acc = _full_pipeline(model, spec)
        assert acc.lasso is not None
        _, cycle = acc.lasso
        scc = acc.accepting_sccs[0]
        # All cycle states should be in the SCC
        for s in cycle:
            assert s in scc

    def test_lasso_prefix_connects_to_cycle(self):
        """Last prefix state should equal first cycle state."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]
        _, _, _, acc = _full_pipeline(model, spec)
        prefix, cycle = acc.lasso
        assert prefix[-1] == cycle[0]


# ======================== Trace Projection ========================

class TestProjection:
    def test_project_removes_buchi(self):
        """Projected trace should have only original model variables."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]
        pi, _, _, acc = _full_pipeline(model, spec)
        assert acc.lasso is not None
        prefix_dicts, cycle_dicts = project_trace(acc.lasso, pi)
        for d in prefix_dicts + cycle_dicts:
            assert BUCHI_VAR not in d
            assert DEAD_STATE not in d
            # Should have all original variable names
            for v in pi.original_var_names:
                assert v in d

    def test_project_preserves_values(self):
        """Projected values should match original model variable values."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]
        pi, result, _, acc = _full_pipeline(model, spec)
        assert acc.lasso is not None
        prefix_dicts, cycle_dicts = project_trace(acc.lasso, pi)
        var_names = list(pi.product_model.variables.keys())
        # Check first prefix state
        prefix, _ = acc.lasso
        full_dict = dict(zip(var_names, prefix[0]))
        proj_dict = prefix_dicts[0]
        for v in pi.original_var_names:
            assert proj_dict[v] == full_dict[v]


# ======================== Accepting State Identification ========================

class TestAcceptingStates:
    def test_accepting_states_correct(self):
        """Accepting product states should have _buchi_q in accepting set."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]
        pi, result, _, acc = _full_pipeline(model, spec)
        var_names = result.var_names
        buchi_idx = var_names.index(BUCHI_VAR)
        for s in acc.accepting_product_states:
            assert s[buchi_idx] in pi.accepting_states

    def test_no_accepting_states_with_dead(self):
        """_dead state is never accepting."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        spec = [s for s in model.specs if s.kind == "LTLSPEC"][2]
        pi, result, _, acc = _full_pipeline(model, spec)
        var_names = result.var_names
        buchi_idx = var_names.index(BUCHI_VAR)
        for s in acc.accepting_product_states:
            assert s[buchi_idx] != DEAD_STATE


# ======================== Full Model Battery ========================

class TestFullModelBattery:
    """Run all LTLSPEC across multiple models and verify no crashes."""

    @pytest.mark.parametrize("fname", [
        "counter.smv", "mutex.smv", "swap.smv", "mult.smv",
        "bubble_sort3.smv", "fibonacci.smv", "gcd_01.smv", "abs_diff.smv",
    ])
    def test_all_specs_run(self, fname):
        """All LTLSPEC in model should complete without errors."""
        path = os.path.join(EXAMPLES_DIR, fname)
        if not os.path.exists(path):
            pytest.skip(f"{fname} not found")
        model = parse_smv_file(path)
        ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
        for spec in ltl_specs:
            _, _, _, acc = _full_pipeline(model, spec)
            assert isinstance(acc.property_holds, bool)
            if acc.has_accepting_cycle:
                assert acc.lasso is not None
            else:
                assert acc.lasso is None
