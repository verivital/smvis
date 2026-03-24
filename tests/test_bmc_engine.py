"""Tests for the BMC engine (Z3-based bounded model checking)."""
from __future__ import annotations
import pytest
from smvis.smv_model import SpecDecl, TemporalUnary, BoolLit, VarRef, BinOp, IntLit
from smvis.bmc_engine import (
    BmcEncoder, run_bmc, run_bmc_all_specs, is_safety_spec, BmcResult,
    _has_temporal,
)
from smvis.explicit_engine import explore, evaluate


# ---------------------------------------------------------------------------
# is_safety_spec
# ---------------------------------------------------------------------------

class TestIsSafetySpec:
    def test_invarspec(self):
        spec = SpecDecl("INVARSPEC", BoolLit(True))
        assert is_safety_spec(spec)

    def test_ctlspec_ag(self):
        spec = SpecDecl("CTLSPEC", TemporalUnary("AG", BoolLit(True)))
        assert is_safety_spec(spec)

    def test_ctlspec_af_not_safety(self):
        spec = SpecDecl("CTLSPEC", TemporalUnary("AF", BoolLit(True)))
        assert not is_safety_spec(spec)

    def test_ltlspec_g(self):
        spec = SpecDecl("LTLSPEC", TemporalUnary("G", BoolLit(True)))
        assert is_safety_spec(spec)

    def test_ltlspec_f_not_safety(self):
        spec = SpecDecl("LTLSPEC", TemporalUnary("F", BoolLit(True)))
        assert not is_safety_spec(spec)

    def test_ltlspec_g_f_not_safety(self):
        """G F(p) is liveness, not safety."""
        inner = TemporalUnary("F", BoolLit(True))
        spec = SpecDecl("LTLSPEC", TemporalUnary("G", inner))
        assert not is_safety_spec(spec)

    def test_ctlspec_ag_af_not_safety(self):
        """AG AF p is liveness, not safety."""
        inner = TemporalUnary("AF", BoolLit(True))
        spec = SpecDecl("CTLSPEC", TemporalUnary("AG", inner))
        assert not is_safety_spec(spec)


class TestHasTemporal:
    def test_no_temporal(self):
        expr = BinOp("=", VarRef("x"), IntLit(5))
        assert not _has_temporal(expr)

    def test_temporal_unary(self):
        expr = TemporalUnary("F", BoolLit(True))
        assert _has_temporal(expr)


# ---------------------------------------------------------------------------
# BmcEncoder basics
# ---------------------------------------------------------------------------

class TestBmcEncoder:
    def test_bool_var_sort(self, counter_model):
        enc = BmcEncoder(counter_model)
        v = enc.get_var("press", 0)
        import z3
        assert v.sort() == z3.BoolSort()

    def test_range_var_sort(self, counter_model):
        enc = BmcEncoder(counter_model)
        v = enc.get_var("x", 0)
        import z3
        assert v.sort() == z3.IntSort()

    def test_enum_var_sort(self, mutex_model):
        enc = BmcEncoder(mutex_model)
        v = enc.get_var("process1", 0)
        import z3
        assert v.sort() == z3.IntSort()

    def test_different_steps_different_vars(self, counter_model):
        enc = BmcEncoder(counter_model)
        v0 = enc.get_var("x", 0)
        v1 = enc.get_var("x", 1)
        assert str(v0) != str(v1)

    def test_init_constraints_nonempty(self, counter_model):
        enc = BmcEncoder(counter_model)
        constraints = enc.encode_init()
        assert len(constraints) > 0

    def test_transition_constraints_nonempty(self, counter_model):
        enc = BmcEncoder(counter_model)
        enc.encode_init()  # allocate step-0 vars
        constraints = enc.encode_transition(0)
        assert len(constraints) > 0


# ---------------------------------------------------------------------------
# BMC on counter.smv
# ---------------------------------------------------------------------------

class TestBmcCounter:
    def test_x_leq_count_max_holds(self, counter_model):
        """INVARSPEC x <= count_max should hold."""
        spec = counter_model.specs[0]  # x <= count_max
        assert spec.kind == "INVARSPEC"
        result = run_bmc(counter_model, spec, max_k=15)
        assert not result.violated

    def test_x_lt_count_max_violated(self, counter_model):
        """INVARSPEC x < count_max should be violated (x reaches count_max)."""
        spec = counter_model.specs[3]  # x < count_max
        assert spec.kind == "INVARSPEC"
        result = run_bmc(counter_model, spec, max_k=15)
        assert result.violated
        assert result.violation_k is not None
        assert result.counterexample is not None

    def test_x_leq_half_violated(self, counter_model):
        """INVARSPEC x <= count_max/2 should be violated."""
        spec = counter_model.specs[4]  # x <= count_max / 2
        assert spec.kind == "INVARSPEC"
        result = run_bmc(counter_model, spec, max_k=15)
        assert result.violated
        assert result.violation_k <= 15

    def test_counterexample_length(self, counter_model):
        """Counterexample trace length should be violation_k + 1."""
        spec = counter_model.specs[3]  # x < count_max
        result = run_bmc(counter_model, spec, max_k=15)
        assert result.counterexample is not None
        assert len(result.counterexample) == result.violation_k + 1

    def test_counterexample_has_all_vars(self, counter_model):
        """Each state in counterexample should have all model variables."""
        spec = counter_model.specs[3]
        result = run_bmc(counter_model, spec, max_k=15)
        assert result.counterexample is not None
        for state in result.counterexample:
            for var_name in counter_model.variables:
                assert var_name in state

    def test_counterexample_initial_state_valid(self, counter_model):
        """Step 0 of counterexample should be a valid initial state."""
        spec = counter_model.specs[3]
        result = run_bmc(counter_model, spec, max_k=15)
        assert result.counterexample is not None
        s0 = result.counterexample[0]
        # Check init conditions
        for var_name, init_expr in counter_model.inits.items():
            expected = evaluate(init_expr, {}, None, counter_model.defines)
            assert s0[var_name] == expected, f"{var_name}: {s0[var_name]} != {expected}"

    def test_all_safety_specs(self, counter_model):
        """Run BMC on all safety specs."""
        results = run_bmc_all_specs(counter_model, max_k=15)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, BmcResult)
            assert len(r.step_results) > 0


# ---------------------------------------------------------------------------
# BMC on mutex.smv
# ---------------------------------------------------------------------------

class TestBmcMutex:
    def test_mutual_exclusion_holds(self, mutex_model):
        """INVARSPEC !(process1=critical & process2=critical) should hold."""
        spec = mutex_model.specs[0]
        assert spec.kind == "INVARSPEC"
        result = run_bmc(mutex_model, spec, max_k=20)
        assert not result.violated

    def test_valid_states_holds(self, mutex_model):
        """INVARSPEC process1=idle | process1=waiting | process1=critical should hold."""
        spec = mutex_model.specs[3]
        assert spec.kind == "INVARSPEC"
        result = run_bmc(mutex_model, spec, max_k=20)
        assert not result.violated

    def test_counterexample_has_enum_values(self, mutex_model):
        """Counterexample should have string enum values, not integer codes."""
        # Find a violated spec
        for spec in mutex_model.specs:
            if not is_safety_spec(spec):
                continue
            result = run_bmc(mutex_model, spec, max_k=20)
            if result.violated and result.counterexample:
                s0 = result.counterexample[0]
                # process1 should be a string like 'idle', not an int
                assert isinstance(s0["process1"], str)
                assert s0["process1"] in ("idle", "waiting", "critical")
                return
        pytest.skip("No violated safety spec found to test")


# ---------------------------------------------------------------------------
# BMC on traffic_light.smv
# ---------------------------------------------------------------------------

class TestBmcTrafficLight:
    def test_no_simultaneous_green_holds(self):
        """Both lights should never be green simultaneously."""
        from smvis.smv_parser import parse_smv_file
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "examples", "traffic_light.smv")
        model = parse_smv_file(path)
        # First INVARSPEC: !(main_light = green & side_light = green)
        spec = model.specs[0]
        assert spec.kind == "INVARSPEC"
        result = run_bmc(model, spec, max_k=15)
        assert not result.violated

    def test_all_safety_specs_hold(self):
        """All INVARSPEC in traffic_light should hold."""
        from smvis.smv_parser import parse_smv_file
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "examples", "traffic_light.smv")
        model = parse_smv_file(path)
        results = run_bmc_all_specs(model, max_k=15)
        for r in results:
            assert not r.violated, f"Unexpected violation: {r.spec_text}"


# ---------------------------------------------------------------------------
# Cross-validation: BMC vs explicit engine
# ---------------------------------------------------------------------------

class TestBmcMatchesExplicit:
    def test_counter_counterexample_is_valid_path(self, counter_model):
        """If BMC finds a violation, the counterexample should be
        a valid execution path in the explicit state graph."""
        explicit = explore(counter_model)
        spec = counter_model.specs[3]  # x < count_max (violated)
        result = run_bmc(counter_model, spec, max_k=15)
        assert result.violated
        trace = result.counterexample
        assert trace is not None

        var_names = explicit.var_names
        # Check each state in trace is reachable
        for state_dict in trace:
            state_key = tuple(state_dict[v] for v in var_names)
            assert state_key in explicit.reachable_states, (
                f"BMC trace state not in reachable set: {state_dict}"
            )
        # Check consecutive states are valid transitions
        transitions_set = set(explicit.transitions)
        for i in range(len(trace) - 1):
            src = tuple(trace[i][v] for v in var_names)
            dst = tuple(trace[i + 1][v] for v in var_names)
            assert (src, dst) in transitions_set, (
                f"BMC trace transition not valid: {trace[i]} -> {trace[i+1]}"
            )

    def test_mutex_counterexample_is_valid_path(self, mutex_model):
        """Validate mutex BMC counterexample against explicit engine."""
        explicit = explore(mutex_model)
        # Find a violated safety spec
        for spec in mutex_model.specs:
            if not is_safety_spec(spec):
                continue
            result = run_bmc(mutex_model, spec, max_k=20)
            if not result.violated:
                continue
            trace = result.counterexample
            assert trace is not None
            var_names = explicit.var_names
            for state_dict in trace:
                state_key = tuple(state_dict[v] for v in var_names)
                assert state_key in explicit.reachable_states
            transitions_set = set(explicit.transitions)
            for i in range(len(trace) - 1):
                src = tuple(trace[i][v] for v in var_names)
                dst = tuple(trace[i + 1][v] for v in var_names)
                assert (src, dst) in transitions_set
            return
        pytest.skip("No violated safety spec found")


# ---------------------------------------------------------------------------
# Non-safety specs should be skipped
# ---------------------------------------------------------------------------

class TestBmcSkipsNonSafety:
    def test_non_safety_returns_empty(self):
        """Non-safety specs should return a result with violated=False, no steps."""
        spec = SpecDecl("LTLSPEC", TemporalUnary("F", BoolLit(True)))
        from smvis.smv_model import SmvModel
        model = SmvModel()
        result = run_bmc(model, spec, max_k=5)
        assert not result.violated
        assert result.step_results == []
