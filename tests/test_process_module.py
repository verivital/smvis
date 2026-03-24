"""Tests for multi-module NuSMV models with process instantiation."""
from __future__ import annotations
import os
import pytest
from smvis.smv_parser import parse_smv, parse_smv_file
from smvis.smv_model import (
    SmvModel, VarDecl, BoolType, EnumType, RangeType,
    BinOp, UnaryOp, VarRef, IntLit, BoolLit, CaseExpr, SetExpr,
)
from smvis.explicit_engine import explore

EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "examples"
)

SEMAPHORE_PATH = os.path.join(EXAMPLES_DIR, "semaphore.smv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def semaphore_model():
    return parse_smv_file(SEMAPHORE_PATH)


@pytest.fixture(scope="module")
def semaphore_result(semaphore_model):
    return explore(semaphore_model)


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestParsing:
    """Test that multi-module models parse correctly."""

    def test_parse_semaphore_file(self, semaphore_model):
        assert isinstance(semaphore_model, SmvModel)

    def test_variables_present(self, semaphore_model):
        expected = {"semaphore", "_pid", "proc1.state", "proc2.state"}
        assert set(semaphore_model.variables.keys()) == expected

    def test_semaphore_type(self, semaphore_model):
        assert isinstance(semaphore_model.variables["semaphore"].var_type, BoolType)

    def test_pid_type(self, semaphore_model):
        vt = semaphore_model.variables["_pid"].var_type
        assert isinstance(vt, RangeType)
        assert vt.lo == 0
        assert vt.hi == 1

    def test_proc_state_type(self, semaphore_model):
        for name in ["proc1.state", "proc2.state"]:
            vt = semaphore_model.variables[name].var_type
            assert isinstance(vt, EnumType)
            assert set(vt.values) == {"idle", "entering", "critical", "exiting"}

    def test_init_assignments(self, semaphore_model):
        assert "semaphore" in semaphore_model.inits
        assert "proc1.state" in semaphore_model.inits
        assert "proc2.state" in semaphore_model.inits
        # _pid has no init (non-deterministic)
        assert "_pid" not in semaphore_model.inits

    def test_next_assignments(self, semaphore_model):
        assert "proc1.state" in semaphore_model.nexts
        assert "proc2.state" in semaphore_model.nexts
        assert "semaphore" in semaphore_model.nexts
        # _pid has no next (non-deterministic)
        assert "_pid" not in semaphore_model.nexts

    def test_fairness_constraints(self, semaphore_model):
        assert len(semaphore_model.fairness) == 2
        # FAIRNESS running → _pid = 0 and _pid = 1
        for i, f in enumerate(semaphore_model.fairness):
            assert isinstance(f, BinOp)
            assert f.op == "="
            assert isinstance(f.left, VarRef) and f.left.name == "_pid"
            assert isinstance(f.right, IntLit) and f.right.value == i

    def test_specs_preserved(self, semaphore_model):
        assert len(semaphore_model.specs) == 2
        assert all(s.kind == "LTLSPEC" for s in semaphore_model.specs)

    def test_next_state_guarded_by_pid(self, semaphore_model):
        """Next assignments should be CaseExpr with _pid guards."""
        for name in ["proc1.state", "proc2.state", "semaphore"]:
            nexpr = semaphore_model.nexts[name]
            assert isinstance(nexpr, CaseExpr), (
                f"next({name}) should be CaseExpr, got {type(nexpr).__name__}"
            )

    def test_stutter_fallback(self, semaphore_model):
        """Each next() CaseExpr should end with TRUE : var (stutter)."""
        for name in ["proc1.state", "proc2.state", "semaphore"]:
            nexpr = semaphore_model.nexts[name]
            last_cond, last_val = nexpr.branches[-1]
            assert isinstance(last_cond, BoolLit) and last_cond.value is True
            assert isinstance(last_val, VarRef) and last_val.name == name


# ---------------------------------------------------------------------------
# Existing models - no regressions
# ---------------------------------------------------------------------------

class TestNoRegressions:
    """Verify all existing single-module models still parse identically."""

    @pytest.mark.parametrize("filename", [
        "counter.smv", "mutex.smv", "gcd_01.smv", "mult.smv",
        "traffic_light.smv", "swap.smv", "fibonacci.smv",
        "two_bit_counter.smv", "abs_diff.smv", "request_grant.smv",
        "bubble_sort3.smv",
    ])
    def test_existing_model_parses(self, filename):
        path = os.path.join(EXAMPLES_DIR, filename)
        if not os.path.exists(path):
            pytest.skip(f"{filename} not found")
        model = parse_smv_file(path)
        assert isinstance(model, SmvModel)
        assert len(model.variables) > 0

    def test_counter_stats_unchanged(self):
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        assert len(model.variables) == 3
        result = explore(model)
        assert result.total_states == 104
        assert len(result.reachable_states) == 24
        assert len(result.initial_states) == 2

    def test_mutex_stats_unchanged(self):
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        assert len(model.variables) == 5
        result = explore(model)
        assert result.total_states == 72
        assert len(result.reachable_states) == 16


# ---------------------------------------------------------------------------
# Explicit exploration tests
# ---------------------------------------------------------------------------

class TestExploration:
    """Test explicit state exploration of the flattened semaphore model."""

    def test_total_states(self, semaphore_result):
        # 2 (semaphore) × 2 (_pid) × 4 (proc1.state) × 4 (proc2.state) = 64
        assert semaphore_result.total_states == 64

    def test_reachable_states(self, semaphore_result):
        assert len(semaphore_result.reachable_states) == 24

    def test_initial_states(self, semaphore_result):
        # semaphore=F, proc1=idle, proc2=idle, _pid∈{0,1} → 2 states
        assert len(semaphore_result.initial_states) == 2

    def test_transitions_count(self, semaphore_result):
        assert len(semaphore_result.transitions) == 72

    def test_mutual_exclusion_holds(self, semaphore_model, semaphore_result):
        """No reachable state has both processes in critical section."""
        var_names = semaphore_result.var_names
        p1_idx = var_names.index("proc1.state")
        p2_idx = var_names.index("proc2.state")
        for s in semaphore_result.reachable_states:
            assert not (s[p1_idx] == "critical" and s[p2_idx] == "critical"), (
                f"Mutual exclusion violated in state {dict(zip(var_names, s))}"
            )

    def test_critical_section_reachable(self, semaphore_result):
        """At least one process can reach the critical section."""
        var_names = semaphore_result.var_names
        p1_idx = var_names.index("proc1.state")
        p2_idx = var_names.index("proc2.state")
        has_p1_critical = any(s[p1_idx] == "critical" for s in semaphore_result.reachable_states)
        has_p2_critical = any(s[p2_idx] == "critical" for s in semaphore_result.reachable_states)
        assert has_p1_critical, "proc1 never reaches critical"
        assert has_p2_critical, "proc2 never reaches critical"

    def test_all_process_states_reachable(self, semaphore_result):
        """All four states (idle, entering, critical, exiting) are reachable for each process."""
        var_names = semaphore_result.var_names
        for proc in ["proc1.state", "proc2.state"]:
            idx = var_names.index(proc)
            reached = {s[idx] for s in semaphore_result.reachable_states}
            assert reached == {"idle", "entering", "critical", "exiting"}, (
                f"{proc} only reached states: {reached}"
            )


# ---------------------------------------------------------------------------
# BDD engine tests
# ---------------------------------------------------------------------------

class TestBDD:
    """Test BDD analysis of the flattened semaphore model."""

    def test_bdd_build(self, semaphore_model, semaphore_result):
        from smvis.bdd_engine import build_from_explicit
        r = semaphore_result
        bdd_result = build_from_explicit(
            semaphore_model,
            r.initial_states,
            r.transitions,
            r.var_names,
            r.state_to_dict,
        )
        assert bdd_result.total_reachable == 24


# ---------------------------------------------------------------------------
# BMC engine tests
# ---------------------------------------------------------------------------

class TestBMC:
    """Test BMC on the flattened semaphore model."""

    def test_bmc_mutual_exclusion_holds(self, semaphore_model):
        from smvis.bmc_engine import run_bmc_all_specs
        results = run_bmc_all_specs(semaphore_model, max_k=15)
        # Only the safety spec (G !(...)) should be checked
        safety_results = [r for r in results if not r.violated or r.violated]
        assert len(safety_results) >= 1
        # The mutual exclusion spec should hold
        mutex_result = safety_results[0]
        assert not mutex_result.violated, (
            f"Mutual exclusion violated at k={mutex_result.violation_k}"
        )

    def test_bmc_has_formulas(self, semaphore_model):
        from smvis.bmc_engine import run_bmc_all_specs
        results = run_bmc_all_specs(semaphore_model, max_k=5)
        for r in results:
            assert r.formulas is not None
            assert r.formulas.encoding
            assert r.formulas.init
            assert r.formulas.smt2


# ---------------------------------------------------------------------------
# Inline model parsing tests
# ---------------------------------------------------------------------------

class TestInlineParsing:
    """Test parsing of multi-module models from string."""

    def test_minimal_process_model(self):
        text = """
MODULE main
VAR
  x : boolean;
  p : process toggler(x);
ASSIGN
  init(x) := FALSE;

MODULE toggler(v)
VAR
  dummy : boolean;
ASSIGN
  init(dummy) := FALSE;
  next(v) := !v;
FAIRNESS running
"""
        model = parse_smv(text)
        assert "x" in model.variables
        assert "p.dummy" in model.variables
        assert "_pid" in model.variables
        assert "x" in model.nexts

    def test_multiple_process_instances(self):
        text = """
MODULE main
VAR
  count : 0..10;
  a : process incr(count);
  b : process incr(count);
  c : process incr(count);
ASSIGN
  init(count) := 0;

MODULE incr(c)
ASSIGN
  next(c) := case c < 10 : c + 1; TRUE : c; esac;
FAIRNESS running
"""
        model = parse_smv(text)
        assert "_pid" in model.variables
        vt = model.variables["_pid"].var_type
        assert isinstance(vt, RangeType)
        assert vt.hi == 2  # 3 processes: 0, 1, 2
        assert len(model.fairness) == 3

    def test_dotted_ident_in_spec(self):
        """Specs referencing proc.var should parse."""
        text = """
MODULE main
VAR
  dummy : boolean;
  p : process worker(dummy);
LTLSPEC G (p.done = TRUE)

MODULE worker(d)
VAR
  done : boolean;
ASSIGN
  init(done) := FALSE;
  next(done) := TRUE;
"""
        model = parse_smv(text)
        assert len(model.specs) == 1
        assert model.specs[0].kind == "LTLSPEC"

    def test_single_module_unchanged(self):
        """Single-module models still work exactly as before."""
        text = """
MODULE main
VAR
  x : boolean;
ASSIGN
  init(x) := TRUE;
  next(x) := !x;
"""
        model = parse_smv(text)
        assert set(model.variables.keys()) == {"x"}
        assert "_pid" not in model.variables


# ---------------------------------------------------------------------------
# Flattener edge cases
# ---------------------------------------------------------------------------

class TestFlattenerEdgeCases:
    """Test edge cases in module flattening."""

    def test_param_substitution_in_conditions(self):
        """Formal params used in case conditions are correctly substituted."""
        text = """
MODULE main
VAR
  flag : boolean;
  p : process checker(flag);
ASSIGN
  init(flag) := FALSE;

MODULE checker(f)
VAR
  result : boolean;
ASSIGN
  init(result) := FALSE;
  next(result) := case f : TRUE; TRUE : FALSE; esac;
FAIRNESS running
"""
        model = parse_smv(text)
        # next(p.result) should reference 'flag' (the actual arg), not 'f'
        nexpr = model.nexts["p.result"]
        assert isinstance(nexpr, CaseExpr)
        # The case branches should reference 'flag' via pid guard
        all_names = set()
        for cond, val in nexpr.branches:
            all_names |= _flatten_expr_names(cond)
        assert "flag" in all_names, (
            f"Parameter 'f' was not substituted to 'flag'. "
            f"Found names: {all_names}"
        )
        assert "f" not in all_names, "Formal param 'f' should not appear"

    def test_shared_variable_merged(self):
        """Multiple processes writing to same variable merge correctly."""
        text = """
MODULE main
VAR
  shared : boolean;
  p1 : process writer(shared);
  p2 : process writer(shared);
ASSIGN
  init(shared) := FALSE;

MODULE writer(s)
ASSIGN
  next(s) := !s;
FAIRNESS running
"""
        model = parse_smv(text)
        assert "shared" in model.nexts
        nexpr = model.nexts["shared"]
        assert isinstance(nexpr, CaseExpr)
        # Should have branches for both _pid=0 and _pid=1 plus stutter
        assert len(nexpr.branches) >= 3

    def test_undefined_module_raises(self):
        text = """
MODULE main
VAR
  p : process nonexistent(x);
"""
        with pytest.raises(Exception, match="undefined module"):
            parse_smv(text)

    def test_wrong_param_count_raises(self):
        text = """
MODULE main
VAR
  p : process foo(a, b);

MODULE foo(x)
VAR
  v : boolean;
"""
        with pytest.raises(Exception, match="expects 1 parameters"):
            parse_smv(text)


def _flatten_expr_names(expr) -> set[str]:
    """Collect all variable names referenced in an expression."""
    names = set()
    if isinstance(expr, VarRef):
        names.add(expr.name)
    elif isinstance(expr, BinOp):
        names |= _flatten_expr_names(expr.left)
        names |= _flatten_expr_names(expr.right)
    elif isinstance(expr, UnaryOp):
        names |= _flatten_expr_names(expr.operand)
    elif isinstance(expr, CaseExpr):
        for c, v in expr.branches:
            names |= _flatten_expr_names(c)
            names |= _flatten_expr_names(v)
    return names
