"""Tests for ltl_buchi module: LTL negation, pattern matching, Buchi construction."""
from __future__ import annotations
import os
import pytest
from smvis.smv_model import (
    Expr, BinOp, UnaryOp, BoolLit, VarRef, IntLit,
    TemporalUnary, TemporalBinary, SpecDecl, expr_to_str,
)
from smvis.ltl_buchi import (
    negate_ltl, simplify_ltl, build_buchi_for_spec, ltl_to_buchi,
    BuchiAutomaton, BuchiState, BuchiTransition, UnsupportedLTLPattern,
    parse_hoa, _has_temporal, _expr_eq,
)
from smvis.smv_parser import parse_smv_file

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)


# ======================== Helpers ========================

def _spec(expr: Expr) -> SpecDecl:
    return SpecDecl(kind="LTLSPEC", expr=expr)


def _var(name: str) -> VarRef:
    return VarRef(name)


p, q, r = _var("p"), _var("q"), _var("r")


# ======================== LTL Negation Tests ========================

class TestNegation:
    def test_neg_var(self):
        """!p = !p"""
        result = negate_ltl(p)
        assert isinstance(result, UnaryOp) and result.op == "!"
        assert isinstance(result.operand, VarRef) and result.operand.name == "p"

    def test_neg_bool_true(self):
        assert _expr_eq(negate_ltl(BoolLit(True)), BoolLit(False))

    def test_neg_bool_false(self):
        assert _expr_eq(negate_ltl(BoolLit(False)), BoolLit(True))

    def test_double_negation(self):
        """!!p = p"""
        result = negate_ltl(UnaryOp("!", p))
        assert isinstance(result, VarRef) and result.name == "p"

    def test_neg_and(self):
        """!(p & q) = !p | !q"""
        result = negate_ltl(BinOp("&", p, q))
        assert isinstance(result, BinOp) and result.op == "|"

    def test_neg_or(self):
        """!(p | q) = !p & !q"""
        result = negate_ltl(BinOp("|", p, q))
        assert isinstance(result, BinOp) and result.op == "&"

    def test_neg_implication(self):
        """!(p -> q) = p & !q"""
        result = negate_ltl(BinOp("->", p, q))
        assert isinstance(result, BinOp) and result.op == "&"
        assert isinstance(result.left, VarRef) and result.left.name == "p"

    def test_neg_G(self):
        """!G p = F !p"""
        result = negate_ltl(TemporalUnary("G", p))
        assert isinstance(result, TemporalUnary) and result.op == "F"
        assert isinstance(result.operand, UnaryOp) and result.operand.op == "!"

    def test_neg_F(self):
        """!F p = G !p"""
        result = negate_ltl(TemporalUnary("F", p))
        assert isinstance(result, TemporalUnary) and result.op == "G"

    def test_neg_X(self):
        """!X p = X !p"""
        result = negate_ltl(TemporalUnary("X", p))
        assert isinstance(result, TemporalUnary) and result.op == "X"
        assert isinstance(result.operand, UnaryOp) and result.operand.op == "!"

    def test_neg_GF(self):
        """!GF p = FG !p"""
        result = negate_ltl(TemporalUnary("G", TemporalUnary("F", p)))
        assert isinstance(result, TemporalUnary) and result.op == "F"
        assert isinstance(result.operand, TemporalUnary) and result.operand.op == "G"

    def test_neg_FG(self):
        """!FG p = GF !p"""
        result = negate_ltl(TemporalUnary("F", TemporalUnary("G", p)))
        assert isinstance(result, TemporalUnary) and result.op == "G"
        assert isinstance(result.operand, TemporalUnary) and result.operand.op == "F"

    def test_neg_response(self):
        """!G(p -> F q) = F(p & G !q)"""
        expr = TemporalUnary("G", BinOp("->", p, TemporalUnary("F", q)))
        result = negate_ltl(expr)
        # F(p & G(!q))
        assert isinstance(result, TemporalUnary) and result.op == "F"
        inner = result.operand
        assert isinstance(inner, BinOp) and inner.op == "&"

    def test_neg_persistence(self):
        """!G(p -> G q) = F(p & F !q)"""
        expr = TemporalUnary("G", BinOp("->", p, TemporalUnary("G", q)))
        result = negate_ltl(expr)
        # F(p & F(!q))
        assert isinstance(result, TemporalUnary) and result.op == "F"
        inner = result.operand
        assert isinstance(inner, BinOp) and inner.op == "&"

    def test_neg_conditional_eventually(self):
        """!(p -> F q) = p & G !q"""
        expr = BinOp("->", p, TemporalUnary("F", q))
        result = negate_ltl(expr)
        # p & G(!q)
        assert isinstance(result, BinOp) and result.op == "&"
        assert isinstance(result.left, VarRef) and result.left.name == "p"
        assert isinstance(result.right, TemporalUnary) and result.right.op == "G"

    def test_neg_comparison(self):
        """!(x = 5) = !(x = 5)"""
        expr = BinOp("=", _var("x"), IntLit(5))
        result = negate_ltl(expr)
        assert isinstance(result, UnaryOp) and result.op == "!"


# ======================== Simplification Tests ========================

class TestSimplification:
    def test_double_neg_elimination(self):
        result = simplify_ltl(UnaryOp("!", UnaryOp("!", p)))
        assert isinstance(result, VarRef) and result.name == "p"

    def test_and_true(self):
        result = simplify_ltl(BinOp("&", BoolLit(True), p))
        assert isinstance(result, VarRef) and result.name == "p"

    def test_and_false(self):
        result = simplify_ltl(BinOp("&", p, BoolLit(False)))
        assert isinstance(result, BoolLit) and result.value is False

    def test_or_true(self):
        result = simplify_ltl(BinOp("|", BoolLit(True), p))
        assert isinstance(result, BoolLit) and result.value is True

    def test_or_false(self):
        result = simplify_ltl(BinOp("|", BoolLit(False), p))
        assert isinstance(result, VarRef) and result.name == "p"

    def test_impl_false_antecedent(self):
        result = simplify_ltl(BinOp("->", BoolLit(False), p))
        assert isinstance(result, BoolLit) and result.value is True

    def test_impl_true_antecedent(self):
        result = simplify_ltl(BinOp("->", BoolLit(True), p))
        assert isinstance(result, VarRef) and result.name == "p"

    def test_simplify_temporal(self):
        """simplify G(!!p) = G(p)"""
        result = simplify_ltl(TemporalUnary("G", UnaryOp("!", UnaryOp("!", p))))
        assert isinstance(result, TemporalUnary) and result.op == "G"
        assert isinstance(result.operand, VarRef) and result.operand.name == "p"


# ======================== Helper Tests ========================

class TestHelpers:
    def test_has_temporal_plain(self):
        assert not _has_temporal(p)
        assert not _has_temporal(BinOp("&", p, q))
        assert not _has_temporal(UnaryOp("!", p))

    def test_has_temporal_unary(self):
        assert _has_temporal(TemporalUnary("G", p))
        assert _has_temporal(TemporalUnary("F", p))

    def test_has_temporal_nested(self):
        assert _has_temporal(BinOp("&", p, TemporalUnary("F", q)))

    def test_has_temporal_binary(self):
        assert _has_temporal(TemporalBinary("U", p, q))

    def test_expr_eq_vars(self):
        assert _expr_eq(VarRef("x"), VarRef("x"))
        assert not _expr_eq(VarRef("x"), VarRef("y"))

    def test_expr_eq_binop(self):
        assert _expr_eq(BinOp("&", p, q), BinOp("&", p, q))
        assert not _expr_eq(BinOp("&", p, q), BinOp("|", p, q))

    def test_expr_eq_temporal(self):
        assert _expr_eq(TemporalUnary("G", p), TemporalUnary("G", p))
        assert not _expr_eq(TemporalUnary("G", p), TemporalUnary("F", p))


# ======================== Pattern: G(p) ========================

class TestPatternG:
    """G(p) — 'always p', 1 accepting state with self-loop on p."""

    def test_g_basic(self):
        buchi = build_buchi_for_spec(_spec(TemporalUnary("F", p)))
        # neg of F(p) = G(!p)
        assert len(buchi.states) == 1
        assert buchi.accepting == ["q0"]
        assert buchi.initial == "q0"

    def test_g_transitions(self):
        buchi = build_buchi_for_spec(_spec(TemporalUnary("F", p)))
        assert len(buchi.transitions) == 1
        t = buchi.transitions[0]
        assert t.src == "q0" and t.dst == "q0"

    def test_g_complex_ap(self):
        """G(x != 5) — complex atomic proposition."""
        expr = TemporalUnary("F", BinOp("=", _var("x"), IntLit(5)))
        buchi = build_buchi_for_spec(_spec(expr))
        assert len(buchi.states) == 1
        assert len(buchi.ap_exprs) == 1


# ======================== Pattern: F(p) ========================

class TestPatternF:
    """F(p) — 'eventually p', 2 states."""

    def test_f_basic(self):
        buchi = build_buchi_for_spec(_spec(TemporalUnary("G", p)))
        # neg of G(p) = F(!p)
        assert len(buchi.states) == 2
        assert buchi.accepting == ["q1"]
        assert buchi.initial == "q0"

    def test_f_transitions(self):
        buchi = build_buchi_for_spec(_spec(TemporalUnary("G", p)))
        assert len(buchi.transitions) == 3
        # q0->q0[t], q0->q1[!p], q1->q1[t]
        src_dst = [(t.src, t.dst) for t in buchi.transitions]
        assert ("q0", "q0") in src_dst
        assert ("q0", "q1") in src_dst
        assert ("q1", "q1") in src_dst


# ======================== Pattern: FG(p) ========================

class TestPatternFG:
    """FG(p) — 'eventually always p', 2 states."""

    def test_fg_basic(self):
        buchi = build_buchi_for_spec(_spec(TemporalUnary("G", TemporalUnary("F", p))))
        # neg of GF(p) = FG(!p)
        assert len(buchi.states) == 2
        assert buchi.accepting == ["q1"]

    def test_fg_transitions(self):
        buchi = build_buchi_for_spec(_spec(TemporalUnary("G", TemporalUnary("F", p))))
        assert len(buchi.transitions) == 3
        # q0->q0[t], q0->q1[!p], q1->q1[!p]
        # q1 self-loop is on the same guard as q0->q1


# ======================== Pattern: GF(p) ========================

class TestPatternGF:
    """GF(p) — 'always eventually p', 2 states."""

    def test_gf_basic(self):
        buchi = build_buchi_for_spec(_spec(TemporalUnary("F", TemporalUnary("G", p))))
        # neg of FG(p) = GF(!p)
        assert len(buchi.states) == 2
        assert buchi.accepting == ["q1"]

    def test_gf_transitions(self):
        buchi = build_buchi_for_spec(_spec(TemporalUnary("F", TemporalUnary("G", p))))
        assert len(buchi.transitions) == 3
        # q0->q0[t], q0->q1[!p], q1->q0[t]
        src_dst = [(t.src, t.dst) for t in buchi.transitions]
        assert ("q0", "q0") in src_dst
        assert ("q0", "q1") in src_dst
        assert ("q1", "q0") in src_dst


# ======================== Pattern: F(p & G(q)) ========================

class TestPatternFpGq:
    """F(p & G(q)) — 'eventually p and henceforth q', 2 states."""

    def test_response_basic(self):
        """G(p -> F q) negates to F(p & G(!q))."""
        expr = TemporalUnary("G", BinOp("->", p, TemporalUnary("F", q)))
        buchi = build_buchi_for_spec(_spec(expr))
        assert len(buchi.states) == 2
        assert buchi.accepting == ["q1"]
        assert len(buchi.transitions) == 3

    def test_response_aps(self):
        """Should extract 2 atomic propositions."""
        expr = TemporalUnary("G", BinOp("->", p, TemporalUnary("F", q)))
        buchi = build_buchi_for_spec(_spec(expr))
        assert len(buchi.ap_exprs) == 2

    def test_mutex_response(self):
        """G(waiting -> F critical) from mutex.smv."""
        waiting = BinOp("=", _var("p1"), _var("waiting"))
        critical = BinOp("=", _var("p1"), _var("critical"))
        expr = TemporalUnary("G", BinOp("->", waiting, TemporalUnary("F", critical)))
        buchi = build_buchi_for_spec(_spec(expr))
        assert len(buchi.states) == 2
        assert buchi.accepting == ["q1"]


# ======================== Pattern: F(p & F(q)) ========================

class TestPatternFpFq:
    """F(p & F(q)) — 'eventually p, then eventually q', 3 states."""

    def test_persistence_basic(self):
        """G(p -> G q) negates to F(p & F(!q))."""
        expr = TemporalUnary("G", BinOp("->", p, TemporalUnary("G", q)))
        buchi = build_buchi_for_spec(_spec(expr))
        assert len(buchi.states) == 3
        assert buchi.accepting == ["q2"]
        assert buchi.initial == "q0"

    def test_persistence_transitions(self):
        """5 transitions: q0->q0[t], q0->q1[p], q1->q1[t], q1->q2[!q], q2->q2[t]."""
        expr = TemporalUnary("G", BinOp("->", p, TemporalUnary("G", q)))
        buchi = build_buchi_for_spec(_spec(expr))
        assert len(buchi.transitions) == 5
        src_dst = [(t.src, t.dst) for t in buchi.transitions]
        assert ("q0", "q0") in src_dst
        assert ("q0", "q1") in src_dst
        assert ("q1", "q1") in src_dst
        assert ("q1", "q2") in src_dst
        assert ("q2", "q2") in src_dst

    def test_done_implies_g_done(self):
        """G(done -> G done) from bubble_sort3.smv and swap.smv."""
        done = BinOp("=", _var("pc"), _var("done"))
        expr = TemporalUnary("G", BinOp("->", done, TemporalUnary("G", done)))
        buchi = build_buchi_for_spec(_spec(expr))
        assert len(buchi.states) == 3
        assert buchi.accepting == ["q2"]
        assert len(buchi.ap_exprs) == 2  # done, !done


# ======================== Pattern: p & G(q) ========================

class TestPatternPandGq:
    """p & G(q) — 'p initially and always q', 2 states."""

    def test_conditional_eventually(self):
        """(cond) -> F(result) negates to cond & G(!result)."""
        cond = BinOp("&", BinOp("=", _var("a"), IntLit(5)),
                      BinOp("=", _var("b"), IntLit(3)))
        result = BinOp("=", _var("result"), IntLit(2))
        expr = BinOp("->", cond, TemporalUnary("F", result))
        buchi = build_buchi_for_spec(_spec(expr))
        assert len(buchi.states) == 2
        assert buchi.accepting == ["q1"]

    def test_conditional_transitions(self):
        """2 transitions: q0->q1[p&q], q1->q1[q]."""
        expr = BinOp("->", p, TemporalUnary("F", q))
        buchi = build_buchi_for_spec(_spec(expr))
        assert len(buchi.transitions) == 2
        src_dst = [(t.src, t.dst) for t in buchi.transitions]
        assert ("q0", "q1") in src_dst
        assert ("q1", "q1") in src_dst

    def test_no_self_loop_on_initial(self):
        """q0 has no self-loop (must leave immediately or reject)."""
        expr = BinOp("->", p, TemporalUnary("F", q))
        buchi = build_buchi_for_spec(_spec(expr))
        for t in buchi.transitions:
            if t.src == "q0":
                assert t.dst == "q1"  # only transition from q0 goes to q1

    def test_mult_spec(self):
        """(a=2 & b=3) -> F(prod=6) from mult.smv."""
        cond = BinOp("&", BinOp("=", _var("a"), IntLit(2)),
                      BinOp("=", _var("b"), IntLit(3)))
        result = BinOp("=", _var("prod"), IntLit(6))
        expr = BinOp("->", cond, TemporalUnary("F", result))
        buchi = build_buchi_for_spec(_spec(expr))
        assert len(buchi.states) == 2

    def test_or_condition(self):
        """(a=0 | b=0) -> F(prod=0) — disjunctive condition."""
        cond = BinOp("|", BinOp("=", _var("a"), IntLit(0)),
                      BinOp("=", _var("b"), IntLit(0)))
        result = BinOp("=", _var("prod"), IntLit(0))
        expr = BinOp("->", cond, TemporalUnary("F", result))
        buchi = build_buchi_for_spec(_spec(expr))
        assert len(buchi.states) == 2
        assert len(buchi.ap_exprs) == 2


# ======================== Pattern: G(p -> G(q)) ========================

class TestPatternGpImplGq:
    """G(p -> G(q)) as a negated formula pattern — 2 states."""

    def test_direct_negated_match(self):
        """Directly construct the negated formula G(p -> G(q)) and match."""
        negated = TemporalUnary("G", BinOp("->", p, TemporalUnary("G", q)))
        buchi = ltl_to_buchi(negated)
        assert len(buchi.states) == 2
        assert buchi.accepting == ["q0"]  # q0 is accepting, q1 is trap

    def test_g_impl_g_transitions(self):
        negated = TemporalUnary("G", BinOp("->", p, TemporalUnary("G", q)))
        buchi = ltl_to_buchi(negated)
        assert len(buchi.transitions) == 3
        # q0->q0[!p|q], q0->q1[p&!q], q1->q1[t]


# ======================== Unsupported Pattern ========================

class TestUnsupported:
    def test_until_not_supported(self):
        """p U q should fail pattern matching (no pattern for Until)."""
        expr = TemporalBinary("U", p, q)
        # Negation wraps in ! — won't match any pattern
        with pytest.raises(UnsupportedLTLPattern):
            build_buchi_for_spec(_spec(expr))


# ======================== HOA Parser Tests ========================

class TestHOAParsing:
    def test_simple_buchi(self):
        """Parse a simple 2-state Buchi from HOA."""
        hoa = """\
HOA: v1
States: 2
Start: 0
AP: 1 "p"
Acceptance: 1 Inf(0)
--BODY--
State: 0
[t] 0
[0] 1
State: 1 {0}
[t] 1
--END--"""
        ap_exprs = {"p": VarRef("p")}
        buchi = parse_hoa(hoa, ap_exprs)
        assert len(buchi.states) == 2
        assert buchi.initial == "q0"
        assert buchi.accepting == ["q1"]
        assert len(buchi.transitions) == 3

    def test_hoa_negation_guard(self):
        """Parse guard with negation: !0."""
        hoa = """\
HOA: v1
States: 1
Start: 0
AP: 1 "p"
Acceptance: 1 Inf(0)
--BODY--
State: 0 {0}
[!0] 0
--END--"""
        ap_exprs = {"p": VarRef("p")}
        buchi = parse_hoa(hoa, ap_exprs)
        assert len(buchi.transitions) == 1
        guard = buchi.transitions[0].guard
        assert isinstance(guard, UnaryOp) and guard.op == "!"

    def test_hoa_conjunction_guard(self):
        """Parse guard with conjunction: 0 & 1."""
        hoa = """\
HOA: v1
States: 1
Start: 0
AP: 2 "p" "q"
Acceptance: 1 Inf(0)
--BODY--
State: 0 {0}
[0 & 1] 0
--END--"""
        ap_exprs = {"p": VarRef("p"), "q": VarRef("q")}
        buchi = parse_hoa(hoa, ap_exprs)
        guard = buchi.transitions[0].guard
        assert isinstance(guard, BinOp) and guard.op == "&"

    def test_hoa_two_aps(self):
        """Parse HOA with 2 atomic propositions."""
        hoa = """\
HOA: v1
States: 2
Start: 0
AP: 2 "a" "b"
Acceptance: 1 Inf(0)
--BODY--
State: 0
[t] 0
[0 & 1] 1
State: 1 {0}
[1] 1
--END--"""
        ap_exprs = {"a": VarRef("a"), "b": VarRef("b")}
        buchi = parse_hoa(hoa, ap_exprs)
        assert len(buchi.states) == 2
        assert len(buchi.transitions) == 3
        assert len(buchi.accepting) == 1


# ======================== Integration: Real Model Specs ========================

class TestRealModelSpecs:
    """Test Buchi construction for actual LTLSPEC formulas in example models."""

    def test_counter_specs(self):
        """All LTLSPEC in counter.smv should produce valid Buchi automata."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "counter.smv"))
        ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
        assert len(ltl_specs) > 0
        for spec in ltl_specs:
            buchi = build_buchi_for_spec(spec)
            assert len(buchi.states) >= 1
            assert len(buchi.accepting) >= 1
            assert buchi.initial in [s.name for s in buchi.states]

    def test_mutex_specs(self):
        """All LTLSPEC in mutex.smv should produce valid Buchi automata."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
        assert len(ltl_specs) >= 3
        for spec in ltl_specs:
            buchi = build_buchi_for_spec(spec)
            assert len(buchi.states) >= 1

    def test_mutex_response_is_2state(self):
        """G(waiting -> F critical) produces a 2-state Buchi."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
        # First two are G(waiting -> F critical) specs
        for spec in ltl_specs[:2]:
            buchi = build_buchi_for_spec(spec)
            assert len(buchi.states) == 2

    def test_mutex_eventually_is_1state(self):
        """F(p1 = critical) produces a 1-state Buchi (negation is G(!...))."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mutex.smv"))
        ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
        # Third spec: F(process1 = critical)
        buchi = build_buchi_for_spec(ltl_specs[2])
        assert len(buchi.states) == 1

    def test_bubble_sort3_persistence(self):
        """G(pc=done -> G(pc=done)) produces 3-state Buchi."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "bubble_sort3.smv"))
        ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
        assert len(ltl_specs) >= 1
        buchi = build_buchi_for_spec(ltl_specs[0])
        assert len(buchi.states) == 3

    def test_swap_specs(self):
        """All LTLSPEC in swap.smv should produce valid Buchi automata."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "swap.smv"))
        ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
        assert len(ltl_specs) >= 2
        for spec in ltl_specs:
            buchi = build_buchi_for_spec(spec)
            assert len(buchi.states) >= 1

    def test_gcd_specs(self):
        """All LTLSPEC in gcd_01.smv should produce valid Buchi automata."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "gcd_01.smv"))
        ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
        assert len(ltl_specs) >= 1
        for spec in ltl_specs:
            buchi = build_buchi_for_spec(spec)
            assert len(buchi.states) >= 1

    def test_mult_specs(self):
        """All LTLSPEC in mult.smv should produce valid Buchi automata."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "mult.smv"))
        ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
        assert len(ltl_specs) >= 1
        for spec in ltl_specs:
            buchi = build_buchi_for_spec(spec)
            assert len(buchi.states) >= 1

    def test_fibonacci_specs(self):
        """All LTLSPEC in fibonacci.smv should produce valid Buchi automata."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "fibonacci.smv"))
        ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
        assert len(ltl_specs) >= 1
        for spec in ltl_specs:
            buchi = build_buchi_for_spec(spec)
            assert len(buchi.states) >= 1

    def test_abs_diff_specs(self):
        """All LTLSPEC in abs_diff.smv should produce valid Buchi automata."""
        model = parse_smv_file(os.path.join(EXAMPLES_DIR, "abs_diff.smv"))
        ltl_specs = [s for s in model.specs if s.kind == "LTLSPEC"]
        assert len(ltl_specs) >= 1
        for spec in ltl_specs:
            buchi = build_buchi_for_spec(spec)
            assert len(buchi.states) >= 1


# ======================== Buchi Structural Invariants ========================

class TestBuchiInvariants:
    """Verify structural properties that every Buchi automaton should satisfy."""

    def _check_invariants(self, buchi: BuchiAutomaton):
        state_names = {s.name for s in buchi.states}
        # Initial state exists
        assert buchi.initial in state_names
        # All accepting states exist
        for a in buchi.accepting:
            assert a in state_names
        # All transitions reference valid states
        for t in buchi.transitions:
            assert t.src in state_names, f"src {t.src} not in states"
            assert t.dst in state_names, f"dst {t.dst} not in states"
        # At least one accepting state
        assert len(buchi.accepting) >= 1
        # Accepting flag consistency
        for s in buchi.states:
            assert s.accepting == (s.name in buchi.accepting)

    def test_invariants_all_patterns(self):
        """Check invariants on Buchi for every pattern."""
        specs = [
            TemporalUnary("G", p),              # neg -> F(!p)
            TemporalUnary("F", p),              # neg -> G(!p)
            TemporalUnary("G", TemporalUnary("F", p)),  # neg -> FG(!p)
            TemporalUnary("F", TemporalUnary("G", p)),  # neg -> GF(!p)
            TemporalUnary("G", BinOp("->", p, TemporalUnary("F", q))),  # F(p&G(!q))
            TemporalUnary("G", BinOp("->", p, TemporalUnary("G", q))),  # F(p&F(!q))
            BinOp("->", p, TemporalUnary("F", q)),  # p & G(!q)
        ]
        for expr in specs:
            buchi = build_buchi_for_spec(_spec(expr))
            self._check_invariants(buchi)

    def test_invariants_real_models(self):
        """Check invariants on Buchi for every LTLSPEC in every example model."""
        for fname in ["counter.smv", "mutex.smv", "swap.smv", "mult.smv",
                       "bubble_sort3.smv", "fibonacci.smv", "gcd_01.smv", "abs_diff.smv"]:
            path = os.path.join(EXAMPLES_DIR, fname)
            if not os.path.exists(path):
                continue
            model = parse_smv_file(path)
            for spec in model.specs:
                if spec.kind == "LTLSPEC":
                    buchi = build_buchi_for_spec(spec)
                    self._check_invariants(buchi)
