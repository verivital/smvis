"""Unit tests for the expression evaluator."""
import pytest
from smvis.smv_model import (
    IntLit, BoolLit, VarRef, NextRef, BinOp, UnaryOp,
    CaseExpr, SetExpr,
)
from smvis.explicit_engine import evaluate


class TestLiterals:
    def test_int_lit(self):
        assert evaluate(IntLit(42), {}) == 42

    def test_bool_true(self):
        assert evaluate(BoolLit(True), {}) is True

    def test_bool_false(self):
        assert evaluate(BoolLit(False), {}) is False


class TestVarRef:
    def test_state_var(self):
        assert evaluate(VarRef("x"), {"x": 5}) == 5

    def test_enum_constant(self):
        assert evaluate(VarRef("idle"), {}) == "idle"

    def test_define_expansion(self):
        defines = {"count_max": IntLit(10)}
        assert evaluate(VarRef("count_max"), {}, defines=defines) == 10


class TestArithmetic:
    def test_add(self):
        assert evaluate(BinOp("+", IntLit(3), IntLit(4)), {}) == 7

    def test_sub(self):
        assert evaluate(BinOp("-", IntLit(10), IntLit(3)), {}) == 7

    def test_mul(self):
        assert evaluate(BinOp("*", IntLit(3), IntLit(4)), {}) == 12

    def test_div(self):
        assert evaluate(BinOp("/", IntLit(10), IntLit(3)), {}) == 3

    def test_div_by_zero(self):
        assert evaluate(BinOp("/", IntLit(10), IntLit(0)), {}) == 0

    def test_mod(self):
        assert evaluate(BinOp("mod", IntLit(10), IntLit(3)), {}) == 1

    def test_mod_by_zero(self):
        assert evaluate(BinOp("mod", IntLit(10), IntLit(0)), {}) == 0


class TestComparison:
    def test_eq_true(self):
        assert evaluate(BinOp("=", IntLit(5), IntLit(5)), {}) is True

    def test_eq_false(self):
        assert evaluate(BinOp("=", IntLit(5), IntLit(3)), {}) is False

    def test_ne(self):
        assert evaluate(BinOp("!=", IntLit(5), IntLit(3)), {}) is True

    def test_gt(self):
        assert evaluate(BinOp(">", IntLit(5), IntLit(3)), {}) is True

    def test_lt(self):
        assert evaluate(BinOp("<", IntLit(3), IntLit(5)), {}) is True

    def test_ge(self):
        assert evaluate(BinOp(">=", IntLit(5), IntLit(5)), {}) is True

    def test_le(self):
        assert evaluate(BinOp("<=", IntLit(5), IntLit(5)), {}) is True


class TestBoolean:
    def test_and_true(self):
        assert evaluate(BinOp("&", BoolLit(True), BoolLit(True)), {}) is True

    def test_and_false(self):
        assert evaluate(BinOp("&", BoolLit(True), BoolLit(False)), {}) is False

    def test_or_true(self):
        assert evaluate(BinOp("|", BoolLit(False), BoolLit(True)), {}) is True

    def test_or_false(self):
        assert evaluate(BinOp("|", BoolLit(False), BoolLit(False)), {}) is False

    def test_implies_true(self):
        assert evaluate(BinOp("->", BoolLit(False), BoolLit(False)), {}) is True

    def test_implies_false(self):
        assert evaluate(BinOp("->", BoolLit(True), BoolLit(False)), {}) is False

    def test_not_true(self):
        assert evaluate(UnaryOp("!", BoolLit(True)), {}) is False

    def test_not_false(self):
        assert evaluate(UnaryOp("!", BoolLit(False)), {}) is True

    def test_unary_minus(self):
        assert evaluate(UnaryOp("-", IntLit(5)), {}) == -5


class TestCaseExpr:
    def test_first_match(self):
        expr = CaseExpr([(BoolLit(True), IntLit(42))])
        assert evaluate(expr, {}) == 42

    def test_fallthrough(self):
        expr = CaseExpr([
            (BoolLit(False), IntLit(1)),
            (BoolLit(True), IntLit(2)),
        ])
        assert evaluate(expr, {}) == 2

    def test_no_match_raises(self):
        expr = CaseExpr([(BoolLit(False), IntLit(1))])
        with pytest.raises(ValueError, match="No case branch matched"):
            evaluate(expr, {})


class TestSetExpr:
    def test_set_returns_list(self):
        expr = SetExpr([IntLit(1), IntLit(2), IntLit(3)])
        assert evaluate(expr, {}) == [1, 2, 3]


class TestNextRef:
    def test_reads_next_state(self):
        assert evaluate(NextRef("x"), {"x": 1}, next_state={"x": 2}) == 2

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="next.*not yet computed"):
            evaluate(NextRef("x"), {"x": 1}, next_state={})
