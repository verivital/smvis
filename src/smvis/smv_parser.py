"""SMV file parser using Lark. Parses the nuXmv subset used in EECS 6315 models."""
from __future__ import annotations
import os
from lark import Lark, Transformer, Token
from smvis.smv_model import (
    SmvModel, VarDecl, BoolType, EnumType, RangeType,
    IntLit, BoolLit, VarRef, NextRef, BinOp, UnaryOp,
    CaseExpr, SetExpr, TemporalUnary, TemporalBinary, SpecDecl,
)

_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "smv_grammar.lark")

_parser = None

def _get_parser():
    global _parser
    if _parser is None:
        with open(_GRAMMAR_PATH, "r") as f:
            grammar_text = f.read()
        _parser = Lark(grammar_text, parser="lalr")
    return _parser


class SmvTransformer(Transformer):
    """Transforms the Lark parse tree into SmvModel dataclasses."""

    # ---- Terminals ----
    def IDENT(self, token):
        return str(token)

    def INT(self, token):
        return int(token)

    # ---- Variable types ----
    def bool_type(self, args):
        return BoolType()

    def enum_type(self, args):
        return EnumType(args[0])

    def ident_list(self, args):
        return list(args)

    def range_type(self, args):
        # Bounds may be int (literal) or str (DEFINE reference, resolved later)
        lo = args[0] if isinstance(args[0], str) else int(args[0])
        hi = args[1] if isinstance(args[1], str) else int(args[1])
        return RangeType(lo, hi)

    def neg_bound(self, args):
        return -args[0]

    def pos_bound(self, args):
        return args[0]

    def ident_bound(self, args):
        return str(args[0])

    # ---- Expressions: literals and references ----
    def var_ref(self, args):
        return VarRef(str(args[0]))

    def int_lit(self, args):
        return IntLit(int(args[0]))

    def true_const(self, args):
        return BoolLit(True)

    def false_const(self, args):
        return BoolLit(False)

    def next_ref(self, args):
        return NextRef(str(args[0]))

    def unary_minus(self, args):
        # Optimize: if operand is IntLit, fold to negative IntLit
        if isinstance(args[0], IntLit):
            return IntLit(-args[0].value)
        return UnaryOp("-", args[0])

    # ---- Boolean / comparison operators ----
    def neg_op(self, args):
        return UnaryOp("!", args[0])

    def impl_op(self, args):
        return BinOp("->", args[0], args[1])

    def or_op(self, args):
        return BinOp("|", args[0], args[1])

    def and_op(self, args):
        return BinOp("&", args[0], args[1])

    def eq_op(self, args):
        return BinOp("=", args[0], args[1])

    def ne_op(self, args):
        return BinOp("!=", args[0], args[1])

    def ge_op(self, args):
        return BinOp(">=", args[0], args[1])

    def le_op(self, args):
        return BinOp("<=", args[0], args[1])

    def gt_op(self, args):
        return BinOp(">", args[0], args[1])

    def lt_op(self, args):
        return BinOp("<", args[0], args[1])

    # ---- Arithmetic operators ----
    def add_op(self, args):
        return BinOp("+", args[0], args[1])

    def sub_op(self, args):
        return BinOp("-", args[0], args[1])

    def mul_op(self, args):
        return BinOp("*", args[0], args[1])

    def div_op(self, args):
        return BinOp("/", args[0], args[1])

    def mod_op(self, args):
        return BinOp("mod", args[0], args[1])

    # ---- Temporal operators ----
    def ag_op(self, args):
        return TemporalUnary("AG", args[0])

    def af_op(self, args):
        return TemporalUnary("AF", args[0])

    def ax_op(self, args):
        return TemporalUnary("AX", args[0])

    def eg_op(self, args):
        return TemporalUnary("EG", args[0])

    def ef_op(self, args):
        return TemporalUnary("EF", args[0])

    def ex_op(self, args):
        return TemporalUnary("EX", args[0])

    def g_op(self, args):
        return TemporalUnary("G", args[0])

    def f_op(self, args):
        return TemporalUnary("F", args[0])

    def x_op(self, args):
        return TemporalUnary("X", args[0])

    def until_op(self, args):
        return TemporalBinary("U", args[0], args[1])

    # ---- Case / Set expressions ----
    def case_expr(self, args):
        return CaseExpr(list(args))

    def case_branch(self, args):
        return (args[0], args[1])

    def set_expr(self, args):
        return SetExpr(list(args))

    # ---- Variable declarations ----
    def var_decl(self, args):
        return VarDecl(str(args[0]), args[1])

    def var_section(self, args):
        return ("VAR", list(args))

    # ---- DEFINE ----
    def define_decl(self, args):
        return (str(args[0]), args[1])

    def define_section(self, args):
        return ("DEFINE", list(args))

    # ---- ASSIGN ----
    def init_assign(self, args):
        return ("init", str(args[0]), args[1])

    def next_assign(self, args):
        return ("next", str(args[0]), args[1])

    def assign_section(self, args):
        return ("ASSIGN", list(args))

    # ---- FAIRNESS ----
    def fairness_decl(self, args):
        return ("FAIRNESS", args[0])

    # ---- Specifications ----
    def invarspec(self, args):
        return ("SPEC", SpecDecl("INVARSPEC", args[0]))

    def ctlspec(self, args):
        return ("SPEC", SpecDecl("CTLSPEC", args[0]))

    def ltlspec(self, args):
        return ("SPEC", SpecDecl("LTLSPEC", args[0]))

    def spec_default(self, args):
        return ("SPEC", SpecDecl("SPEC", args[0]))

    # ---- Top-level ----
    def module_decl(self, args):
        return None

    def start(self, args):
        model = SmvModel()
        for item in args:
            if item is None:
                continue
            kind = item[0]
            if kind == "VAR":
                for vd in item[1]:
                    model.variables[vd.name] = vd
            elif kind == "DEFINE":
                for name, expr in item[1]:
                    model.defines[name] = expr
            elif kind == "ASSIGN":
                for a in item[1]:
                    if a[0] == "init":
                        model.inits[a[1]] = a[2]
                    elif a[0] == "next":
                        model.nexts[a[1]] = a[2]
            elif kind == "FAIRNESS":
                model.fairness.append(item[1])
            elif kind == "SPEC":
                model.specs.append(item[1])
        return model


_transformer = SmvTransformer()


def _eval_const_expr(expr, defines: dict) -> int:
    """Evaluate a constant expression (DEFINE body) to an integer."""
    if isinstance(expr, IntLit):
        return expr.value
    if isinstance(expr, VarRef):
        # Recursive DEFINE lookup
        if expr.name in defines:
            return _eval_const_expr(defines[expr.name], defines)
        raise ValueError(f"Cannot resolve '{expr.name}' to an integer")
    if isinstance(expr, BinOp):
        l = _eval_const_expr(expr.left, defines)
        r = _eval_const_expr(expr.right, defines)
        ops = {"+": l + r, "-": l - r, "*": l * r, "/": l // r, "mod": l % r}
        if expr.op in ops:
            return ops[expr.op]
    if isinstance(expr, UnaryOp) and expr.op == "-":
        return -_eval_const_expr(expr.operand, defines)
    raise ValueError(f"Cannot evaluate expression to integer: {expr}")


def _resolve_range_bounds(model: SmvModel):
    """Resolve DEFINE references in range bounds to integer values."""
    for vd in model.variables.values():
        if isinstance(vd.var_type, RangeType):
            rt = vd.var_type
            if isinstance(rt.lo, str):
                if rt.lo in model.defines:
                    rt.lo = _eval_const_expr(model.defines[rt.lo], model.defines)
                else:
                    raise ValueError(
                        f"Range bound '{rt.lo}' for variable '{vd.name}' "
                        f"is not defined"
                    )
            if isinstance(rt.hi, str):
                if rt.hi in model.defines:
                    rt.hi = _eval_const_expr(model.defines[rt.hi], model.defines)
                else:
                    raise ValueError(
                        f"Range bound '{rt.hi}' for variable '{vd.name}' "
                        f"is not defined"
                    )


def parse_smv(text: str) -> SmvModel:
    """Parse an SMV model string and return an SmvModel."""
    parser = _get_parser()
    tree = parser.parse(text)
    model = _transformer.transform(tree)
    _resolve_range_bounds(model)
    return model


def parse_smv_file(filepath: str) -> SmvModel:
    """Parse an SMV file and return an SmvModel."""
    with open(filepath, "r") as f:
        text = f.read()
    return parse_smv(text)
