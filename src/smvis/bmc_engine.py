"""Bounded Model Checking engine using Z3 SMT solver."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any

import z3

from smvis.smv_model import (
    SmvModel, VarDecl, BoolType, EnumType, RangeType,
    Expr, IntLit, BoolLit, VarRef, NextRef, BinOp, UnaryOp,
    CaseExpr, SetExpr, TemporalUnary, TemporalBinary,
    SpecDecl, get_domain, expr_to_str,
)
from smvis.explicit_engine import compute_dep_order


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BmcStepResult:
    """Result of checking one bound k."""
    k: int
    status: str          # "SAT" | "UNSAT" | "UNKNOWN" | "SKIPPED"
    time_s: float
    counterexample: list[dict[str, Any]] | None = None


@dataclass
class BmcFormulas:
    """Captured SMT formulas for inspection."""
    encoding: str              # variable encoding legend
    init: str                  # I(s₀) constraints
    transition: str            # T(s₀, s₁) representative transition
    property_negation: str     # ¬P(sₖ) at violation bound (or last k)
    full_check: str            # full solver assertion at the relevant bound
    smt2: str                  # SMT-LIB2 export of the full check


@dataclass
class BmcResult:
    """Full BMC result across all bounds checked."""
    spec_text: str
    spec_kind: str
    max_k: int
    step_results: list[BmcStepResult]
    violated: bool
    violation_k: int | None
    counterexample: list[dict[str, Any]] | None
    total_time_s: float
    formulas: BmcFormulas | None = None


# ---------------------------------------------------------------------------
# Z3 variable info
# ---------------------------------------------------------------------------

@dataclass
class _Z3VarInfo:
    """Z3 variable metadata for one SMV variable."""
    smv_name: str
    var_type: Any
    z3_vars: dict[int, Any] = field(default_factory=dict)  # step -> z3 var
    domain_values: list = field(default_factory=list)
    z3_sort: Any = None          # z3 sort (BoolSort, DatatypeSort, IntSort)
    z3_datatype: Any = None      # for EnumType only
    val_to_z3: dict = field(default_factory=dict)
    z3_to_val: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BmcEncoder
# ---------------------------------------------------------------------------

class BmcEncoder:
    """Translates an SmvModel into Z3 constraints for BMC."""

    def __init__(self, model: SmvModel):
        self.model = model
        self.var_info: dict[str, _Z3VarInfo] = {}
        self._aux_counter = 0
        self._pending_constraints: list[z3.BoolRef] = []
        self._dep_order = compute_dep_order(model)
        self._setup_types()

    # ---- Type setup ----

    def _setup_types(self):
        """Create Z3 sorts and value mappings for each SMV variable."""
        for name, vd in self.model.variables.items():
            vt = vd.var_type
            domain = get_domain(vd)
            info = _Z3VarInfo(smv_name=name, var_type=vt, domain_values=domain)

            if isinstance(vt, BoolType):
                info.z3_sort = z3.BoolSort()
                info.val_to_z3 = {True: z3.BoolVal(True), False: z3.BoolVal(False)}
                info.z3_to_val = {"True": True, "False": False}

            elif isinstance(vt, EnumType):
                # Encode enum as Int with domain constraints.
                # Each enum value maps to a unique integer code.
                info.z3_sort = z3.IntSort()
                for code, val in enumerate(vt.values):
                    info.val_to_z3[val] = z3.IntVal(code)
                    info.z3_to_val[str(code)] = val

            elif isinstance(vt, RangeType):
                info.z3_sort = z3.IntSort()
                for v in domain:
                    info.val_to_z3[v] = z3.IntVal(v)
                    info.z3_to_val[str(v)] = v

            self.var_info[name] = info

    # ---- Variable access ----

    def get_var(self, name: str, step: int) -> z3.ExprRef:
        """Get or create the Z3 variable for *name* at *step*."""
        info = self.var_info[name]
        if step not in info.z3_vars:
            suffix = f"{name}_{step}"
            if isinstance(info.var_type, BoolType):
                info.z3_vars[step] = z3.Bool(suffix)
            else:
                # EnumType and RangeType both use Int encoding
                info.z3_vars[step] = z3.Int(suffix)
        return info.z3_vars[step]

    # ---- Constraint builders ----

    def domain_constraints(self, step: int) -> list[z3.BoolRef]:
        """Return domain constraints for all Int-encoded variables at *step*."""
        constraints: list[z3.BoolRef] = []
        for name, info in self.var_info.items():
            if isinstance(info.var_type, RangeType):
                v = self.get_var(name, step)
                constraints.append(v >= info.var_type.lo)
                constraints.append(v <= info.var_type.hi)
            elif isinstance(info.var_type, EnumType):
                v = self.get_var(name, step)
                n_vals = len(info.var_type.values)
                constraints.append(v >= 0)
                constraints.append(v < n_vals)
        return constraints

    def encode_init(self) -> list[z3.BoolRef]:
        """Encode initial-state constraints on step-0 variables."""
        constraints = list(self.domain_constraints(0))
        for var_name, init_expr in self.model.inits.items():
            z3_val = self.expr_to_z3(init_expr, step=0)
            z3_var = self.get_var(var_name, 0)
            constraints.append(z3_var == z3_val)
        constraints.extend(self._flush_pending())
        return constraints

    def encode_transition(self, step: int) -> list[z3.BoolRef]:
        """Encode T(s_step, s_{step+1})."""
        constraints = list(self.domain_constraints(step + 1))
        computed_nexts: dict[str, z3.ExprRef] = {}

        for var_name in self._dep_order:
            if var_name in self.model.nexts:
                z3_next = self.expr_to_z3(
                    self.model.nexts[var_name],
                    step=step,
                    next_step=step + 1,
                    computed_nexts=computed_nexts,
                )
                z3_var_next = self.get_var(var_name, step + 1)
                constraints.append(z3_var_next == z3_next)
                computed_nexts[var_name] = z3_var_next
            else:
                # No next(): unconstrained (domain constraints already added)
                computed_nexts[var_name] = self.get_var(var_name, step + 1)

        constraints.extend(self._flush_pending())
        return constraints

    def encode_property_negation(self, spec: SpecDecl, step: int) -> z3.BoolRef:
        """Return the Z3 expression for 'property violated at *step*'."""
        body = _extract_safety_body(spec)
        z3_prop = self.expr_to_z3(body, step=step)
        return z3.Not(z3_prop)

    # ---- Expression translator ----

    def expr_to_z3(
        self,
        expr: Expr,
        step: int,
        next_step: int | None = None,
        computed_nexts: dict[str, z3.ExprRef] | None = None,
    ) -> z3.ExprRef:
        """Recursively translate an Expr AST to a Z3 expression."""
        if isinstance(expr, IntLit):
            return z3.IntVal(expr.value)

        if isinstance(expr, BoolLit):
            return z3.BoolVal(expr.value)

        if isinstance(expr, VarRef):
            name = expr.name
            # DEFINE inline expansion
            if name in self.model.defines:
                return self.expr_to_z3(
                    self.model.defines[name], step, next_step, computed_nexts,
                )
            # State variable
            if name in self.var_info:
                return self.get_var(name, step)
            # Enum constant
            return self._enum_const_to_z3(name)

        if isinstance(expr, NextRef):
            if computed_nexts and expr.name in computed_nexts:
                return computed_nexts[expr.name]
            if next_step is not None:
                return self.get_var(expr.name, next_step)
            raise ValueError(f"next({expr.name}) without next_step context")

        if isinstance(expr, UnaryOp):
            operand = self.expr_to_z3(expr.operand, step, next_step, computed_nexts)
            if expr.op == "!":
                return z3.Not(_to_bool(operand))
            if expr.op == "-":
                return -_to_int(operand)
            raise ValueError(f"Unknown unary op: {expr.op}")

        if isinstance(expr, BinOp):
            left = self.expr_to_z3(expr.left, step, next_step, computed_nexts)
            right = self.expr_to_z3(expr.right, step, next_step, computed_nexts)
            return _binop_to_z3(expr.op, left, right)

        if isinstance(expr, CaseExpr):
            return self._case_to_z3(expr, step, next_step, computed_nexts)

        if isinstance(expr, SetExpr):
            return self._set_to_z3(expr, step, next_step, computed_nexts)

        raise ValueError(f"Cannot translate to Z3: {type(expr).__name__}")

    # ---- Case expression ----

    def _case_to_z3(self, expr: CaseExpr, step, next_step, computed_nexts):
        """Nested z3.If encoding (first-match semantics)."""
        result = None
        for cond_expr, val_expr in reversed(expr.branches):
            z3_val = self.expr_to_z3(val_expr, step, next_step, computed_nexts)
            z3_cond = self.expr_to_z3(cond_expr, step, next_step, computed_nexts)
            z3_cond = _to_bool(z3_cond)
            if result is None:
                result = z3_val
            else:
                # Ensure consistent sorts for If branches
                z3_val, result = _coerce_pair(z3_val, result)
                result = z3.If(z3_cond, z3_val, result)
        return result

    # ---- Set expression (non-determinism) ----

    def _set_to_z3(self, expr: SetExpr, step, next_step, computed_nexts):
        """Introduce a fresh aux variable constrained to one of the values."""
        self._aux_counter += 1
        values = [
            self.expr_to_z3(v, step, next_step, computed_nexts)
            for v in expr.values
        ]
        sort = values[0].sort()
        aux = z3.Const(f"_aux_{step}_{self._aux_counter}", sort)
        self._pending_constraints.append(z3.Or(*[aux == v for v in values]))
        return aux

    # ---- Enum constant lookup ----

    def _enum_const_to_z3(self, name: str) -> z3.ExprRef:
        """Look up an enum constant across all variable Datatypes."""
        for info in self.var_info.values():
            if name in info.val_to_z3:
                return info.val_to_z3[name]
        raise ValueError(f"Unknown identifier: {name}")

    # ---- Counterexample extraction ----

    def extract_counterexample(
        self, z3_model: z3.ModelRef, max_step: int,
    ) -> list[dict[str, Any]]:
        """Extract concrete state trace from a Z3 satisfying model."""
        trace: list[dict[str, Any]] = []
        for step in range(max_step + 1):
            state: dict[str, Any] = {}
            for name, info in self.var_info.items():
                z3_var = info.z3_vars.get(step)
                if z3_var is None:
                    z3_var = self.get_var(name, step)
                z3_val = z3_model.evaluate(z3_var, model_completion=True)
                state[name] = self._z3_val_to_smv(name, z3_val)
            trace.append(state)
        return trace

    def _z3_val_to_smv(self, var_name: str, z3_val) -> Any:
        """Convert a Z3 value back to an SMV domain value."""
        info = self.var_info[var_name]
        if isinstance(info.var_type, BoolType):
            return z3.is_true(z3_val)
        if isinstance(info.var_type, EnumType):
            code = z3_val.as_long()
            return info.z3_to_val.get(str(code), str(z3_val))
        if isinstance(info.var_type, RangeType):
            return z3_val.as_long()
        return str(z3_val)

    # ---- Formula inspection ----

    def encoding_legend(self) -> str:
        """Human-readable description of the variable encoding."""
        lines = []
        for name, info in self.var_info.items():
            vt = info.var_type
            if isinstance(vt, BoolType):
                lines.append(f"  {name} : Bool")
            elif isinstance(vt, EnumType):
                mapping = ", ".join(f"{v}={c}" for v, c in
                                   sorted(info.val_to_z3.items(),
                                          key=lambda x: x[1].as_long()))
                lines.append(f"  {name} : Int  ({mapping})")
            elif isinstance(vt, RangeType):
                lines.append(f"  {name} : Int  [{vt.lo}..{vt.hi}]")
        header = ("; Variable encoding\n"
                  "; Each SMV variable at step k is named <var>_<k>\n")
        return header + "\n".join(lines)

    @staticmethod
    def _fmt_constraints(constraints: list[z3.BoolRef], header: str) -> str:
        """Format a list of Z3 constraints as readable text."""
        lines = [f"; {header}"]
        for i, c in enumerate(constraints):
            lines.append(f"  ({i+1})  {c.sexpr()}")
        if not constraints:
            lines.append("  (none)")
        return "\n".join(lines)

    # ---- Internal helpers ----

    def _flush_pending(self) -> list[z3.BoolRef]:
        out = list(self._pending_constraints)
        self._pending_constraints.clear()
        return out


# ---------------------------------------------------------------------------
# Top-level BMC functions
# ---------------------------------------------------------------------------

def run_bmc(
    model: SmvModel,
    spec: SpecDecl,
    max_k: int = 30,
    timeout_ms: int = 30000,
) -> BmcResult:
    """Run bounded model checking for a safety property.

    Checks whether the property is violated at any step 0..max_k.
    """
    spec_text = spec.text or expr_to_str(spec.expr)
    if not is_safety_spec(spec):
        return BmcResult(
            spec_text=spec_text, spec_kind=spec.kind,
            max_k=0, step_results=[], violated=False,
            violation_k=None, counterexample=None, total_time_s=0.0,
        )

    encoder = BmcEncoder(model)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    start_time = time.time()
    step_results: list[BmcStepResult] = []

    # Add init constraints (persistent across all bounds)
    init_constraints = encoder.encode_init()
    solver.add(*init_constraints)

    # Capture formula strings for inspection
    init_str = encoder._fmt_constraints(init_constraints, "I(s_0): Initial state")
    trans_str = ""  # filled after first transition

    final_k = 0
    for k in range(max_k + 1):
        step_start = time.time()
        final_k = k

        if k > 0:
            trans = encoder.encode_transition(k - 1)
            solver.add(*trans)
            if k == 1:
                trans_str = encoder._fmt_constraints(
                    trans, "T(s_0, s_1): Transition relation (step 0 -> 1)")

        # Check: is the property violated at step k?
        solver.push()
        bad_k = encoder.encode_property_negation(spec, k)
        solver.add(bad_k)
        pending = encoder._flush_pending()
        if pending:
            solver.add(*pending)

        result = solver.check()
        step_time = time.time() - step_start

        if result == z3.sat:
            trace = encoder.extract_counterexample(solver.model(), k)
            sr = BmcStepResult(k=k, status="SAT", time_s=step_time,
                               counterexample=trace)
            step_results.append(sr)
            formulas = _capture_formulas(
                encoder, solver, spec, init_str, trans_str, k)
            solver.pop()
            return BmcResult(
                spec_text=spec_text, spec_kind=spec.kind,
                max_k=k, step_results=step_results,
                violated=True, violation_k=k, counterexample=trace,
                total_time_s=time.time() - start_time,
                formulas=formulas,
            )

        status = "UNSAT" if result == z3.unsat else "UNKNOWN"
        step_results.append(BmcStepResult(k=k, status=status, time_s=step_time))
        solver.pop()

    # Capture formulas at the last checked bound
    solver.push()
    bad_last = encoder.encode_property_negation(spec, final_k)
    solver.add(bad_last)
    pending = encoder._flush_pending()
    if pending:
        solver.add(*pending)
    formulas = _capture_formulas(
        encoder, solver, spec, init_str, trans_str, final_k)
    solver.pop()

    return BmcResult(
        spec_text=spec_text, spec_kind=spec.kind,
        max_k=max_k, step_results=step_results,
        violated=False, violation_k=None, counterexample=None,
        total_time_s=time.time() - start_time,
        formulas=formulas,
    )


def run_bmc_all_specs(
    model: SmvModel,
    max_k: int = 30,
    timeout_ms: int = 30000,
) -> list[BmcResult]:
    """Run BMC for all safety specs in the model."""
    results = []
    for spec in model.specs:
        if is_safety_spec(spec):
            results.append(run_bmc(model, spec, max_k, timeout_ms))
    return results


def is_safety_spec(spec: SpecDecl) -> bool:
    """Check if a spec is a safety property amenable to BMC.

    Supported forms: INVARSPEC p, CTLSPEC AG p, LTLSPEC G p
    where *p* contains no temporal operators.
    """
    if spec.kind == "INVARSPEC":
        return not _has_temporal(spec.expr)
    if spec.kind in ("CTLSPEC", "SPEC"):
        if isinstance(spec.expr, TemporalUnary) and spec.expr.op == "AG":
            return not _has_temporal(spec.expr.operand)
    if spec.kind == "LTLSPEC":
        if isinstance(spec.expr, TemporalUnary) and spec.expr.op == "G":
            return not _has_temporal(spec.expr.operand)
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _capture_formulas(
    encoder: BmcEncoder,
    solver: z3.Solver,
    spec: SpecDecl,
    init_str: str,
    trans_str: str,
    k: int,
) -> BmcFormulas:
    """Snapshot the SMT formulas at bound k for inspection."""
    # Property negation at step k
    body = _extract_safety_body(spec)
    prop_z3 = encoder.expr_to_z3(body, step=k)
    neg_z3 = z3.Not(prop_z3)
    encoder._flush_pending()  # discard any side-effects
    prop_str = (
        f"; Property: {expr_to_str(spec.expr)}\n"
        f"; Body at step {k}:\n"
        f"  P(s_{k})  = {prop_z3.sexpr()}\n"
        f"; Negation (bad state):\n"
        f"  !P(s_{k}) = {neg_z3.sexpr()}"
    )

    # Full solver assertions (human-readable)
    assertions = solver.assertions()
    full_lines = [f"; Full solver state at k={k} ({len(assertions)} assertions)"]
    for i, a in enumerate(assertions):
        full_lines.append(f"  ({i+1})  {a.sexpr()}")
    full_str = "\n".join(full_lines)

    # SMT-LIB2 export
    smt2 = solver.to_smt2()

    return BmcFormulas(
        encoding=encoder.encoding_legend(),
        init=init_str,
        transition=trans_str or "; (no transitions at k=0)",
        property_negation=prop_str,
        full_check=full_str,
        smt2=smt2,
    )


def _has_temporal(expr: Expr) -> bool:
    """Return True if *expr* contains any temporal operator."""
    if isinstance(expr, (TemporalUnary, TemporalBinary)):
        return True
    if isinstance(expr, UnaryOp):
        return _has_temporal(expr.operand)
    if isinstance(expr, BinOp):
        return _has_temporal(expr.left) or _has_temporal(expr.right)
    if isinstance(expr, CaseExpr):
        return any(_has_temporal(c) or _has_temporal(v) for c, v in expr.branches)
    if isinstance(expr, SetExpr):
        return any(_has_temporal(v) for v in expr.values)
    return False


def _extract_safety_body(spec: SpecDecl) -> Expr:
    """Extract the body *p* from INVARSPEC p / AG p / G p."""
    if spec.kind == "INVARSPEC":
        return spec.expr
    if isinstance(spec.expr, TemporalUnary) and spec.expr.op in ("AG", "G"):
        return spec.expr.operand
    return spec.expr


def _to_bool(e: z3.ExprRef) -> z3.BoolRef:
    """Coerce a Z3 expression to BoolRef if needed."""
    if z3.is_bool(e):
        return e
    # Int in boolean context: nonzero is true
    return e != z3.IntVal(0)


def _to_int(e: z3.ExprRef) -> z3.ArithRef:
    """Coerce a Z3 expression to ArithRef if needed."""
    if z3.is_int(e):
        return e
    if z3.is_bool(e):
        return z3.If(e, z3.IntVal(1), z3.IntVal(0))
    return e


def _coerce_pair(a: z3.ExprRef, b: z3.ExprRef):
    """Ensure two Z3 expressions have compatible sorts for If branches."""
    if a.sort() == b.sort():
        return a, b
    # Bool vs Int coercion
    if z3.is_bool(a) and z3.is_int(b):
        return z3.If(a, z3.IntVal(1), z3.IntVal(0)), b
    if z3.is_int(a) and z3.is_bool(b):
        return a, z3.If(b, z3.IntVal(1), z3.IntVal(0))
    return a, b


def _binop_to_z3(op: str, left: z3.ExprRef, right: z3.ExprRef) -> z3.ExprRef:
    """Translate a binary operator to its Z3 equivalent."""
    # Arithmetic operators (need int operands)
    if op in ("+", "-", "*", "/", "mod"):
        l, r = _to_int(left), _to_int(right)
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            return z3.If(r != 0, l / r, z3.IntVal(0))
        if op == "mod":
            return z3.If(r != 0, l % r, z3.IntVal(0))

    # Comparison operators (work on same-sort operands)
    if op in ("=", "!="):
        left, right = _coerce_pair(left, right)
        return (left == right) if op == "=" else (left != right)

    if op in (">", "<", ">=", "<="):
        l, r = _to_int(left), _to_int(right)
        if op == ">":
            return l > r
        if op == "<":
            return l < r
        if op == ">=":
            return l >= r
        return l <= r

    # Boolean operators
    if op == "&":
        return z3.And(_to_bool(left), _to_bool(right))
    if op == "|":
        return z3.Or(_to_bool(left), _to_bool(right))
    if op == "->":
        return z3.Implies(_to_bool(left), _to_bool(right))

    raise ValueError(f"Unknown binary op: {op}")
