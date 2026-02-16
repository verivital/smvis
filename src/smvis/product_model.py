"""Synchronous product of SmvModel and BuchiAutomaton.

The product is represented as a new SmvModel with an additional enum variable
`_buchi_q` tracking the Buchi state. This allows reuse of the entire existing
exploration pipeline (explicit_engine, cycle_analysis) unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
from smvis.smv_model import (
    SmvModel, VarDecl, EnumType, Expr,
    BoolLit, VarRef, BinOp, CaseExpr, SetExpr,
)
from smvis.ltl_buchi import BuchiAutomaton, BuchiTransition


BUCHI_VAR = "_buchi_q"
DEAD_STATE = "_dead"


@dataclass
class ProductInfo:
    original_model: SmvModel
    buchi: BuchiAutomaton
    buchi_var_name: str              # "_buchi_q"
    buchi_domain: list[str]          # ["q0", "q1", ..., "_dead"]
    accepting_states: list[str]      # ["q1"]
    product_model: SmvModel          # the composed model
    original_var_names: list[str]    # var names from original model
    spec: object                     # SpecDecl


def compose(model: SmvModel, buchi: BuchiAutomaton) -> ProductInfo:
    """Build the synchronous product M × A.

    The product model has:
    - All original variables with same init/next
    - A new enum variable _buchi_q tracking the Buchi state (+ _dead sink)
    - AP expressions added as DEFINEs for guard evaluation
    - Fairness = original fairness + Buchi acceptance constraints
    """
    product = SmvModel()

    # --- Copy original variables ---
    for name, vd in model.variables.items():
        product.variables[name] = vd

    # --- Check if _dead state is needed ---
    needs_dead = _needs_dead_state(buchi)

    # --- Add Buchi state variable ---
    buchi_states = [s.name for s in buchi.states]
    domain = list(buchi_states)
    if needs_dead:
        domain.append(DEAD_STATE)
    product.variables[BUCHI_VAR] = VarDecl(BUCHI_VAR, EnumType(domain))

    # --- Copy original inits ---
    for name, expr in model.inits.items():
        product.inits[name] = expr

    # --- Init Buchi state ---
    product.inits[BUCHI_VAR] = VarRef(buchi.initial)

    # --- Copy original nexts ---
    for name, expr in model.nexts.items():
        product.nexts[name] = expr

    # --- Copy original defines + add AP defines ---
    for name, expr in model.defines.items():
        product.defines[name] = expr
    for ap_name, ap_expr in buchi.ap_exprs.items():
        product.defines[ap_name] = ap_expr

    # --- Build next(_buchi_q) ---
    product.nexts[BUCHI_VAR] = _build_buchi_next(buchi, needs_dead)

    # --- Fairness: original + Buchi acceptance ---
    product.fairness = list(model.fairness)
    for acc_name in buchi.accepting:
        product.fairness.append(BinOp("=", VarRef(BUCHI_VAR), VarRef(acc_name)))

    return ProductInfo(
        original_model=model,
        buchi=buchi,
        buchi_var_name=BUCHI_VAR,
        buchi_domain=domain,
        accepting_states=list(buchi.accepting),
        product_model=product,
        original_var_names=list(model.variables.keys()),
        spec=buchi.original_spec,
    )


def _needs_dead_state(buchi: BuchiAutomaton) -> bool:
    """Check if any Buchi state lacks an unconditional (TRUE) outgoing transition.

    If a state has only conditional guards, the Buchi can "get stuck" when no
    guard fires. The _dead sink state handles this correctly.
    """
    trans_by_src: dict[str, list[tuple[str, Expr]]] = {}
    for t in buchi.transitions:
        trans_by_src.setdefault(t.src, []).append((t.dst, t.guard))

    for state in buchi.states:
        trans = trans_by_src.get(state.name, [])
        has_unconditional = any(
            isinstance(g, BoolLit) and g.value for _, g in trans
        )
        if not has_unconditional:
            return True
    return False


def _build_buchi_next(buchi: BuchiAutomaton, needs_dead: bool) -> CaseExpr:
    """Build the case expression for next(_buchi_q).

    Groups transitions by source state and handles non-determinism:
    when multiple transitions from the same state can fire simultaneously,
    the destination set includes all of them (via SetExpr).

    When no guard fires from a state, transitions to _dead (non-accepting sink).
    """
    # Group transitions by source
    trans_by_src: dict[str, list[tuple[str, Expr]]] = {}
    for t in buchi.transitions:
        trans_by_src.setdefault(t.src, []).append((t.dst, t.guard))

    outer_branches: list[tuple[Expr, Expr]] = []

    for state in buchi.states:
        src = state.name
        trans = trans_by_src.get(src, [])
        if not trans:
            continue

        inner = _build_inner_case(trans, needs_dead)
        src_cond = BinOp("=", VarRef(BUCHI_VAR), VarRef(src))
        outer_branches.append((src_cond, inner))

    # Fallback: _dead state self-loops (or stays in current state)
    if needs_dead:
        outer_branches.append((BoolLit(True), VarRef(DEAD_STATE)))
    else:
        outer_branches.append((BoolLit(True), VarRef(BUCHI_VAR)))
    return CaseExpr(outer_branches)


def _build_inner_case(trans: list[tuple[str, Expr]], needs_dead: bool) -> Expr:
    """Build inner expression for transitions from a single source state.

    Handles non-determinism by enumerating guard combinations.
    For n conditional guards, produces 2^n branches (n is small: typically 0-2).

    When unconditional_dsts is empty and no conditional guard fires,
    transitions to _dead (the run has no valid continuation).
    """
    # Separate unconditional (TRUE guard) from conditional
    unconditional_dsts: set[str] = set()
    conditional: list[tuple[str, Expr]] = []

    for dst, guard in trans:
        if isinstance(guard, BoolLit) and guard.value:
            unconditional_dsts.add(dst)
        else:
            conditional.append((dst, guard))

    if not conditional:
        # All transitions are unconditional
        return _make_dst_expr(unconditional_dsts, needs_dead)

    n = len(conditional)
    branches: list[tuple[Expr, Expr]] = []

    # Enumerate guard combinations: most specific (most guards true) first
    for mask in range(2**n - 1, 0, -1):
        dsts = set(unconditional_dsts)
        guard_parts: list[Expr] = []
        for i in range(n):
            if mask & (1 << i):
                dsts.add(conditional[i][0])
                guard_parts.append(conditional[i][1])

        # Build guard conjunction
        guard = guard_parts[0]
        for g in guard_parts[1:]:
            guard = BinOp("&", guard, g)

        branches.append((guard, _make_dst_expr(dsts, needs_dead)))

    # mask = 0: no conditional guards fire — only unconditional destinations
    branches.append((BoolLit(True), _make_dst_expr(unconditional_dsts, needs_dead)))

    return CaseExpr(branches)


def _make_dst_expr(dsts: set[str], needs_dead: bool) -> Expr:
    """Create expression for a set of destination states.

    When dsts is empty and dead state exists, go to _dead (run has no valid
    Buchi continuation). This correctly models Buchi automaton semantics where
    an absent transition means the run fails.
    """
    if not dsts:
        if needs_dead:
            return VarRef(DEAD_STATE)
        return VarRef(BUCHI_VAR)
    dsts_sorted = sorted(dsts)
    if len(dsts_sorted) == 1:
        return VarRef(dsts_sorted[0])
    return SetExpr([VarRef(d) for d in dsts_sorted])
