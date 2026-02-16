"""LTL-to-Buchi automaton conversion via pattern matching + Spot fallback."""
from __future__ import annotations
import logging
import subprocess
from dataclasses import dataclass, field
from smvis.smv_model import (
    Expr, BinOp, UnaryOp, BoolLit, VarRef, IntLit,
    TemporalUnary, TemporalBinary, SpecDecl, expr_to_str,
)

log = logging.getLogger("smvis")


# ============================================================
# Data Structures
# ============================================================

@dataclass
class BuchiState:
    name: str
    accepting: bool


@dataclass
class BuchiTransition:
    src: str
    dst: str
    guard: Expr


@dataclass
class BuchiAutomaton:
    states: list[BuchiState]
    initial: str
    transitions: list[BuchiTransition]
    accepting: list[str]
    ap_exprs: dict[str, Expr]  # "p0" -> actual model expression
    formula_text: str          # human-readable negated formula
    original_spec: SpecDecl


class UnsupportedLTLPattern(Exception):
    """Raised when a formula doesn't match any known Buchi pattern."""
    pass


# ============================================================
# LTL Negation
# ============================================================

def negate_ltl(expr: Expr) -> Expr:
    """Negate an LTL formula, pushing negation inward.

    Uses temporal duals and De Morgan's laws:
      !G p  = F !p       !F p  = G !p       !X p  = X !p
      !(p & q) = !p | !q   !(p | q) = !p & !q
      !(p -> q) = p & !q   !!p = p
    """
    return simplify_ltl(_negate_inner(expr))


def _negate_inner(expr: Expr) -> Expr:
    if isinstance(expr, BoolLit):
        return BoolLit(not expr.value)
    if isinstance(expr, (VarRef, IntLit)):
        return UnaryOp("!", expr)
    if isinstance(expr, UnaryOp) and expr.op == "!":
        return expr.operand  # double negation
    if isinstance(expr, UnaryOp) and expr.op == "-":
        return UnaryOp("!", expr)
    if isinstance(expr, BinOp):
        if expr.op == "&":
            return BinOp("|", _negate_inner(expr.left), _negate_inner(expr.right))
        if expr.op == "|":
            return BinOp("&", _negate_inner(expr.left), _negate_inner(expr.right))
        if expr.op == "->":
            return BinOp("&", expr.left, _negate_inner(expr.right))
        # Comparison operators: wrap in !
        if expr.op in ("=", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "mod"):
            return UnaryOp("!", expr)
    if isinstance(expr, TemporalUnary):
        if expr.op == "G":
            return TemporalUnary("F", _negate_inner(expr.operand))
        if expr.op == "F":
            return TemporalUnary("G", _negate_inner(expr.operand))
        if expr.op == "X":
            return TemporalUnary("X", _negate_inner(expr.operand))
        # CTL operators — just wrap in negation
        return UnaryOp("!", expr)
    if isinstance(expr, TemporalBinary):
        if expr.op == "U":
            # !(p U q) = (!q) U (!p & !q) | G(!q)
            # Simpler: wrap in negation and let simplify handle it
            return UnaryOp("!", expr)
    return UnaryOp("!", expr)


def simplify_ltl(expr: Expr) -> Expr:
    """Simplify an LTL formula: double negation, TRUE/FALSE propagation."""
    if isinstance(expr, UnaryOp) and expr.op == "!":
        inner = simplify_ltl(expr.operand)
        if isinstance(inner, BoolLit):
            return BoolLit(not inner.value)
        if isinstance(inner, UnaryOp) and inner.op == "!":
            return inner.operand  # !!p = p
        return UnaryOp("!", inner)
    if isinstance(expr, BinOp):
        left = simplify_ltl(expr.left)
        right = simplify_ltl(expr.right)
        if expr.op == "&":
            if isinstance(left, BoolLit):
                return BoolLit(False) if not left.value else right
            if isinstance(right, BoolLit):
                return BoolLit(False) if not right.value else left
        if expr.op == "|":
            if isinstance(left, BoolLit):
                return BoolLit(True) if left.value else right
            if isinstance(right, BoolLit):
                return BoolLit(True) if right.value else left
        if expr.op == "->":
            if isinstance(left, BoolLit):
                return BoolLit(True) if not left.value else right
        return BinOp(expr.op, left, right)
    if isinstance(expr, TemporalUnary):
        inner = simplify_ltl(expr.operand)
        return TemporalUnary(expr.op, inner)
    if isinstance(expr, TemporalBinary):
        left = simplify_ltl(expr.left)
        right = simplify_ltl(expr.right)
        return TemporalBinary(expr.op, left, right)
    return expr


# ============================================================
# Atomic Proposition Extraction
# ============================================================

def _has_temporal(expr: Expr) -> bool:
    """Check if expression contains any temporal operators."""
    if isinstance(expr, TemporalUnary):
        return True
    if isinstance(expr, TemporalBinary):
        return True
    if isinstance(expr, BinOp):
        return _has_temporal(expr.left) or _has_temporal(expr.right)
    if isinstance(expr, UnaryOp):
        return _has_temporal(expr.operand)
    return False


def _extract_aps(expr: Expr, aps: dict[str, Expr]) -> str:
    """Extract atomic propositions, returning a placeholder name.

    If the expression is non-temporal, assign it a placeholder name (p0, p1, ...)
    and store it in aps. Returns the placeholder name.
    """
    # Check if this exact expression is already in aps
    for name, existing in aps.items():
        if _expr_eq(existing, expr):
            return name
    name = f"p{len(aps)}"
    aps[name] = expr
    return name


def _expr_eq(a: Expr, b: Expr) -> bool:
    """Structural equality of expressions."""
    if type(a) != type(b):
        return False
    if isinstance(a, BoolLit):
        return a.value == b.value
    if isinstance(a, IntLit):
        return a.value == b.value
    if isinstance(a, VarRef):
        return a.name == b.name
    if isinstance(a, UnaryOp):
        return a.op == b.op and _expr_eq(a.operand, b.operand)
    if isinstance(a, BinOp):
        return a.op == b.op and _expr_eq(a.left, b.left) and _expr_eq(a.right, b.right)
    if isinstance(a, TemporalUnary):
        return a.op == b.op and _expr_eq(a.operand, b.operand)
    if isinstance(a, TemporalBinary):
        return a.op == b.op and _expr_eq(a.left, b.left) and _expr_eq(a.right, b.right)
    return False


# ============================================================
# Pattern Matching
# ============================================================

def _match_and_build(negated: Expr) -> BuchiAutomaton | None:
    """Try to match the negated formula against known Buchi patterns.

    Returns BuchiAutomaton or None if no pattern matches.
    """
    aps: dict[str, Expr] = {}
    result = _try_patterns(negated, aps)
    if result is None:
        return None
    states, initial, transitions, accepting, formula_text = result
    return BuchiAutomaton(
        states=[BuchiState(name=n, accepting=(n in accepting)) for n, _ in states],
        initial=initial,
        transitions=[BuchiTransition(s, d, g) for s, d, g in transitions],
        accepting=list(accepting),
        ap_exprs=aps,
        formula_text=formula_text,
        original_spec=SpecDecl(kind="LTLSPEC", expr=BoolLit(True)),  # placeholder
    )


def _try_patterns(expr: Expr, aps: dict[str, Expr]):
    """Match against known patterns. Returns (states, initial, transitions, accepting, text) or None."""

    # Pattern: G(p) — "Always p" (negated is just G(p))
    # Buchi: q0(acc) self-loop on p
    if isinstance(expr, TemporalUnary) and expr.op == "G":
        inner = expr.operand
        if not _has_temporal(inner):
            p_name = _extract_aps(inner, aps)
            p_expr = VarRef(p_name)
            return (
                [("q0", True)],
                "q0",
                [("q0", "q0", p_expr)],
                {"q0"},
                f"G({expr_to_str(inner)})",
            )

    # Pattern: F(p) — "Eventually p"
    # Buchi: q0 -> q0 [t], q0 -> q1 [p], q1(acc) -> q1 [t]
    if isinstance(expr, TemporalUnary) and expr.op == "F":
        inner = expr.operand
        if not _has_temporal(inner):
            p_name = _extract_aps(inner, aps)
            p_expr = VarRef(p_name)
            return (
                [("q0", False), ("q1", True)],
                "q0",
                [
                    ("q0", "q0", BoolLit(True)),
                    ("q0", "q1", p_expr),
                    ("q1", "q1", BoolLit(True)),
                ],
                {"q1"},
                f"F({expr_to_str(inner)})",
            )

    # Pattern: FG(p) — "Eventually always p"
    # Buchi: q0 -> q0 [t], q0 -> q1 [p], q1(acc) -> q1 [p]
    if isinstance(expr, TemporalUnary) and expr.op == "F":
        if isinstance(expr.operand, TemporalUnary) and expr.operand.op == "G":
            inner = expr.operand.operand
            if not _has_temporal(inner):
                p_name = _extract_aps(inner, aps)
                p_expr = VarRef(p_name)
                return (
                    [("q0", False), ("q1", True)],
                    "q0",
                    [
                        ("q0", "q0", BoolLit(True)),
                        ("q0", "q1", p_expr),
                        ("q1", "q1", p_expr),
                    ],
                    {"q1"},
                    f"FG({expr_to_str(inner)})",
                )

    # Pattern: GF(p) — "Always eventually p" (infinitely often)
    # Buchi: q0 -> q0 [t], q0 -> q1 [p], q1(acc) -> q0 [t]
    if isinstance(expr, TemporalUnary) and expr.op == "G":
        if isinstance(expr.operand, TemporalUnary) and expr.operand.op == "F":
            inner = expr.operand.operand
            if not _has_temporal(inner):
                p_name = _extract_aps(inner, aps)
                p_expr = VarRef(p_name)
                return (
                    [("q0", False), ("q1", True)],
                    "q0",
                    [
                        ("q0", "q0", BoolLit(True)),
                        ("q0", "q1", p_expr),
                        ("q1", "q0", BoolLit(True)),
                    ],
                    {"q1"},
                    f"GF({expr_to_str(inner)})",
                )

    # Pattern: F(p & G(q)) — "Eventually p and henceforth q"
    # Used for negation of G(p -> F !q) = F(p & G(q))
    # Buchi: q0 -> q0 [t], q0 -> q1 [p & q], q1(acc) -> q1 [q]
    if isinstance(expr, TemporalUnary) and expr.op == "F":
        if isinstance(expr.operand, BinOp) and expr.operand.op == "&":
            left, right = expr.operand.left, expr.operand.right
            # Try both orderings: F(p & G(q)) or F(G(q) & p)
            for state_part, temporal_part in [(left, right), (right, left)]:
                if isinstance(temporal_part, TemporalUnary) and temporal_part.op == "G":
                    if not _has_temporal(state_part) and not _has_temporal(temporal_part.operand):
                        p_name = _extract_aps(state_part, aps)
                        q_name = _extract_aps(temporal_part.operand, aps)
                        p_expr = VarRef(p_name)
                        q_expr = VarRef(q_name)
                        return (
                            [("q0", False), ("q1", True)],
                            "q0",
                            [
                                ("q0", "q0", BoolLit(True)),
                                ("q0", "q1", BinOp("&", p_expr, q_expr)),
                                ("q1", "q1", q_expr),
                            ],
                            {"q1"},
                            f"F({expr_to_str(state_part)} & G({expr_to_str(temporal_part.operand)}))",
                        )

    # Pattern: F(p & F(q)) — "Eventually p, then eventually q"
    # Used for negation of G(p -> G(q)):  ¬G(p->Gq) = F(p & F(!q))
    # Buchi: q0 -> q0 [t], q0 -> q1 [p], q1 -> q1 [t], q1 -> q2 [q], q2(acc) -> q2 [t]
    if isinstance(expr, TemporalUnary) and expr.op == "F":
        if isinstance(expr.operand, BinOp) and expr.operand.op == "&":
            left, right = expr.operand.left, expr.operand.right
            # Try both orderings: F(p & F(q)) or F(F(q) & p)
            for state_part, temporal_part in [(left, right), (right, left)]:
                if isinstance(temporal_part, TemporalUnary) and temporal_part.op == "F":
                    if not _has_temporal(state_part) and not _has_temporal(temporal_part.operand):
                        p_name = _extract_aps(state_part, aps)
                        q_name = _extract_aps(temporal_part.operand, aps)
                        p_expr = VarRef(p_name)
                        q_expr = VarRef(q_name)
                        return (
                            [("q0", False), ("q1", False), ("q2", True)],
                            "q0",
                            [
                                ("q0", "q0", BoolLit(True)),
                                ("q0", "q1", p_expr),
                                ("q1", "q1", BoolLit(True)),
                                ("q1", "q2", q_expr),
                                ("q2", "q2", BoolLit(True)),
                            ],
                            {"q2"},
                            f"F({expr_to_str(state_part)} & F({expr_to_str(temporal_part.operand)}))",
                        )

    # Pattern: p & G(q) — "p holds initially and q always holds"
    # Used for negation of (cond -> F result):  ¬(p -> Fq) = p & G(!q)
    # Buchi: q0 -> q1 [p & q], q1(acc) -> q1 [q]
    if isinstance(expr, BinOp) and expr.op == "&":
        left, right = expr.left, expr.right
        # Try both orderings: p & G(q) or G(q) & p
        for state_part, temporal_part in [(left, right), (right, left)]:
            if isinstance(temporal_part, TemporalUnary) and temporal_part.op == "G":
                if not _has_temporal(state_part) and not _has_temporal(temporal_part.operand):
                    p_name = _extract_aps(state_part, aps)
                    q_name = _extract_aps(temporal_part.operand, aps)
                    p_expr = VarRef(p_name)
                    q_expr = VarRef(q_name)
                    return (
                        [("q0", False), ("q1", True)],
                        "q0",
                        [
                            ("q0", "q1", BinOp("&", p_expr, q_expr)),
                            ("q1", "q1", q_expr),
                        ],
                        {"q1"},
                        f"({expr_to_str(state_part)} & G({expr_to_str(temporal_part.operand)}))",
                    )

    # Pattern: G(p -> G(q)) — "Once p, then always q"
    # Buchi: q0(acc) -> q0 [!p | q], q0 -> q1 [p & !q], q1 -> q1 [t]
    # Only accepting state is q0; q1 is a "trap" (non-accepting sink)
    if isinstance(expr, TemporalUnary) and expr.op == "G":
        if isinstance(expr.operand, BinOp) and expr.operand.op == "->":
            impl_left = expr.operand.left
            impl_right = expr.operand.right
            if isinstance(impl_right, TemporalUnary) and impl_right.op == "G":
                if not _has_temporal(impl_left) and not _has_temporal(impl_right.operand):
                    p_name = _extract_aps(impl_left, aps)
                    q_name = _extract_aps(impl_right.operand, aps)
                    p_expr = VarRef(p_name)
                    q_expr = VarRef(q_name)
                    return (
                        [("q0", True), ("q1", False)],
                        "q0",
                        [
                            ("q0", "q0", BinOp("|", UnaryOp("!", p_expr), q_expr)),
                            ("q0", "q1", BinOp("&", p_expr, UnaryOp("!", q_expr))),
                            ("q1", "q1", BoolLit(True)),
                        ],
                        {"q0"},
                        f"G({expr_to_str(impl_left)} -> G({expr_to_str(impl_right.operand)}))",
                    )

    # Pattern: FG(p) | FG(q) — "eventually always p, or eventually always q"
    # Used for negation of GF(a) & GF(b): ¬(GF(a)&GF(b)) = FG(!a) | FG(!b)
    # Buchi: q0 -> q0 [t], q0 -> q1 [p], q0 -> q2 [q], q1(acc) -> q1 [p], q2(acc) -> q2 [q]
    if isinstance(expr, BinOp) and expr.op == "|":
        left, right = expr.left, expr.right
        if (isinstance(left, TemporalUnary) and left.op == "F"
                and isinstance(left.operand, TemporalUnary) and left.operand.op == "G"
                and isinstance(right, TemporalUnary) and right.op == "F"
                and isinstance(right.operand, TemporalUnary) and right.operand.op == "G"):
            left_inner = left.operand.operand
            right_inner = right.operand.operand
            if not _has_temporal(left_inner) and not _has_temporal(right_inner):
                p_name = _extract_aps(left_inner, aps)
                q_name = _extract_aps(right_inner, aps)
                p_expr = VarRef(p_name)
                q_expr = VarRef(q_name)
                return (
                    [("q0", False), ("q1", True), ("q2", True)],
                    "q0",
                    [
                        ("q0", "q0", BoolLit(True)),
                        ("q0", "q1", p_expr),
                        ("q0", "q2", q_expr),
                        ("q1", "q1", p_expr),
                        ("q2", "q2", q_expr),
                    ],
                    {"q1", "q2"},
                    f"FG({expr_to_str(left_inner)}) | FG({expr_to_str(right_inner)})",
                )

    return None


# ============================================================
# HOA Format Parser
# ============================================================

def parse_hoa(text: str, ap_exprs: dict[str, Expr]) -> BuchiAutomaton:
    """Parse HOA (Hanoi Omega-Automata) format into BuchiAutomaton.

    Args:
        text: HOA format string from ltl2tgba -B -S
        ap_exprs: mapping from AP names to model expressions
    """
    lines = text.strip().splitlines()
    n_states = 0
    start = 0
    ap_names: list[str] = []
    accepting_sets: set[int] = set()
    states: list[BuchiState] = []
    transitions: list[BuchiTransition] = []
    state_accepting: dict[int, bool] = {}
    in_body = False
    current_state: int | None = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("HOA:") or line.startswith("name:"):
            continue
        if line.startswith("States:"):
            n_states = int(line.split(":")[1].strip())
        elif line.startswith("Start:"):
            start = int(line.split(":")[1].strip())
        elif line.startswith("AP:"):
            # AP: 2 "a" "b"
            parts = line.split('"')
            ap_names = [parts[i] for i in range(1, len(parts), 2)]
        elif line.startswith("Acceptance:"):
            # Parse Inf(0) style - extract set numbers
            import re
            for m in re.finditer(r'Inf\((\d+)\)', line):
                accepting_sets.add(int(m.group(1)))
        elif line == "--BODY--":
            in_body = True
        elif line == "--END--":
            break
        elif in_body and line.startswith("State:"):
            # State: 0 {0}  or  State: 0
            parts = line[6:].strip().split()
            current_state = int(parts[0])
            # Check for acceptance marks {0} {1}
            has_acc = False
            for p in parts[1:]:
                if p.startswith("{"):
                    acc_marks = p.strip("{}")
                    if acc_marks:
                        for m in acc_marks.split(","):
                            if int(m.strip()) in accepting_sets:
                                has_acc = True
            state_accepting[current_state] = has_acc
        elif in_body and line.startswith("[") and current_state is not None:
            # [guard] destination  or  [guard] destination {acc}
            bracket_end = line.index("]")
            guard_str = line[1:bracket_end]
            rest = line[bracket_end + 1:].strip().split()
            dst = int(rest[0])
            # Check edge-based acceptance
            for p in rest[1:]:
                if p.startswith("{"):
                    acc_marks = p.strip("{}")
                    if acc_marks:
                        for m in acc_marks.split(","):
                            if int(m.strip()) in accepting_sets:
                                state_accepting[current_state] = True

            guard = _parse_hoa_guard(guard_str, ap_names, ap_exprs)
            transitions.append(BuchiTransition(
                src=f"q{current_state}", dst=f"q{dst}", guard=guard
            ))

    # Build states
    accepting_state_names: list[str] = []
    for i in range(n_states):
        name = f"q{i}"
        acc = state_accepting.get(i, False)
        states.append(BuchiState(name=name, accepting=acc))
        if acc:
            accepting_state_names.append(name)

    return BuchiAutomaton(
        states=states,
        initial=f"q{start}",
        transitions=transitions,
        accepting=accepting_state_names,
        ap_exprs=ap_exprs,
        formula_text="(from HOA)",
        original_spec=SpecDecl(kind="LTLSPEC", expr=BoolLit(True)),
    )


def _parse_hoa_guard(guard_str: str, ap_names: list[str],
                      ap_exprs: dict[str, Expr]) -> Expr:
    """Parse a HOA guard expression like '0 & !1' or 't'."""
    guard_str = guard_str.strip()
    if guard_str == "t":
        return BoolLit(True)
    if guard_str == "f":
        return BoolLit(False)

    # Tokenize: split on & and |, handle ! and parentheses
    # Simple recursive descent for HOA boolean formulas
    return _parse_hoa_or(guard_str, ap_names, ap_exprs)


def _parse_hoa_or(s: str, ap_names: list[str], ap_exprs: dict[str, Expr]) -> Expr:
    """Parse OR-level of HOA guard."""
    # Split on top-level |
    parts = _split_top_level(s, "|")
    if len(parts) == 1:
        return _parse_hoa_and(parts[0], ap_names, ap_exprs)
    result = _parse_hoa_and(parts[0], ap_names, ap_exprs)
    for part in parts[1:]:
        result = BinOp("|", result, _parse_hoa_and(part, ap_names, ap_exprs))
    return result


def _parse_hoa_and(s: str, ap_names: list[str], ap_exprs: dict[str, Expr]) -> Expr:
    """Parse AND-level of HOA guard."""
    parts = _split_top_level(s, "&")
    if len(parts) == 1:
        return _parse_hoa_atom(parts[0], ap_names, ap_exprs)
    result = _parse_hoa_atom(parts[0], ap_names, ap_exprs)
    for part in parts[1:]:
        result = BinOp("&", result, _parse_hoa_atom(part, ap_names, ap_exprs))
    return result


def _parse_hoa_atom(s: str, ap_names: list[str], ap_exprs: dict[str, Expr]) -> Expr:
    """Parse atom-level of HOA guard."""
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        return _parse_hoa_or(s[1:-1], ap_names, ap_exprs)
    if s.startswith("!"):
        return UnaryOp("!", _parse_hoa_atom(s[1:], ap_names, ap_exprs))
    if s == "t":
        return BoolLit(True)
    if s == "f":
        return BoolLit(False)
    # Must be an AP index
    idx = int(s)
    ap_name = ap_names[idx] if idx < len(ap_names) else f"ap{idx}"
    # Map HOA AP name to our ap_exprs key
    if ap_name in ap_exprs:
        return VarRef(ap_name)
    # Try matching by position
    keys = list(ap_exprs.keys())
    if idx < len(keys):
        return VarRef(keys[idx])
    return VarRef(ap_name)


def _split_top_level(s: str, op: str) -> list[str]:
    """Split string on operator at top level (respecting parentheses)."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == op and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    return parts


# ============================================================
# Spot Fallback
# ============================================================

_spot_check_done = False
_spot_is_available = False


def _spot_available() -> bool:
    """Check if Spot's ltl2tgba is available via WSL."""
    global _spot_check_done, _spot_is_available
    if _spot_check_done:
        return _spot_is_available
    _spot_check_done = True
    try:
        result = subprocess.run(
            ["wsl", "which", "ltl2tgba"],
            capture_output=True, text=True, timeout=5,
        )
        _spot_is_available = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _spot_is_available = False
    log.info(f"Spot availability: {_spot_is_available}")
    return _spot_is_available


def _ltl_to_buchi_via_spot(formula_str: str, ap_exprs: dict[str, Expr]) -> BuchiAutomaton:
    """Call ltl2tgba via WSL, parse HOA output."""
    try:
        result = subprocess.run(
            ["wsl", "ltl2tgba", "-B", "-S", "-d", formula_str],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise UnsupportedLTLPattern(
                f"Spot ltl2tgba failed: {result.stderr.strip()}"
            )
        return parse_hoa(result.stdout, ap_exprs)
    except FileNotFoundError:
        raise UnsupportedLTLPattern("WSL not available for Spot fallback")
    except subprocess.TimeoutExpired:
        raise UnsupportedLTLPattern("Spot ltl2tgba timed out")


# ============================================================
# Main Entry Points
# ============================================================

def ltl_to_buchi(negated: Expr) -> BuchiAutomaton:
    """Convert a negated LTL formula to a Buchi automaton.

    Strategy:
    1. Try pattern matching against known Buchi templates
    2. If no match, try Spot via WSL subprocess
    3. If Spot unavailable, raise UnsupportedLTLPattern
    """
    buchi = _match_and_build(negated)
    if buchi is not None:
        return buchi

    # Fallback: try Spot
    formula_str = expr_to_str(negated)
    if _spot_available():
        log.info(f"Pattern matching failed, trying Spot for: {formula_str}")
        aps: dict[str, Expr] = {}
        # Extract APs from the formula for Spot
        _collect_aps_for_spot(negated, aps)
        buchi = _ltl_to_buchi_via_spot(formula_str, aps)
        buchi.formula_text = formula_str
        return buchi

    supported = [
        "G(p)", "F(p)", "GF(p)", "FG(p)", "F(p & G(q))", "F(p & F(q))",
        "p & G(q)", "G(p -> G(q))", "FG(p) | FG(q)",
    ]
    raise UnsupportedLTLPattern(
        f"No Buchi pattern matches '{formula_str}'. "
        f"Supported negated patterns: {', '.join(supported)}. "
        f"Install Spot in WSL for arbitrary LTL support."
    )


def _collect_aps_for_spot(expr: Expr, aps: dict[str, Expr]):
    """Collect atomic propositions from an LTL formula for Spot."""
    if isinstance(expr, (BoolLit, IntLit)):
        return
    if isinstance(expr, VarRef):
        if expr.name not in aps:
            aps[expr.name] = expr
        return
    if isinstance(expr, UnaryOp):
        _collect_aps_for_spot(expr.operand, aps)
    elif isinstance(expr, BinOp):
        _collect_aps_for_spot(expr.left, aps)
        _collect_aps_for_spot(expr.right, aps)
    elif isinstance(expr, TemporalUnary):
        _collect_aps_for_spot(expr.operand, aps)
    elif isinstance(expr, TemporalBinary):
        _collect_aps_for_spot(expr.left, aps)
        _collect_aps_for_spot(expr.right, aps)


def build_buchi_for_spec(spec: SpecDecl) -> BuchiAutomaton:
    """Full pipeline: negate the LTL formula, then convert to Buchi.

    Args:
        spec: An LTLSPEC declaration from the model.

    Returns:
        BuchiAutomaton for the negated formula.
    """
    negated = negate_ltl(spec.expr)
    buchi = ltl_to_buchi(negated)
    buchi.original_spec = spec
    buchi.formula_text = expr_to_str(negated)
    return buchi
