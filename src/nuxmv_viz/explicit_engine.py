"""Explicit-state engine: enumerate states, compute transitions, BFS reachability."""
from __future__ import annotations
import itertools
from collections import deque
from dataclasses import dataclass, field
from nuxmv_viz.smv_model import (
    SmvModel, VarDecl, BoolType, EnumType, RangeType,
    Expr, IntLit, BoolLit, VarRef, NextRef, BinOp, UnaryOp,
    CaseExpr, SetExpr, get_domain,
)

# A state is a tuple of values (hashable), with a fixed variable ordering.
State = tuple
# A state dict maps variable names to their current values.
StateDict = dict[str, object]


@dataclass
class ExplicitResult:
    """Result of explicit-state exploration."""
    var_names: list[str]
    total_states: int
    initial_states: list[State]
    reachable_states: set[State]
    transitions: list[tuple[State, State]]  # (src, dst) pairs
    bfs_layers: list[set[State]]  # states discovered at each BFS depth
    state_to_dict: dict[State, StateDict] = field(default_factory=dict)


def evaluate(expr: Expr, state: StateDict,
             next_state: StateDict | None = None,
             defines: dict[str, Expr] | None = None) -> object:
    """Evaluate an expression AST node given current state (and optional next state)."""
    if defines is None:
        defines = {}

    if isinstance(expr, IntLit):
        return expr.value
    elif isinstance(expr, BoolLit):
        return expr.value
    elif isinstance(expr, VarRef):
        name = expr.name
        # Check DEFINE first
        if name in defines:
            return evaluate(defines[name], state, next_state, defines)
        # Check state variables
        if name in state:
            return state[name]
        # Must be an enum constant (e.g., "idle", "off", "l1")
        return name
    elif isinstance(expr, NextRef):
        if next_state is not None and expr.name in next_state:
            return next_state[expr.name]
        raise ValueError(f"next({expr.name}) referenced but not yet computed")
    elif isinstance(expr, UnaryOp):
        val = evaluate(expr.operand, state, next_state, defines)
        if expr.op == "!":
            return not val
        elif expr.op == "-":
            return -val
        raise ValueError(f"Unknown unary op: {expr.op}")
    elif isinstance(expr, BinOp):
        left = evaluate(expr.left, state, next_state, defines)
        right = evaluate(expr.right, state, next_state, defines)
        op = expr.op
        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        elif op == "*":
            return left * right
        elif op == "/":
            return left // right if right != 0 else 0
        elif op == "mod":
            return left % right if right != 0 else 0
        elif op == "=":
            return left == right
        elif op == "!=":
            return left != right
        elif op == ">":
            return left > right
        elif op == "<":
            return left < right
        elif op == ">=":
            return left >= right
        elif op == "<=":
            return left <= right
        elif op == "&":
            return left and right
        elif op == "|":
            return left or right
        elif op == "->":
            return (not left) or right
        raise ValueError(f"Unknown binary op: {op}")
    elif isinstance(expr, CaseExpr):
        for cond, val in expr.branches:
            cond_result = evaluate(cond, state, next_state, defines)
            if cond_result:
                result = evaluate(val, state, next_state, defines)
                return result
        raise ValueError("No case branch matched (missing TRUE default?)")
    elif isinstance(expr, SetExpr):
        # Return a list of possible values (non-determinism)
        return [evaluate(v, state, next_state, defines) for v in expr.values]
    else:
        raise ValueError(f"Cannot evaluate expression type: {type(expr)}")


def _find_next_refs(expr: Expr) -> set[str]:
    """Find all next(var) references in an expression (for dependency analysis)."""
    refs = set()
    if isinstance(expr, NextRef):
        refs.add(expr.name)
    elif isinstance(expr, BinOp):
        refs |= _find_next_refs(expr.left)
        refs |= _find_next_refs(expr.right)
    elif isinstance(expr, UnaryOp):
        refs |= _find_next_refs(expr.operand)
    elif isinstance(expr, CaseExpr):
        for cond, val in expr.branches:
            refs |= _find_next_refs(cond)
            refs |= _find_next_refs(val)
    elif isinstance(expr, SetExpr):
        for v in expr.values:
            refs |= _find_next_refs(v)
    return refs


def _topological_sort(deps: dict[str, set[str]], all_vars: list[str]) -> list[str]:
    """Topological sort of variables based on next-state dependencies."""
    in_degree = {v: 0 for v in all_vars}
    for v, d in deps.items():
        for dep in d:
            if dep in in_degree:
                in_degree[v] = in_degree.get(v, 0)  # ensure v exists

    # Build adjacency: dep -> v means v depends on dep, so dep comes first
    adj = {v: [] for v in all_vars}
    for v, d in deps.items():
        for dep in d:
            if dep in adj:
                adj[dep].append(v)
                in_degree[v] += 1

    # Reset in_degree properly
    in_degree = {v: 0 for v in all_vars}
    for v, d in deps.items():
        for dep in d:
            if dep in in_degree:
                in_degree[v] += 1

    queue = deque(v for v in all_vars if in_degree[v] == 0)
    result = []
    while queue:
        v = queue.popleft()
        result.append(v)
        for u in adj.get(v, []):
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append(u)

    # Any remaining (cycle) just append in original order
    for v in all_vars:
        if v not in result:
            result.append(v)
    return result


def compute_dep_order(model: SmvModel) -> list[str]:
    """Compute the evaluation order for next-state variables."""
    var_names = list(model.variables.keys())
    deps = {}
    for var_name in var_names:
        if var_name in model.nexts:
            deps[var_name] = _find_next_refs(model.nexts[var_name])
        else:
            deps[var_name] = set()
    return _topological_sort(deps, var_names)


def compute_initial_states(model: SmvModel) -> list[tuple[State, StateDict]]:
    """Compute all initial states. Returns (state_key, state_dict) pairs."""
    var_names = list(model.variables.keys())

    # For each variable, compute the set of possible initial values
    init_values = {}
    for var_name in var_names:
        if var_name in model.inits:
            val = evaluate(model.inits[var_name], {}, None, model.defines)
            if isinstance(val, list):
                init_values[var_name] = val
            else:
                init_values[var_name] = [val]
        else:
            # No init: unconstrained, any value in domain
            init_values[var_name] = get_domain(model.variables[var_name])

    # Cartesian product
    results = []
    for combo in itertools.product(*(init_values[v] for v in var_names)):
        sd = dict(zip(var_names, combo))
        sk = tuple(combo)
        results.append((sk, sd))
    return results


def compute_successors(state_dict: StateDict, model: SmvModel,
                       dep_order: list[str]) -> list[StateDict]:
    """Compute all successor states from a given state."""
    # Build next-state assignments incrementally, respecting dependencies
    # Each entry in partial_assignments is a partial next-state dict
    partial_assignments = [{}]

    for var_name in dep_order:
        new_assignments = []
        for partial in partial_assignments:
            if var_name in model.nexts:
                val = evaluate(model.nexts[var_name], state_dict,
                               partial, model.defines)
                if isinstance(val, list):
                    # Non-deterministic: branch
                    for v in val:
                        new_partial = dict(partial)
                        new_partial[var_name] = v
                        new_assignments.append(new_partial)
                else:
                    partial[var_name] = val
                    new_assignments.append(partial)
            else:
                # No next assignment: non-deterministic over entire domain
                domain = get_domain(model.variables[var_name])
                for v in domain:
                    new_partial = dict(partial)
                    new_partial[var_name] = v
                    new_assignments.append(new_partial)
        partial_assignments = new_assignments

    # Validate: successor values must be in domain
    var_names = list(model.variables.keys())
    valid = []
    for pa in partial_assignments:
        sd = {v: pa[v] for v in var_names}
        in_domain = True
        for v in var_names:
            domain = get_domain(model.variables[v])
            if sd[v] not in domain:
                in_domain = False
                break
        if in_domain:
            valid.append(sd)
    return valid


def explore(model: SmvModel) -> ExplicitResult:
    """Full explicit-state exploration: compute initial states, transitions, reachable states."""
    var_names = list(model.variables.keys())
    dep_order = compute_dep_order(model)

    # Compute total state space size
    total = 1
    for v in var_names:
        total *= len(get_domain(model.variables[v]))

    # Compute initial states
    init_pairs = compute_initial_states(model)
    initial_keys = [sk for sk, _ in init_pairs]

    # BFS
    visited: set[State] = set()
    state_map: dict[State, StateDict] = {}
    transitions: list[tuple[State, State]] = []
    bfs_layers: list[set[State]] = []

    queue = deque()
    first_layer = set()
    for sk, sd in init_pairs:
        if sk not in visited:
            visited.add(sk)
            state_map[sk] = sd
            queue.append((sk, sd))
            first_layer.add(sk)
    bfs_layers.append(first_layer)

    while queue:
        current_layer_size = len(queue)
        next_layer = set()

        for _ in range(current_layer_size):
            src_key, src_dict = queue.popleft()
            succs = compute_successors(src_dict, model, dep_order)
            for dst_dict in succs:
                dst_key = tuple(dst_dict[v] for v in var_names)
                transitions.append((src_key, dst_key))
                if dst_key not in visited:
                    visited.add(dst_key)
                    state_map[dst_key] = dst_dict
                    queue.append((dst_key, dst_dict))
                    next_layer.add(dst_key)
                elif dst_key not in state_map:
                    state_map[dst_key] = dst_dict

        if next_layer:
            bfs_layers.append(next_layer)

    return ExplicitResult(
        var_names=var_names,
        total_states=total,
        initial_states=initial_keys,
        reachable_states=visited,
        transitions=transitions,
        bfs_layers=bfs_layers,
        state_to_dict=state_map,
    )
