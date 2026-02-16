"""Accepting cycle detection for LTL model checking via Buchi product.

After composing the system M with the Buchi automaton A_¬φ, an LTL property
φ is violated iff the product M × A has an accepting cycle — a cycle that
visits a Buchi accepting state infinitely often AND satisfies all fairness
constraints.

This module:
1. Identifies accepting product states (where _buchi_q is an accepting state)
2. Filters nontrivial SCCs to find accepting SCCs (containing ≥1 accepting state)
3. Further filters by fairness (each SCC must contain a state satisfying each
   FAIRNESS expression)
4. Extracts lasso counterexamples (prefix + cycle)
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from smvis.explicit_engine import ExplicitResult, State, StateDict, evaluate
from smvis.cycle_analysis import CycleAnalysisResult
from smvis.product_model import ProductInfo, BUCHI_VAR


@dataclass
class AcceptingCycleResult:
    """Result of accepting cycle analysis on the product graph."""
    has_accepting_cycle: bool
    accepting_sccs: list[frozenset[State]]      # SCCs with accepting + fairness
    lasso: tuple[list[State], list[State]] | None  # (prefix, cycle) or None
    accepting_product_states: set[State]         # states where Buchi is accepting
    property_holds: bool                         # True if LTL property holds


def find_accepting_cycles(
    explicit_result: ExplicitResult,
    cycle_result: CycleAnalysisResult,
    product_info: ProductInfo,
) -> AcceptingCycleResult:
    """Find accepting cycles in the product graph.

    An accepting SCC must:
    1. Be nontrivial (contain a real cycle)
    2. Contain at least one state where _buchi_q is an accepting state
    3. Contain states satisfying every FAIRNESS constraint in the product model
    """
    state_to_dict = explicit_result.state_to_dict
    buchi_var = product_info.buchi_var_name
    accepting_names = set(product_info.accepting_states)

    # Step 1: Find all accepting product states
    accepting_product_states: set[State] = set()
    for s in explicit_result.reachable_states:
        sd = state_to_dict[s]
        if sd.get(buchi_var) in accepting_names:
            accepting_product_states.add(s)

    # Step 2: Filter nontrivial SCCs to those with accepting states
    accepting_sccs: list[frozenset[State]] = []
    for scc in cycle_result.nontrivial_sccs:
        if scc & accepting_product_states:
            accepting_sccs.append(scc)

    # Step 3: Filter by fairness constraints
    fairness = product_info.product_model.fairness
    defines = product_info.product_model.defines
    if fairness:
        fair_sccs: list[frozenset[State]] = []
        for scc in accepting_sccs:
            if _scc_satisfies_fairness(scc, state_to_dict, fairness, defines):
                fair_sccs.append(scc)
        accepting_sccs = fair_sccs

    has_cycle = len(accepting_sccs) > 0

    # Step 4: Extract lasso if property is violated
    lasso = None
    if has_cycle:
        lasso = extract_lasso(
            explicit_result, accepting_sccs[0],
            accepting_product_states, product_info,
        )

    return AcceptingCycleResult(
        has_accepting_cycle=has_cycle,
        accepting_sccs=accepting_sccs,
        lasso=lasso,
        accepting_product_states=accepting_product_states,
        property_holds=not has_cycle,
    )


def _scc_satisfies_fairness(
    scc: frozenset[State],
    state_to_dict: dict[State, StateDict],
    fairness: list,
    defines: dict,
) -> bool:
    """Check if an SCC satisfies all fairness constraints.

    For each FAIRNESS expression, there must be at least one state in the SCC
    where the expression evaluates to True.
    """
    for f_expr in fairness:
        satisfied = False
        for s in scc:
            sd = state_to_dict[s]
            try:
                val = evaluate(f_expr, sd, None, defines)
                if val:
                    satisfied = True
                    break
            except (ValueError, KeyError):
                continue
        if not satisfied:
            return False
    return True


def extract_lasso(
    explicit_result: ExplicitResult,
    accepting_scc: frozenset[State],
    accepting_states: set[State],
    product_info: ProductInfo,
) -> tuple[list[State], list[State]]:
    """Extract a lasso counterexample: (prefix, cycle).

    prefix: path from an initial state to the accepting SCC
    cycle: path within the SCC through an accepting state and back
    """
    # Build adjacency
    adj: dict[State, list[State]] = defaultdict(list)
    for src, dst in explicit_result.transitions:
        adj[src].append(dst)

    # --- Prefix: BFS from initial states to SCC ---
    prefix = _bfs_to_scc(
        explicit_result.initial_states, accepting_scc, adj,
        explicit_result.reachable_states,
    )

    # --- Cycle: find path within SCC through an accepting state ---
    entry = prefix[-1] if prefix else next(iter(accepting_scc))
    cycle = _find_cycle_through_accepting(
        entry, accepting_scc, accepting_states, adj,
    )

    return (prefix, cycle)


def _bfs_to_scc(
    initial_states: list[State],
    target_scc: frozenset[State],
    adj: dict[State, list[State]],
    reachable: set[State],
) -> list[State]:
    """BFS from initial states to first state in target SCC. Returns path."""
    # If any initial state is in the SCC, return just that state
    for s in initial_states:
        if s in target_scc:
            return [s]

    visited: set[State] = set()
    parent: dict[State, State | None] = {}
    queue: deque[State] = deque()

    for s in initial_states:
        if s not in visited:
            visited.add(s)
            parent[s] = None
            queue.append(s)

    target = None
    while queue:
        current = queue.popleft()
        for succ in adj.get(current, []):
            if succ not in visited and succ in reachable:
                visited.add(succ)
                parent[succ] = current
                if succ in target_scc:
                    target = succ
                    break
                queue.append(succ)
        if target is not None:
            break

    if target is None:
        # Should not happen if SCC is reachable
        return [next(iter(target_scc))]

    # Reconstruct path
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path


def _find_cycle_through_accepting(
    entry: State,
    scc: frozenset[State],
    accepting_states: set[State],
    adj: dict[State, list[State]],
) -> list[State]:
    """Find a cycle within the SCC that passes through an accepting state.

    Returns a list of states forming the cycle (first = last for a true cycle).
    """
    acc_in_scc = scc & accepting_states
    if not acc_in_scc:
        return [entry]  # shouldn't happen

    # Pick an accepting state in the SCC
    acc_target = next(iter(acc_in_scc))

    # BFS within SCC from entry to acc_target
    path_to_acc = _bfs_within_scc(entry, acc_target, scc, adj)

    # BFS within SCC from acc_target back to entry
    path_back = _bfs_within_scc(acc_target, entry, scc, adj)

    if path_to_acc and path_back:
        # Combine: entry -> ... -> acc_target -> ... -> entry
        cycle = path_to_acc + path_back[1:]  # avoid duplicating acc_target
        return cycle

    # Fallback: just return entry
    return [entry]


def _bfs_within_scc(
    start: State,
    target: State,
    scc: frozenset[State],
    adj: dict[State, list[State]],
) -> list[State]:
    """BFS from start to target within an SCC. Returns path including both endpoints."""
    if start == target:
        # Find a cycle back to start within SCC
        visited: set[State] = set()
        parent: dict[State, State | None] = {}
        queue: deque[State] = deque()

        for succ in adj.get(start, []):
            if succ in scc and succ not in visited:
                visited.add(succ)
                parent[succ] = start
                if succ == start:
                    return [start, start]
                queue.append(succ)

        while queue:
            current = queue.popleft()
            for succ in adj.get(current, []):
                if succ in scc and succ not in visited:
                    visited.add(succ)
                    parent[succ] = current
                    if succ == start:
                        # Found path back to start
                        path = [start]
                        cur = current
                        sub = []
                        while cur != start:
                            sub.append(cur)
                            cur = parent[cur]
                        sub.reverse()
                        path.extend(sub)
                        path.append(start)
                        return path
                    queue.append(succ)

        return [start]

    # Standard BFS
    visited: set[State] = {start}
    parent: dict[State, State | None] = {start: None}
    queue: deque[State] = deque([start])

    while queue:
        current = queue.popleft()
        for succ in adj.get(current, []):
            if succ in scc and succ not in visited:
                visited.add(succ)
                parent[succ] = current
                if succ == target:
                    path = []
                    cur: State | None = target
                    while cur is not None:
                        path.append(cur)
                        cur = parent.get(cur)
                    path.reverse()
                    return path
                queue.append(succ)

    return []  # unreachable within SCC (shouldn't happen)


def project_trace(
    lasso: tuple[list[State], list[State]],
    product_info: ProductInfo,
) -> tuple[list[dict], list[dict]]:
    """Project product states to original model variables only.

    Returns (prefix_dicts, cycle_dicts) where each dict has only the
    original model variable values (no _buchi_q).
    """
    original_vars = product_info.original_var_names
    var_names = list(product_info.product_model.variables.keys())

    def project_state(state: State) -> dict:
        sd = dict(zip(var_names, state))
        return {k: sd[k] for k in original_vars}

    prefix, cycle = lasso
    return (
        [project_state(s) for s in prefix],
        [project_state(s) for s in cycle],
    )
