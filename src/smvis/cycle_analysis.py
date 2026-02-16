"""Cycle analysis: Tarjan's SCC decomposition and R(n) repeatable state computation."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from smvis.explicit_engine import ExplicitResult, State


@dataclass
class CycleAnalysisResult:
    """Result of cycle analysis on a state-transition graph."""

    # SCC decomposition
    sccs: list[frozenset[State]]
    nontrivial_sccs: list[frozenset[State]]
    state_to_scc_id: dict[State, int]

    # Repeatable state layers
    r_sets: dict[int, set[State]]        # n -> states first repeatable at step n
    cumulative_r: dict[int, set[State]]  # n -> R(1) | ... | R(n)
    min_return_time: dict[State, int]    # state -> min{n : s in R(n)}
    convergence_step: int                # k where cumulative_r[k] == r_star
    r_star: set[State]                   # all repeatable states
    transient_states: set[State]         # reachable \ r_star

    # Cycle edges
    cycle_edges: set[tuple[State, State]]


def compute_sccs(
    reachable_states: set[State],
    transitions: list[tuple[State, State]],
) -> tuple[list[frozenset[State]], dict[State, int]]:
    """Iterative Tarjan's SCC decomposition.

    Returns:
        (sccs, state_to_scc_id) where sccs is a list of frozensets and
        state_to_scc_id maps each state to its SCC index.
    """
    adj: dict[State, list[State]] = defaultdict(list)
    for src, dst in transitions:
        if src in reachable_states and dst in reachable_states:
            adj[src].append(dst)

    index_counter = 0
    tarjan_stack: list[State] = []
    on_stack: set[State] = set()
    indices: dict[State, int] = {}
    lowlinks: dict[State, int] = {}
    sccs: list[frozenset[State]] = []

    for start in reachable_states:
        if start in indices:
            continue

        # Iterative DFS: stack of (node, neighbor_index)
        call_stack: list[tuple[State, int]] = [(start, 0)]
        indices[start] = lowlinks[start] = index_counter
        index_counter += 1
        tarjan_stack.append(start)
        on_stack.add(start)

        while call_stack:
            v, ni = call_stack[-1]
            neighbors = adj.get(v, [])

            if ni < len(neighbors):
                w = neighbors[ni]
                call_stack[-1] = (v, ni + 1)

                if w not in indices:
                    indices[w] = lowlinks[w] = index_counter
                    index_counter += 1
                    tarjan_stack.append(w)
                    on_stack.add(w)
                    call_stack.append((w, 0))
                elif w in on_stack:
                    lowlinks[v] = min(lowlinks[v], indices[w])
            else:
                # All neighbors processed; pop frame
                call_stack.pop()

                if lowlinks[v] == indices[v]:
                    scc_members: list[State] = []
                    while True:
                        w = tarjan_stack.pop()
                        on_stack.discard(w)
                        scc_members.append(w)
                        if w == v:
                            break
                    sccs.append(frozenset(scc_members))

                # Update parent's lowlink
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlinks[parent] = min(lowlinks[parent], lowlinks[v])

    state_to_scc_id: dict[State, int] = {}
    for i, scc in enumerate(sccs):
        for s in scc:
            state_to_scc_id[s] = i

    return sccs, state_to_scc_id


def _find_nontrivial_sccs(
    sccs: list[frozenset[State]],
    transitions: list[tuple[State, State]],
    reachable_states: set[State],
) -> list[frozenset[State]]:
    """Identify non-trivial SCCs (size > 1, or single state with self-loop)."""
    self_loops: set[State] = set()
    for src, dst in transitions:
        if src == dst and src in reachable_states:
            self_loops.add(src)

    nontrivial = []
    for scc in sccs:
        if len(scc) > 1:
            nontrivial.append(scc)
        elif len(scc) == 1:
            s = next(iter(scc))
            if s in self_loops:
                nontrivial.append(scc)
    return nontrivial


def _compute_r_sets_for_scc(
    scc: frozenset[State],
    adj: dict[State, list[State]],
    max_steps: int,
) -> tuple[dict[int, set[State]], dict[State, int]]:
    """Compute R(n) layers for a single SCC using batch forward expansion.

    Returns:
        (new_at_step, min_return_time) where new_at_step[k] = states first
        found repeatable at step k within this SCC.
    """
    # reach[s] = states reachable from s in exactly k steps (within SCC)
    reach: dict[State, set[State]] = {
        s: set(adj.get(s, [])) for s in scc
    }

    new_at_step: dict[int, set[State]] = {}
    min_return: dict[State, int] = {}
    found: set[State] = set()

    for k in range(1, max_steps + 1):
        new_k: set[State] = set()
        for s in scc:
            if s not in found and s in reach[s]:
                new_k.add(s)
                min_return[s] = k
        new_at_step[k] = new_k
        found |= new_k

        if found == scc:
            break

        # Advance: reach_{k+1}[s] = union of adj[t] for t in reach_k[s]
        new_reach: dict[State, set[State]] = {}
        for s in scc:
            if s in found:
                # Optimization: skip states already found
                new_reach[s] = set()
                continue
            next_set: set[State] = set()
            for t in reach[s]:
                for u in adj.get(t, []):
                    next_set.add(u)
            new_reach[s] = next_set
        reach = new_reach

    return new_at_step, min_return


def compute_r_sets(
    nontrivial_sccs: list[frozenset[State]],
    transitions: list[tuple[State, State]],
    reachable_states: set[State],
    max_steps: int = 100,
) -> tuple[dict[int, set[State]], dict[int, set[State]], dict[State, int], int]:
    """Compute R(n) sets across all non-trivial SCCs.

    Returns:
        (r_sets, cumulative_r, min_return_time, convergence_step)
    """
    # Build adjacency restricted to reachable states
    global_adj: dict[State, list[State]] = defaultdict(list)
    for src, dst in transitions:
        if src in reachable_states and dst in reachable_states:
            global_adj[src].append(dst)

    # Build per-SCC adjacency and compute R(n) per SCC
    all_new_at_step: dict[int, set[State]] = defaultdict(set)
    all_min_return: dict[State, int] = {}

    for scc in nontrivial_sccs:
        scc_adj: dict[State, list[State]] = {}
        for s in scc:
            scc_adj[s] = [t for t in global_adj.get(s, []) if t in scc]

        new_at_step, min_return = _compute_r_sets_for_scc(scc, scc_adj, max_steps)

        for k, states in new_at_step.items():
            all_new_at_step[k] |= states
        all_min_return.update(min_return)

    # Build cumulative sets and find convergence
    r_sets: dict[int, set[State]] = {}
    cumulative_r: dict[int, set[State]] = {}
    cumulative_so_far: set[State] = set()
    convergence_step = 0

    r_star_total: set[State] = set()
    for scc in nontrivial_sccs:
        r_star_total |= scc

    if not r_star_total:
        return {}, {}, {}, 0

    max_k = max(all_new_at_step.keys()) if all_new_at_step else 0
    for k in range(1, max_k + 1):
        r_sets[k] = all_new_at_step.get(k, set())
        cumulative_so_far = cumulative_so_far | r_sets[k]
        cumulative_r[k] = set(cumulative_so_far)
        if cumulative_so_far == r_star_total:
            convergence_step = k
            break
    else:
        convergence_step = max_k if max_k > 0 else 0

    return r_sets, cumulative_r, all_min_return, convergence_step


def analyze_cycles(
    explicit_result: ExplicitResult,
    max_steps: int = 100,
) -> CycleAnalysisResult:
    """Full cycle analysis: Tarjan's SCC -> R(n) computation -> cycle edges.

    Args:
        explicit_result: Result from explicit-state exploration.
        max_steps: Maximum R(n) step to compute.

    Returns:
        CycleAnalysisResult with SCC decomposition, R(n) layers, and cycle edges.
    """
    reachable = explicit_result.reachable_states
    transitions = explicit_result.transitions

    # Step 1: Tarjan's SCC decomposition
    sccs, state_to_scc_id = compute_sccs(reachable, transitions)

    # Step 2: Find non-trivial SCCs
    nontrivial = _find_nontrivial_sccs(sccs, transitions, reachable)

    # Step 3: Compute R(n) sets
    r_sets, cumulative_r, min_return_time, convergence_step = compute_r_sets(
        nontrivial, transitions, reachable, max_steps
    )

    # R* = union of all non-trivial SCCs
    r_star: set[State] = set()
    for scc in nontrivial:
        r_star |= scc

    # Transient = reachable \ R*
    transient = reachable - r_star

    # Step 4: Cycle edges = edges within non-trivial SCCs
    cycle_edges: set[tuple[State, State]] = set()
    for src, dst in transitions:
        if src in r_star and dst in r_star:
            if state_to_scc_id.get(src) == state_to_scc_id.get(dst):
                cycle_edges.add((src, dst))

    return CycleAnalysisResult(
        sccs=sccs,
        nontrivial_sccs=nontrivial,
        state_to_scc_id=state_to_scc_id,
        r_sets=r_sets,
        cumulative_r=cumulative_r,
        min_return_time=min_return_time,
        convergence_step=convergence_step,
        r_star=r_star,
        transient_states=transient,
        cycle_edges=cycle_edges,
    )
