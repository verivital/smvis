# smvis V4: Repeatable States & SCC Visualization

## Status: COMPLETE (394 tests passing)

## Context

V3 was complete with nuXmv integration, interactive terminal, and counterexample trace visualization. V4 adds the theoretical foundation for understanding liveness properties by making **repeatable states** and **strongly connected components** visible.

**Pedagogical goal**: Students need to understand which states can be visited infinitely often (repeatable/recurring states) to reason about fairness and liveness. V4 makes this visible.

---

## Summary of Changes

| Area | Scope |
|------|-------|
| **1. Cycle analysis** | New `cycle_analysis.py` — Tarjan SCC, R(n) computation, convergence |
| **2. Repeatable overlay** | New `apply_repeatable_overlay()` in graph_builder — 3 modes |
| **3. UI controls** | Checkbox, mode dropdown, R(n) slider with convergence labels |
| **4. R(n) fading** | Edges between non-active nodes fade in R(n) mode |
| **5. Tests** | New `test_cycle_analysis.py` covering Tarjan, R(n), convergence |

---

## 1. Cycle Analysis (`src/smvis/cycle_analysis.py`)

### Core Algorithm
- **Tarjan's SCC**: Find all strongly connected components in O(V+E)
- **Nontrivial SCCs**: SCCs with >1 state or a self-loop (contain real cycles)
- **R(1)**: States with self-loops (return in 1 step)
- **R(n)**: States reachable from themselves in exactly n steps
- **R***: Union of all R(n) = all repeatable/recurring states = states in nontrivial SCCs
- **Transient states**: Reachable states NOT in any nontrivial SCC
- **Convergence**: Smallest n where cumulative R(1)|...|R(n) = R*

### Data Structure
```python
@dataclass
class CycleAnalysisResult:
    nontrivial_sccs: list[frozenset[State]]
    r_star: set[State]           # all repeatable states
    transient_states: set[State] # reachable but not repeatable
    r_sets: dict[int, set[State]] # R(n) for each n
    cumulative_r: dict[int, set[State]]  # R(1)|...|R(n)
    min_return_time: dict[State, int]    # min n where state in R(n)
    convergence_step: int        # smallest n where cumR(n) = R*
    cycle_edges: set[tuple[State, State]]
    state_to_scc_id: dict[State, int]
```

---

## 2. Visualization Modes

### R* Mode
- Repeatable states: purple
- Transient states: gray/faded
- Cycle edges (within SCCs): highlighted purple

### R(n) Layers Mode
- Slider from 1 to convergence step
- Cumulative view: shows R(1)|...|R(k) at slider position k
- Heat-map coloring by min return time (red=1, orange=2, ..., purple=high)
- Non-active nodes (not yet in cumulative set): faded
- Non-active edges: faded to very light gray

### SCC Coloring Mode
- Each nontrivial SCC gets a distinct color (8 rotating colors)
- Transient states: gray

---

## 3. UI Controls

- **Checkbox**: "Repeatable states" in graph options
- **Mode dropdown**: R* / R(n) layers / SCC coloring
- **Step slider**: 1 to convergence, with marks
- **Step label**: Shows "R(k) / R(convergence)"
- Controls hidden when checkbox unchecked

---

## 4. Model Statistics

| Model | Reachable | R* | Transient | SCCs | Convergence |
|-------|-----------|-----|-----------|------|-------------|
| counter.smv | 24 | 16 | 8 | 1 | 6 |
| mutex.smv | 16 | 10 | 6 | 1 | 5 |
| gcd_01.smv | 352 | 121 | 231 | 121 | 1 |
| traffic_light.smv | 6 | 6 | 0 | 1 | 3 |

---

## Tests: 394 total (all passing)

- `test_cycle_analysis.py`: Tarjan SCC, R(n) computation, convergence detection, edge cases
- All V1-V3 tests continue to pass unchanged
