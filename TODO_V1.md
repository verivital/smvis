# smvis V1 Robustness & Enhancement Plan

## Context

smvis is a working prototype for EECS 6315 that visualizes nuXmv models as transition systems with explicit-state and BDD-based reachability. The core functionality works (parser, engines, graph rendering), but it has GUI bugs, no tests, no packaging, limited examples, and the BDD visualization needs significant improvement. This plan addresses all of those to make it shareable and robust.

## Summary of Changes

| Area | Scope |
|------|-------|
| **1. Packaging & Install** | `pyproject.toml`, `requirements.txt`, cross-platform `setup_env.py`, README |
| **2. Bug Fixes** | 6 known bugs in app.py, graph_builder.py, __main__.py |
| **3. Tests** | pytest suite: parser, engines, expression eval, graph builder, BDD, integration |
| **4. BDD Visualization** | ROBDD hierarchical layout, truth tables, multiple BDD views, construction stages |
| **5. New Examples** | 5 new algorithm models with Python source annotations |
| **6. Image Export** | PNG download for state graph and BDD visualizations |

---

## 1. Packaging & Installation

### Files to Create

**`smvis/pyproject.toml`**
```toml
[project]
name = "smvis"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["lark", "dd", "dash", "dash-cytoscape"]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[project.scripts]
smvis = "smvis.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
smvis = ["smv_grammar.lark", "assets/*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**`smvis/requirements.txt`** -- pinned versions for reproducibility
**`smvis/requirements-dev.txt`** -- adds pytest, pytest-cov

**`smvis/setup_env.py`** -- cross-platform bootstrap script that:
1. Creates a venv (`.venv/`)
2. Installs the package in editable mode with dev deps
3. Verifies `dd.autoref` imports correctly
4. Runs a smoke test (parse counter.smv)
5. Prints success message with launch instructions

Works on Windows (PowerShell/cmd) and Unix (bash) by detecting platform with `sys.platform`.

**`smvis/README.md`** -- Quick start, manual setup, usage, testing instructions

---

## 2. Bug Fixes

### Bug 2.1: Predecessor Bug (`graph_builder.py:159`)
**Line 159**: `preds.add(dst)` → `preds.add(src)`

When `dst == sk`, the predecessor is `src`, not `dst`. One-character fix.

### Bug 2.2: Redundant Recomputation (`app.py`)
Three callbacks (`update_graph`, `update_bdd_view`, `show_state_detail`) re-parse and re-explore from scratch on every UI interaction.

**Fix**: Add module-level cache in `app.py`:
```python
_compute_cache: dict[str, tuple[ExplicitResult, BddResult]] = {}
```
Keyed by SHA-256 of SMV text. `compute_all` populates the cache; other callbacks read from it. Fallback to recompute on cache miss. Limit cache to 10 entries.

### Bug 2.3: Silent Exception Handling (`app.py`)
Callbacks at lines 381, 408, 439 catch `Exception` and silently return `no_update`.

**Fix**: Add `import logging; log = logging.getLogger("smvis")` and replace each bare except with:
- `log.exception("Error in <callback_name>")`
- Return visible error feedback (error message in `bdd-info`, `state-detail`, or empty graph elements)

### Bug 2.4: Concentric Layout Broken (`app.py:379`)
Passing a JavaScript function string `"function(n) {...}"` doesn't work -- dash-cytoscape serializes it as a JSON string.

**Fix**: Use **preset layout** with manually computed positions. Add `compute_concentric_positions(elements)` to `graph_builder.py`:
- Group nodes by `depth` data attribute
- Place each depth level in a ring with radius proportional to depth
- Set `node["position"] = {"x": ..., "y": ...}` for each node
- Return `layout_dict = {"name": "preset", "animate": False}`

### Bug 2.5: Hover Tooltip Fixed Position (`app.py:442-503`)
Tooltip always at `top:200px; right:40px` regardless of cursor. Also stays visible when mouse leaves graph area.

**Fix**: Convert to **clientside callback** (runs in browser JS):
1. Create `assets/tooltip.js` with `mousemove` listener storing `window._lastMouseX/Y`
2. Add `mouseleave` listener on graph container to hide tooltip
3. Replace Python callback with `app.clientside_callback(...)` that positions tooltip at cursor + 15px offset
4. Remove duplicate style definitions (keep only in clientside JS)

### Bug 2.6: Stale "nuXmv" References
Replace in 4 files:
- `__init__.py:1` → `"""smvis - Interactive visualization of finite-state SMV models."""`
- `__main__.py:16` → `"Starting smvis at http://localhost:{port}"`
- `app.py:50` → `html.H2("smvis", ...)`
- `assets/style.css:1` → `/* smvis - SMV Model Visualizer Styles */`

---

## 3. Tests

### Directory Structure
```
smvis/
  tests/
    __init__.py
    conftest.py              # Shared fixtures (parsed models, explicit results, BDD results)
    test_parser.py           # Parser correctness for all models
    test_expression_eval.py  # Unit tests for evaluate() function
    test_explicit_engine.py  # State counts, transitions, BFS for all models
    test_bdd_engine.py       # BDD reachable counts match explicit, encoding correctness
    test_edge_cases.py       # Unconstrained init/next, next() refs, topological sort
    test_graph_builder.py    # Element generation, predecessor fix regression
    test_bdd_visualizer.py   # Summary formatting
    test_app.py              # App creation, model file discovery
    test_integration.py      # Full pipeline: parse → explore → BDD → graph → visualize
```

### `conftest.py` Fixtures (session-scoped for speed)
```python
@pytest.fixture(scope="session")
def counter_model():
    return parse_smv_file("examples/counter.smv")

@pytest.fixture(scope="session")
def counter_explicit(counter_model):
    return explore(counter_model)

@pytest.fixture(scope="session")
def counter_bdd(counter_model, counter_explicit):
    return build_from_explicit(counter_model, ...)
```
Same pattern for gcd, mult, mutex.

### Key Test Cases

**`test_parser.py`** -- For each model verify:
- Variable count and names
- Variable types (BoolType, EnumType, RangeType with correct bounds)
- Init/next assignment counts
- Spec counts by kind (INVARSPEC, CTLSPEC, LTLSPEC, SPEC)
- DEFINE count and values
- FAIRNESS count
- mutex.smv: NextRef nodes exist in flag1/flag2 next expressions

**`test_expression_eval.py`** -- Unit tests for `evaluate()`:
- Literals (int, bool), VarRef, DEFINE expansion
- Arithmetic: +, -, *, /, mod, division by zero → 0
- Comparison: =, !=, >, <, >=, <=
- Boolean: &, |, !, ->
- CaseExpr: first match, fallthrough, no match → ValueError
- SetExpr: returns list of values
- NextRef: reads next_state, missing → ValueError
- UnaryMinus: fold IntLit, wrap UnaryOp

**`test_explicit_engine.py`** -- For each model verify:
- `total_states` matches arithmetic (e.g., counter: 2×2×26=104)
- `len(initial_states)` matches expected
- `len(reachable_states)` matches expected
- `len(set(transitions))` matches expected (deduplicated)
- All initial states are reachable
- All transitions connect reachable states (for reachable-only BFS)
- `state_to_dict` has entries for all reachable states

**`test_bdd_engine.py`**:
- Reachable count matches explicit engine for all 4 models
- Encoding bit counts are correct (e.g., counter: mode=1bit, press=1bit, x=5bits)
- Fixpoint converges (last iteration has 0 new states)
- Domain constraint excludes invalid bit patterns

**`test_edge_cases.py`**:
- gcd/mult: unconstrained init produces full domain Cartesian product
- counter: `press` without `next()` is non-deterministic over full domain
- mutex: topological sort puts process1/process2 before flag1/flag2

**`test_graph_builder.py`**:
- `build_elements` produces correct node/edge counts
- Initial nodes have "initial" class
- `max_nodes` limit works
- **Regression**: predecessor bug fix (after Bug 2.1 fix, verify `preds.add(src)`)

**`test_integration.py`**:
- Parametrized over all model files: parse → explore → BDD → graph build → summary -- all without error
- BDD structure extraction works for init/reached/trans selectors

**`test_app.py`**:
- `create_app()` succeeds without error
- `_find_model_files()` returns expected file count
- Layout is not None

### Expected Values (from MEMORY.md, verified)
| Model | Total | Reachable | Init | Transitions |
|-------|-------|-----------|------|-------------|
| counter.smv | 104 | 24 | 2 | 48 |
| gcd_01.smv | 605 | 352 | 121 | 352 |
| mult.smv | 48884* | 242 | 121 | 242 |
| mutex.smv | 72 | 16 | 1 | 30 |

*Note: mult.smv total = 11×11×101×4 = 48,884. MEMORY.md says 12,342 which may reflect a different prod range. Will verify on first test run and update.

---

## 4. BDD Visualization Improvements

### 4.1 ROBDD Hierarchical Layout

**Problem**: Current breadthfirst layout doesn't show the variable ordering levels that are characteristic of ROBDDs.

**Solution**: Use **preset layout** with computed coordinates in `bdd_engine.py:get_bdd_structure()`:

1. Extract the BDD variable ordering from `dd.autoref` (the variable at each internal node via `node.var`)
2. Assign each variable a **level** (y-coordinate) based on its position in the ordering
3. Within each level, space nodes horizontally
4. Terminal nodes (0, 1) go at the bottom level
5. Add **level labels** as invisible "label" nodes on the left margin showing the variable name at each level

Modify `get_bdd_structure()` to return position data:
```python
elements.append({
    "data": {"id": str(node_id), "label": var},
    "position": {"x": x, "y": level * 80},
    "classes": "internal",
})
```

Add variable ordering labels as a separate list for the info panel.

### 4.2 Truth Table for Small BDDs

Add `build_truth_table(bdd_node, bdd, encoding, var_names)` to `bdd_engine.py`:
- For BDDs with ≤ 6 relevant variables (≤ 64 rows): generate full truth table
- For larger BDDs: show only satisfying assignments (or first N rows with "...")
- Return as list of dicts for rendering as an HTML table in the info panel

Display in `app.py` BDD info panel: a scrollable table showing all variable assignments and the boolean function value.

### 4.3 Multiple BDD Views

Expand the BDD selector dropdown to include:
- **Initial States** (existing)
- **Reachable States** (existing)
- **Transition Relation** (existing)
- **Domain Constraint** (new -- shows what bit patterns are valid)
- **Iteration N** (new -- show the reached BDD at each fixpoint iteration)

For iteration views, store per-iteration BDD nodes in `BddResult`:
```python
@dataclass
class BddResult:
    ...
    iteration_bdds: list[object]  # reached BDD after each iteration
```

### 4.4 BDD Construction Stages (Educational)

Since `dd.autoref` only provides the final reduced BDD, simulate the reduction process:

1. **Show reduction rules as text** in the info panel:
   - "Redundant test removal: node where high=low child is eliminated"
   - "Isomorphic subgraph merging: identical sub-BDDs share one node"

2. **Count reductions**: Compare `2^n_vars` (max possible nodes in decision tree) vs actual BDD node count. Display: "Full tree: 127 nodes → Reduced BDD: 12 nodes (90.6% reduction)"

3. **Optional**: For very small BDDs (≤4 variables), generate the unreduced decision tree manually and display it alongside the reduced BDD for comparison.

### 4.5 Files to Modify
- `bdd_engine.py`: Add `build_truth_table()`, store `iteration_bdds`, add position computation to `get_bdd_structure()`
- `bdd_visualizer.py`: Add reduction stats formatting
- `graph_builder.py`: Add `BDD_STYLESHEET` updates for level labels
- `app.py`: Expand BDD selector dropdown, add truth table rendering, add iteration stepper

---

## 5. New Example Models

Each model includes Python algorithm as SMV comments and clear spec annotations.

### 5.1 `fibonacci.smv` -- Fibonacci Sequence
```
Python: a,b = 0,1; for i in range(n): a,b = b,a+b
Vars: a:0..15, b:0..15, i:0..6, pc:{l0,l1,l2}
DEFINE n := 5
Init: a=0, b=1, i=0, pc=l0
Est. reachable: ~12 states (tiny, clean loop)
```
Key insight: SMV simultaneous next-state semantics means `next(a):=b; next(b):=a+b` works without a temp variable.

### 5.2 `bubble_sort3.smv` -- Bubble Sort on 3 Elements
```
Python: Two passes of compare-and-swap on a[0],a[1],a[2]
Vars: a0:0..4, a1:0..4, a2:0..4, pc:{l0..l6}
Init: pc=l0, a0/a1/a2 unconstrained (all permutations)
Est. reachable: ~300-400 states (branching paths)
```
Specs: `INVARSPEC pc=l6 -> a0<=a1 & a1<=a2` (sorted at end), `CTLSPEC AF(pc=l6)` (terminates)

### 5.3 `traffic_light.smv` -- Traffic Light Controller
```
Python: Main/side road lights with timer-based cycling
Vars: main_light:{green,yellow,red}, side_light:{green,yellow,red}, timer:0..5
Init: main=green, side=red, timer=0
Est. reachable: ~10-12 states (cyclic, no termination)
```
Specs: `INVARSPEC !(main=green & side=green)` (safety), `CTLSPEC AG AF(main=green)` (liveness)

### 5.4 `abs_diff.smv` -- Absolute Difference
```
Python: if a>=b: result=a-b; else: result=b-a
Vars: a:0..7, b:0..7, result:0..7, pc:{l0,l1,l2,l3}
Init: pc=l0, result=0 (a,b unconstrained)
Est. reachable: ~192 states (branching, all-input)
```

### 5.5 `swap.smv` -- Swap Two Variables with Temp
```
Python: temp=a; a=b; b=temp
Vars: a:0..4, b:0..4, temp:0..4, pc:{l0,l1,l2,l3}
Init: pc=l0, temp=0 (a,b unconstrained)
Est. reachable: ~100 states
```
Specs: `LTLSPEC (a=3 & b=1) -> F(a=1 & b=3)` (swap works)

### Implementation Notes
- Each `.smv` file has comment header showing the Python algorithm
- All models verified against existing Lark grammar (no new syntax needed)
- Range bounds chosen to keep total state space under 10K for visualization

---

## 6. Image Export

### State Graph PNG Export
- Add "Export PNG" button next to layout selector
- Use dash-cytoscape's built-in `generateImage` property:
  ```python
  Output("state-graph", "generateImage")
  # Returns {"type": "png", "action": "download"}
  ```

### BDD PNG Export
- Same approach for BDD graph component

### GIF Generation (stretch goal)
- Server-side: render each BFS layer as a frame, combine with Pillow
- Or: render each fixpoint iteration BDD as a frame
- Save as downloadable GIF

---

## Implementation Order

| Phase | Tasks | Dependencies |
|-------|-------|-------------|
| **Phase 1** | Bug fixes 2.1, 2.3, 2.6 (quick fixes) | None |
| **Phase 2** | Packaging: pyproject.toml, requirements, setup_env.py, README | None |
| **Phase 3** | Bug fix 2.2 (caching), 2.4 (concentric layout) | Phase 1 |
| **Phase 4** | Bug fix 2.5 (tooltip clientside callback) | Phase 1 |
| **Phase 5** | Tests: conftest + all test files | Phase 1-2 |
| **Phase 6** | New example models (5 files) | Phase 1 |
| **Phase 7** | BDD visualization improvements | Phase 1, 3 |
| **Phase 8** | Image export | Phase 3 |
| **Phase 9** | Run all tests, fix any failures, final verification | All |

## Verification

After implementation:
1. `python setup_env.py` succeeds on clean checkout
2. `pytest` passes all tests (100% of backend, GUI tests pass where browser available)
3. `python -m smvis` launches, all 9 models load and compute without errors
4. BDD visualization shows proper ROBDD hierarchy for counter.smv
5. Hover tooltip follows cursor, concentric layout works
6. PNG export produces valid images
7. No callback errors in browser console or server logs
