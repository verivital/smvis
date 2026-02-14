# smvis V2: Bug Fixes, GUI Regression Tests & Layout Improvements

## Context

V1 is committed at `54a82ec` with 127 passing tests and 9 example models. Manual testing of the BDD visualization dropdowns revealed two runtime bugs (broken BDD edge references, invalid Cytoscape property) and layout deficiencies (fixed-width panels, undersized BDD graph). This plan fixes the bugs, adds comprehensive regression tests that exercise every GUI callback permutation across all 9 models, adds server-side GUI error logging, and improves the panel layout.

---

## Summary of Changes

| Area | Scope |
|------|-------|
| **1. Bug: BDD dangling edges** | Fix `get_bdd_structure()` to never emit edges to truncated nodes |
| **2. Bug: mouseoutNodeData** | Replace invalid property with `clearOnUnhover=True` pattern |
| **3. GUI error logging** | Add server-side callback logging + `/debug` error log panel |
| **4. Regression tests** | New `test_gui_regression.py` covering all models x all BDD selectors x all graph options |
| **5. Resizable panels** | CSS splitter for left/right panels with drag handle |
| **6. Full-width BDD** | Make BDD graph 100% panel width, info panel below instead of side-by-side |

---

## 1. Bug: BDD Dangling Edges (`bdd_engine.py:303-402`)

### Root Cause

In `get_bdd_structure()`, `_traverse()` checks `node_count[0] > max_nodes` at line 319 and returns early without adding the node. But the **parent** already called `_traverse(high)` / `_traverse(low)` and unconditionally appends edges at lines 347-348 and 354-355 pointing to the child's ID -- even though the child node was never added to `elements`. This creates Cytoscape errors like:

> "Can not create edge with nonexistant target `7364`"

### Fix

After traversal, filter out edges whose `target` or `source` points to a node ID not in the `visited` set. Add a post-processing step before the positioning code:

```python
# File: src/smvis/bdd_engine.py, after _traverse(bdd_node) try block (~line 363)
# Remove edges that reference nodes not in 'visited' (truncated by max_nodes)
visited_ids = {str(nid) for nid in visited}
elements = [e for e in elements
            if "source" not in e.get("data", {})  # keep all nodes
            or (e["data"]["source"] in visited_ids and e["data"]["target"] in visited_ids)]
```

This is a 3-line fix. An alternative (checking before appending each edge) would require restructuring the recursion; the post-filter is simpler and equally correct.

---

## 2. Bug: `mouseoutNodeData` Invalid Property (`app.py:594`)

### Root Cause

`mouseoutNodeData` is not a supported property of `dash_cytoscape.Cytoscape`. The valid hover property is `mouseoverNodeData`, and the correct "mouseout" mechanism is:
- Set `clearOnUnhover=True` on the Cytoscape component
- When the mouse leaves a node, `mouseoverNodeData` is automatically cleared to `None`

### Fix (3 changes)

**A. Add `clearOnUnhover=True`** to the state-graph Cytoscape component (`app.py:194`):
```python
cyto.Cytoscape(
    id="state-graph",
    clearOnUnhover=True,  # ADD THIS
    ...
)
```

**B. Remove `mouseoutNodeData` from the clientside callback** (`app.py:593-596`). Change inputs from:
```python
Input("state-graph", "mouseoverNodeData"),
Input("state-graph", "mouseoutNodeData"),   # REMOVE
State("parsed-model-store", "data"),
```
to:
```python
Input("state-graph", "mouseoverNodeData"),
State("parsed-model-store", "data"),
```

**C. Update the JS function** (`app.py:521-596`) to handle `None` hover data as the mouseout signal:
```javascript
function(hoverData, modelData) {
    if (!hoverData) {
        return ['', {display: 'none'}];
    }
    // ... existing tooltip building code, minus the triggerId check ...
}
```

The `tooltip.js` `mouseleave` listener in `assets/tooltip.js` still serves as a safety fallback -- keep it.

---

## 3. GUI Error Logging

### 3.1 Server-side Callback Logging

Add a small log buffer that records every callback invocation and any exceptions. This helps debug issues that only appear in the browser.

**File: `src/smvis/app.py`** -- add at module level:
```python
_callback_log: list[dict] = []
_CALLBACK_LOG_MAX = 200

def _log_callback(name: str, inputs: dict, error: str | None = None):
    import time
    entry = {"time": time.time(), "callback": name, "inputs": inputs}
    if error:
        entry["error"] = error
    _callback_log.append(entry)
    if len(_callback_log) > _CALLBACK_LOG_MAX:
        _callback_log.pop(0)
```

Call `_log_callback(...)` at the start (and in `except` blocks) of each callback: `compute_all`, `update_graph`, `update_bdd_view`, `show_state_detail`.

### 3.2 Debug Log Panel

Add a hidden debug panel toggled by a button in the header:

```python
html.Button("Debug Log", id="btn-debug", n_clicks=0, style={...}),
html.Div(id="debug-panel", style={"display": "none", ...}),
```

Callback: on click, toggle visibility and render `_callback_log` as a scrollable pre-formatted list showing timestamp, callback name, inputs, and any errors. This lets the user inspect what happened server-side without browser dev tools.

### 3.3 Callback Error Surfacing

Ensure all callbacks show errors visually rather than swallowing them. The V1 code already has `log.exception(...)` in `except` blocks -- verify and fill any gaps.

---

## 4. Regression Tests: `tests/test_gui_regression.py`

### Goal

Programmatically invoke every callback combination across all 9 models to verify no exceptions. This catches issues like the BDD dangling-edge bug without needing a browser.

### Test Structure

```python
"""GUI callback regression tests.

Exercises every callback x every model x every option permutation
to ensure no exceptions are raised.
"""
import os, pytest
from smvis.smv_parser import parse_smv_file
from smvis.explicit_engine import explore
from smvis.bdd_engine import build_from_explicit, get_bdd_structure, build_truth_table
from smvis.graph_builder import build_elements, compute_concentric_positions, get_state_detail
from smvis.bdd_visualizer import get_bdd_summary, get_reduction_stats

EXAMPLES_DIR = ...
MODEL_FILES = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".smv")]

@pytest.fixture(scope="module", params=MODEL_FILES)
def model_pipeline(request):
    """Parse -> explore -> BDD for each model."""
    path = os.path.join(EXAMPLES_DIR, request.param)
    model = parse_smv_file(path)
    explicit = explore(model)
    bdd = build_from_explicit(model, explicit.initial_states,
                               explicit.transitions, explicit.var_names,
                               explicit.state_to_dict)
    return request.param, model, explicit, bdd
```

### Test Cases

**4.1 Graph builder permutations** (parametrized):
- `reachable_only` in [True, False]
- `max_nodes` in [10, 100, 500]
- Validates Cytoscape element integrity for each combination

**4.2 Layout permutations**:
- All 5 layouts: cose, breadthfirst, grid, circle, concentric
- Concentric: verify `compute_concentric_positions()` adds position dicts

**4.3 BDD selector permutations**:
- Static selectors: init, reached, trans, domain
- Dynamic selectors: iter_0, iter_1, ... iter_N (per model's iteration count)
- For each: call `get_bdd_structure()` + `_build_bdd_view` equivalent + validate elements

**4.4 BDD structure edge integrity** (regression for Bug 1):
```python
def test_bdd_structure_no_dangling_edges(model_pipeline):
    """Every edge target and source must reference an existing node."""
    _, _, _, bdd = model_pipeline
    for bdd_node in [bdd.init_bdd, bdd.reached_bdd, bdd.trans_bdd,
                     bdd.domain_constraint] + bdd.iteration_bdds:
        elems = get_bdd_structure(bdd_node, bdd.bdd)
        node_ids = {e["data"]["id"] for e in elems if "source" not in e["data"]}
        for e in elems:
            if "source" in e["data"]:
                assert e["data"]["source"] in node_ids, f"dangling source: {e}"
                assert e["data"]["target"] in node_ids, f"dangling target: {e}"
```

**4.5 Truth table permutations**:
- For init, reached, domain BDD nodes across all models
- Verify returns None (too many vars) or valid list of dicts

**4.6 State detail for all reachable states**:
- Build elements, then call `get_state_detail()` for every node
- Verify non-None result for each

### Helper: `_validate_cytoscape_elements()`
```python
def _validate_cytoscape_elements(elements):
    """Validate that elements form a valid Cytoscape graph."""
    node_ids = set()
    for e in elements:
        d = e["data"]
        if "source" not in d:
            assert "id" in d
            node_ids.add(d["id"])
    for e in elements:
        d = e["data"]
        if "source" in d:
            assert d["source"] in node_ids, f"edge source {d['source']} not in nodes"
            assert d["target"] in node_ids, f"edge target {d['target']} not in nodes"
```

### Expected Test Count
9 models x (6 graph combos + 5 layouts + ~5-10 BDD selectors + truth table + state detail + edge integrity) ~ 200-300 additional test cases.

---

## 5. Resizable Panels

### Approach: CSS + JS Drag Handle

Use a thin vertical drag handle between the left (editor) and right (graphs) panels. No external library needed.

**File: `src/smvis/assets/style.css`** -- add:
```css
.main-container {
    display: flex;
    height: calc(100vh - 60px);
}
.left-panel {
    min-width: 200px;
    max-width: 60%;
    overflow-y: auto;
    padding: 12px;
}
.resize-handle {
    width: 6px;
    cursor: col-resize;
    background: #ddd;
    transition: background 0.2s;
}
.resize-handle:hover, .resize-handle.active {
    background: #3498db;
}
.right-panel {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
}
```

**File: `src/smvis/assets/resize.js`** (new):
```javascript
document.addEventListener('DOMContentLoaded', function() {
    var observer = new MutationObserver(function() {
        var handle = document.querySelector('.resize-handle');
        if (!handle || handle._resizeInit) return;
        handle._resizeInit = true;
        var leftPanel = handle.previousElementSibling;
        var dragging = false;
        handle.addEventListener('mousedown', function(e) {
            dragging = true;
            handle.classList.add('active');
            e.preventDefault();
        });
        document.addEventListener('mousemove', function(e) {
            if (!dragging) return;
            var containerLeft = leftPanel.parentElement.getBoundingClientRect().left;
            var newWidth = e.clientX - containerLeft;
            newWidth = Math.max(200, Math.min(newWidth, window.innerWidth * 0.6));
            leftPanel.style.width = newWidth + 'px';
            leftPanel.style.flex = 'none';
        });
        document.addEventListener('mouseup', function() {
            dragging = false;
            handle.classList.remove('active');
        });
    });
    observer.observe(document.body, {childList: true, subtree: true});
});
```

**File: `src/smvis/app.py`** -- update the main layout div structure:
```python
html.Div([
    html.Div([...], className="left-panel", style={"width": "30%"}),
    html.Div(className="resize-handle"),  # NEW drag handle
    html.Div([...], className="right-panel"),
], className="main-container"),
```

Remove the inline `style={"width": "30%"}` and `style={"width": "70%"}` from the left/right columns; use CSS class-based sizing instead.

---

## 6. Full-Width BDD Visualization

### Problem

BDD graph is `"width": "50%"` with info panel side-by-side at `"width": "48%"`. This wastes space -- the BDD graph should be as wide as the transition system graph above it.

### Fix (`app.py:252-267`)

Change BDD layout from side-by-side to stacked (graph on top, info below):

```python
# BDD Section -- BEFORE (side-by-side)
html.Div([
    cyto.Cytoscape(id="bdd-graph", style={"width": "50%", ...}),
    html.Div(id="bdd-info", style={"width": "48%", ...}),
]),

# BDD Section -- AFTER (stacked, full width)
cyto.Cytoscape(
    id="bdd-graph",
    style={"width": "100%", "height": "350px", "border": "1px solid #ddd"},
    ...
),
html.Div(id="bdd-info", style={
    "fontSize": "12px", "marginTop": "8px",
    "maxHeight": "250px", "overflowY": "auto",
}),
```

Also increase the BDD graph height from 250px to 350px to match the state graph height (380px).

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/smvis/bdd_engine.py` | Filter dangling edges after traversal (Bug 1) |
| `src/smvis/app.py` | Fix mouseoutNodeData (Bug 2), add clearOnUnhover, add callback logging, add debug panel, BDD layout stacked full-width, add resize-handle div, use CSS classes |
| `src/smvis/assets/style.css` | Add .main-container, .left-panel, .resize-handle, .right-panel |
| `src/smvis/assets/resize.js` | **New**: drag-to-resize JS |
| `src/smvis/assets/tooltip.js` | No change (keep as fallback) |
| `tests/test_gui_regression.py` | **New**: comprehensive callback regression tests |

---

## Implementation Order

| Phase | Tasks |
|-------|-------|
| **1** | Fix Bug 1 (BDD dangling edges) + Bug 2 (mouseoutNodeData) |
| **2** | Add GUI error logging + debug panel |
| **3** | Resizable panels (CSS + JS) |
| **4** | Full-width BDD layout |
| **5** | Regression tests (`test_gui_regression.py`) |
| **6** | Run full test suite, start server, verify all 9 models x all BDD selectors |

---

## Verification

1. `pytest` passes all existing 127 tests + new regression tests (~300 new)
2. `python -m smvis` starts without errors
3. For each of the 9 models: click "Compute All", then cycle through all BDD dropdown options -- no console errors
4. Hover tooltip appears on node hover, disappears on mouseout -- no `mouseoutNodeData` warning
5. Left/right panels are resizable by dragging the handle
6. BDD graph fills the full right-panel width
7. Debug Log button shows callback history with no errors
8. No regressions in state graph layouts (cose, breadthfirst, grid, circle, concentric)
