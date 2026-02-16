# smvis V3: nuXmv Integration, Interactive Terminal & Counterexample Visualization

## Context

V2 is committed at `0b6e80d` with 317 passing tests, 9 example models, resizable panels, and BDD visualization. The example models already contain CTL, LTL, and INVAR specifications (parsed into AST but not yet verified). The nuXmv binary is available at `bin/nuxmv/nuXmv.exe` (v2.1.0). This plan adds:

1. **nuXmv model checking** — run specs through nuXmv and display pass/fail results
2. **Interactive terminal** — embed a nuXmv `-int` session in a panel below the editor
3. **Counterexample trace visualization** — highlight trace paths on the transition system graph
4. **Foundation for future Python-native model checking** (CTL/LTL/SCC analysis)

---

## Summary of Changes

| Area | Scope |
|------|-------|
| **1. nuXmv runner backend** | New `nuxmv_runner.py` — subprocess management, trace parsing |
| **2. Spec results panel** | Show pass/fail for each spec with expandable counterexamples |
| **3. Interactive terminal** | nuXmv `-int` session in a terminal panel below the editor |
| **4. Counterexample trace overlay** | Highlight trace states/edges on the transition system graph |
| **5. Tests** | New `test_nuxmv_runner.py` covering trace parsing + subprocess |

---

## 1. nuXmv Runner Backend (`src/smvis/nuxmv_runner.py`)

### 1.1 Architecture

New module that wraps nuXmv subprocess calls. Two modes:

**Batch mode** (for "Check All Specs" button):
```python
@dataclass
class SpecResult:
    spec_text: str           # "AG (x <= count_max)"
    spec_kind: str           # "CTLSPEC", "LTLSPEC", "INVARSPEC"
    passed: bool
    trace: list[dict] | None # counterexample trace if failed

@dataclass
class NuxmvResult:
    specs: list[SpecResult]
    raw_output: str          # full nuXmv stdout for debug panel
    error: str | None

def run_batch_check(smv_text: str, nuxmv_path: str) -> NuxmvResult:
    """Write SMV text to temp file, run nuXmv, parse results."""
```

**Interactive mode** (for terminal panel):
```python
class NuxmvSession:
    """Manages a long-lived nuXmv -int subprocess."""
    def __init__(self, nuxmv_path: str):
        self.process: subprocess.Popen | None = None
        self._output_buffer: list[str] = []
        self._lock = threading.Lock()

    def start(self, model_path: str | None = None) -> str: ...
    def send_command(self, cmd: str) -> str: ...
    def get_new_output(self) -> str: ...
    def stop(self): ...
```

### 1.2 Batch Check Implementation

```python
def run_batch_check(smv_text: str, nuxmv_path: str) -> NuxmvResult:
    # 1. Write smv_text to temp file
    # 2. Build command script:
    #      go
    #      check_ctlspec
    #      check_ltlspec
    #      check_invar
    #      show_traces -p 4    (XML format for reliable parsing)
    #      quit
    # 3. Run: subprocess.run([nuxmv_path, "-int", temp_model],
    #         input=script, capture_output=True, text=True, timeout=30)
    # 4. Parse text output for "-- specification ... is true/false"
    # 5. Parse XML traces from show_traces output
    # 6. Match traces to failed specs
```

### 1.3 Output Parsing

**Spec results** — regex on text output:
```
-- specification (.+?) is (true|false)
-- invariant (.+?) is (true|false)
```

**Counterexample traces** — parse XML blocks from `show_traces -p 4`:
```xml
<counter-example type="0" id="1" desc="CTL Counterexample">
  <node>
    <state id="1">
      <value variable="mode">off</value>
      <value variable="x">0</value>
    </state>
  </node>
  ...
  <loops> 1 </loops>   <!-- loop-back to state 1 for lasso traces -->
</counter-example>
```

Parse into:
```python
@dataclass
class Trace:
    states: list[dict[str, str]]  # [{var: value, ...}, ...]
    loop_start: int | None        # index of loop-back state (for LTL lasso)
    description: str              # "CTL Counterexample" / "LTL Counterexample"
```

### 1.4 Interactive Session

Use `subprocess.Popen` with pipes (works on Windows, no pty needed since nuXmv `-int` works fine via stdin/stdout pipes — verified above):

```python
class NuxmvSession:
    def start(self, model_path=None):
        args = [self.nuxmv_path, "-int"]
        if model_path:
            args.append(model_path)
        self.process = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        # Start background reader thread
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        """Background thread reading stdout line-by-line."""
        for line in self.process.stdout:
            with self._lock:
                self._output_buffer.append(line)

    def send_command(self, cmd: str) -> None:
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def get_new_output(self) -> str:
        with self._lock:
            lines = list(self._output_buffer)
            self._output_buffer.clear()
        return "".join(lines)
```

### 1.5 nuXmv Binary Path Resolution

```python
_NUXMV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "bin", "nuxmv", "nuXmv.exe"
)
```

Verify at import time; raise clear error if not found.

---

## 2. Spec Results Panel (GUI)

### 2.1 Layout Changes (`app.py`)

Add a "Verification" section below the Statistics panel in the left column:

```
Left Panel:
  ├── SMV Editor + buttons
  ├── Statistics
  └── Verification Results   ← NEW
       ├── [Check All Specs] button
       ├── Spec results table (pass/fail with icons)
       └── Expandable counterexample detail per failed spec
```

Components:
```python
html.Div([
    html.H4("Verification"),
    html.Button("Check All Specs", id="btn-check-specs", ...),
    html.Div(id="spec-results", ...),    # results table
    dcc.Store(id="traces-store", data=None),  # parsed traces for graph overlay
], style={"marginTop": "12px"}),
```

### 2.2 Results Display

Each spec result as a row:
```
✓ CTLSPEC  AG (process1 = waiting -> AF process1 = critical)     [true]
✗ CTLSPEC  AG process1 = critical                                 [false] [Show Trace]
✓ LTLSPEC  G (process1 = waiting -> F process1 = critical)       [true]
✗ LTLSPEC  F process1 = critical                                  [false] [Show Trace]
```

- Green check / red X icons
- "Show Trace" button on failed specs → stores selected trace in `traces-store` → triggers graph overlay
- Raw nuXmv output available via Debug Log panel

### 2.3 Callback: `check_all_specs`

```python
@app.callback(
    Output("spec-results", "children"),
    Output("traces-store", "data"),
    Input("btn-check-specs", "n_clicks"),
    State("smv-editor", "value"),
    prevent_initial_call=True,
)
def check_all_specs(n, text):
    result = run_batch_check(text, _NUXMV_PATH)
    # Build results table + serialize traces for store
    ...
```

---

## 3. Interactive Terminal Panel

### 3.1 Approach: Textarea-based Terminal (No WebSockets)

Rather than xterm.js + WebSockets (heavy dependency), use a simple but effective approach:
- A `<pre>` output area showing nuXmv output (monospace, dark theme, scrollable)
- A text `<input>` for typing commands
- A `dcc.Interval` component polling for new output every 500ms
- Server-side `NuxmvSession` instance managing the subprocess

This works because nuXmv responds to commands quickly and doesn't require real-time character-by-character streaming.

### 3.2 Layout

Add a collapsible terminal panel below the left-panel editor (or below the main-container as a bottom panel):

```
┌─────────────────────────────────────────────────────────────┐
│ Header                                                       │
├────────────┬────┬───────────────────────────────────────────┤
│ Left Panel │ ↔  │ Right Panel (graphs)                       │
├────────────┴────┴───────────────────────────────────────────┤
│ ▼ nuXmv Terminal                          [Start] [Stop]     │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ nuXmv > go                                               │ │
│ │ nuXmv > check_ctlspec                                    │ │
│ │ -- specification AG x <= count_max is true                │ │
│ │ ...                                                       │ │
│ └─────────────────────────────────────────────────────────┘ │
│ nuXmv > [___________________________________] [Send]         │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 CSS

```css
.terminal-panel {
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: Consolas, monospace;
    font-size: 12px;
    border-top: 2px solid #3498db;
}
.terminal-output {
    height: 200px;
    overflow-y: auto;
    padding: 8px;
    white-space: pre-wrap;
}
.terminal-input {
    display: flex;
    padding: 4px 8px;
    border-top: 1px solid #333;
}
```

### 3.4 Components

```python
# Terminal section (below main-container)
html.Div([
    html.Div([
        html.H4("nuXmv Terminal", style={"margin": "0", "flex": "1", "color": "#ecf0f1"}),
        html.Button("Start", id="btn-terminal-start", ...),
        html.Button("Stop", id="btn-terminal-stop", ...),
        html.Button("▼", id="btn-terminal-toggle", ...),  # collapse
    ], style={"display": "flex", ...}),
    html.Pre(id="terminal-output", className="terminal-output"),
    html.Div([
        html.Span("nuXmv > ", style={"color": "#3498db"}),
        dcc.Input(id="terminal-input", type="text", placeholder="Type command...",
                  style={"flex": "1", "background": "#2d2d2d", "color": "#d4d4d4", ...}),
        html.Button("Send", id="btn-terminal-send", ...),
    ], className="terminal-input"),
    dcc.Interval(id="terminal-poll", interval=500, disabled=True),  # polling
    dcc.Store(id="terminal-session-store", data=None),  # session state
], className="terminal-panel", id="terminal-panel"),
```

### 3.5 Callbacks

**Start session:**
```python
# Module-level session (one per server instance)
_nuxmv_session: NuxmvSession | None = None

@app.callback(
    Output("terminal-output", "children"),
    Output("terminal-poll", "disabled"),
    Input("btn-terminal-start", "n_clicks"),
    State("smv-editor", "value"),
    prevent_initial_call=True,
)
def start_terminal(n, text):
    global _nuxmv_session
    # Write text to temp file, start session with it
    _nuxmv_session = NuxmvSession(_NUXMV_PATH)
    _nuxmv_session.start(temp_path)
    return "Session started.\n", False  # enable polling
```

**Send command:**
```python
@app.callback(
    Output("terminal-input", "value"),
    Input("btn-terminal-send", "n_clicks"),
    Input("terminal-input", "n_submit"),  # Enter key
    State("terminal-input", "value"),
    prevent_initial_call=True,
)
def send_command(n_click, n_submit, cmd):
    if _nuxmv_session and cmd:
        _nuxmv_session.send_command(cmd.strip())
    return ""  # clear input
```

**Poll for output:**
```python
@app.callback(
    Output("terminal-output", "children", allow_duplicate=True),
    Input("terminal-poll", "n_intervals"),
    State("terminal-output", "children"),
    prevent_initial_call=True,
)
def poll_terminal(n, current_output):
    if not _nuxmv_session:
        return no_update
    new = _nuxmv_session.get_new_output()
    if not new:
        return no_update
    return (current_output or "") + new
```

### 3.6 Quick-Command Buttons

Add convenience buttons above the terminal for common operations:
```python
html.Div([
    html.Button("go", id="btn-cmd-go", ...),
    html.Button("check_ctlspec", id="btn-cmd-ctl", ...),
    html.Button("check_ltlspec", id="btn-cmd-ltl", ...),
    html.Button("check_invar", id="btn-cmd-invar", ...),
    html.Button("show_traces", id="btn-cmd-traces", ...),
], style={"display": "flex", "gap": "4px", ...}),
```

Each triggers `send_command` with the corresponding command string.

---

## 4. Counterexample Trace Visualization

### 4.1 Trace-to-Graph Mapping

When user clicks "Show Trace" on a failed spec:

1. Parse the trace's state dicts from `traces-store`
2. Map each trace state to a node ID in the transition system graph by matching variable assignments
3. Build a list of (source_id, target_id) edges for consecutive trace states
4. Store as `active-trace` data

### 4.2 Graph Overlay

Update `build_elements()` or add a post-processing step to apply trace highlighting:

```python
def apply_trace_overlay(elements: list[dict], trace_states: list[dict],
                        state_to_dict: dict, loop_start: int | None) -> list[dict]:
    """Add 'trace' CSS class to nodes/edges in the trace path."""
    # Map trace states to node IDs
    trace_node_ids = []
    for ts in trace_states:
        for state, sd in state_to_dict.items():
            if all(str(sd.get(k)) == str(v) for k, v in ts.items()):
                trace_node_ids.append(state_to_id(state))
                break

    # Mark trace nodes
    trace_set = set(trace_node_ids)
    for elem in elements:
        d = elem["data"]
        if "source" not in d and d["id"] in trace_set:
            elem["classes"] = elem.get("classes", "") + " trace-node"
        if "source" in d:
            # Check if this edge is in the trace path
            for i in range(len(trace_node_ids) - 1):
                if d["source"] == trace_node_ids[i] and d["target"] == trace_node_ids[i+1]:
                    elem["classes"] = elem.get("classes", "") + " trace-edge"
            # Loop-back edge
            if loop_start is not None:
                if (d["source"] == trace_node_ids[-1] and
                    d["target"] == trace_node_ids[loop_start]):
                    elem["classes"] = elem.get("classes", "") + " trace-loop"

    return elements
```

### 4.3 Cytoscape Stylesheet Additions (`graph_builder.py`)

```python
# Trace overlay styles
{"selector": ".trace-node", "style": {
    "border-color": "#e67e22", "border-width": 4,
    "background-color": "#fdebd0",
    "z-index": 999,
}},
{"selector": ".trace-edge", "style": {
    "line-color": "#e67e22", "target-arrow-color": "#e67e22",
    "width": 3, "z-index": 998,
}},
{"selector": ".trace-loop", "style": {
    "line-color": "#e74c3c", "line-style": "dashed",
    "target-arrow-color": "#e74c3c", "width": 3,
}},
# Numbered labels for trace sequence
{"selector": ".trace-node", "style": {
    "label": "data(trace_label)",  # "1: mode=off, x=0" or just step number
}},
```

### 4.4 Trace Step Numbering

Add `trace_label` and `trace_step` to node data for trace nodes:
```python
# In apply_trace_overlay:
for i, node_id in enumerate(trace_node_ids):
    for elem in elements:
        if elem["data"].get("id") == node_id:
            elem["data"]["trace_step"] = i
            elem["data"]["trace_label"] = f"[{i}]"
```

### 4.5 Callback: Show Trace

```python
@app.callback(
    Output("state-graph", "elements", allow_duplicate=True),
    Input("traces-store", "data"),
    State("smv-editor", "value"),
    State("graph-options", "value"),
    State("max-nodes", "value"),
    prevent_initial_call=True,
)
def show_trace_on_graph(trace_data, text, graph_opts, max_nodes):
    if not trace_data:
        return no_update
    # Rebuild elements with trace overlay
    ...
```

---

## 5. Tests (`tests/test_nuxmv_runner.py`)

### 5.1 Test Categories

**Trace parsing tests** (no subprocess needed):
```python
def test_parse_spec_results_text():
    """Parse '-- specification ... is true/false' lines."""

def test_parse_xml_trace_simple():
    """Parse single-path counterexample XML."""

def test_parse_xml_trace_with_loop():
    """Parse lasso trace with <loops> element."""

def test_parse_xml_multiple_traces():
    """Parse output with multiple counterexamples."""
```

**Integration tests** (require nuXmv binary):
```python
@pytest.fixture
def nuxmv_path():
    path = os.path.join(..., "bin", "nuxmv", "nuXmv.exe")
    if not os.path.exists(path):
        pytest.skip("nuXmv binary not found")
    return path

def test_batch_check_counter(nuxmv_path):
    """Run full batch check on counter.smv, verify spec results."""

def test_batch_check_mutex(nuxmv_path):
    """Mutual exclusion specs should all pass."""

def test_trace_maps_to_states(nuxmv_path):
    """Counterexample trace states match explicit engine states."""

def test_interactive_session_basic(nuxmv_path):
    """Start session, send 'go', verify output contains model info."""

def test_interactive_session_check_spec(nuxmv_path):
    """Send check commands, verify results appear in output."""
```

### 5.2 Trace Mapping Validation

Cross-validate: for each trace state from nuXmv, verify it exists in the explicit engine's `state_to_dict`. This catches encoding mismatches between our parser and nuXmv.

---

## 6. Future Foundation: Python-Native Model Checking

This V3 plan **does not implement** Python-native CTL/LTL checking, but sets up the architecture:

### 6.1 What V3 Provides for Future Work

- Parsed `SpecDecl` AST with all temporal operators (already in `smv_model.py`)
- Explicit state graph with transitions (already in `explicit_engine.py`)
- BDD encoding of states and transitions (already in `bdd_engine.py`)
- Trace visualization infrastructure (new in V3 — reusable for Python-native traces)
- Cross-validation framework (nuXmv results vs Python results)

### 6.2 Future V4 Targets (Not in This Plan)

- **CTL model checking**: Implement `check_ctl(model, spec)` using BDD fixpoint operations (EX, EU, EG operators via BDD image computation — the BDD infrastructure is already there)
- **LTL model checking**: Tableau/automata-based approach — construct Buchi automaton from negated LTL formula, compose with system, check emptiness
- **SCC decomposition**: Tarjan's algorithm on the explicit transition graph to find strongly connected components; visualize each SCC as a colored cluster in Cytoscape
- **Repeatability analysis**: For LTL fair model checking, identify SCCs that contain fair states; visualize which state subsets are "repeatable" (can be visited infinitely often)
- **Visual state subset highlighting**: Color-code states by membership in different sets (reachable, satisfying a sub-formula, in an SCC, on a counterexample path) — overlay on the same transition system graph

---

## Files to Create/Modify

| File | Action | Changes |
|------|--------|---------|
| `src/smvis/nuxmv_runner.py` | **New** | Batch check, trace parsing, interactive session |
| `src/smvis/app.py` | Modify | Add spec results panel, terminal panel, trace overlay callback, quick-command buttons |
| `src/smvis/graph_builder.py` | Modify | Add `apply_trace_overlay()`, trace CSS styles to `CYTO_STYLESHEET` |
| `src/smvis/assets/style.css` | Modify | Add `.terminal-panel`, `.terminal-output`, `.terminal-input` styles |
| `tests/test_nuxmv_runner.py` | **New** | Trace parsing + integration tests |

---

## Implementation Order

| Phase | Tasks |
|-------|-------|
| **1** | `nuxmv_runner.py`: batch check + text/XML trace parsing |
| **2** | Tests for trace parsing (unit tests, no subprocess) |
| **3** | Integration: batch check callback + spec results panel in GUI |
| **4** | Counterexample trace visualization (graph overlay + styles) |
| **5** | Interactive terminal: session management + terminal panel UI |
| **6** | Terminal polling + quick-command buttons |
| **7** | Integration tests (with nuXmv binary) |
| **8** | Full test suite + manual verification across all 9 models |

---

## Verification

1. `pytest` passes all existing 317 tests + new tests
2. `python -m smvis` starts without errors
3. For counter.smv: "Check All Specs" shows mix of pass/fail results
4. Failed spec "Show Trace" highlights trace path on transition system graph
5. LTL lasso traces show loop-back edge in red dashed style
6. Terminal: Start → type `go` → type `check_ctlspec` → see results → Stop
7. Terminal quick-command buttons work
8. mutex.smv: mutual exclusion specs pass, `AG process1 = critical` fails with 1-state trace
9. No regressions in V2 functionality (resizable panels, BDD visualization, etc.)
