# smvis

Interactive visualizer for finite-state SMV (nuXmv/NuSMV) models. Generates explicit state-transition graphs, BDD visualizations, and integrates with the nuXmv model checker for specification verification.

Built for EECS 6315.

## Features

- **SMV Editor** -- edit models directly in the browser with live parsing
- **State-Transition Graph** -- explicit enumeration of states and transitions with Cytoscape.js visualization
- **BDD Visualization** -- symbolic encoding using BDDs (via `dd` library) with reduction statistics
- **nuXmv Spec Checking** -- run CTL, LTL, and INVARSPEC checks through nuXmv and display pass/fail results
- **Counterexample Traces** -- highlight counterexample paths on the state graph (with loop-back visualization for lasso traces)
- **Interactive nuXmv Terminal** -- embedded nuXmv `-int` session with quick-command buttons
- **9 Example Models** -- counter, mutex, GCD, multiplication, bubble sort, traffic light, swap, abs_diff, fibonacci

## Setup

### 1. Install Python dependencies

Requires Python 3.10+.

```bash
cd smvis
pip install -e ".[dev]"
```

This installs `lark`, `dd`, `dash`, `dash-cytoscape`, and dev tools (`pytest`).

### 2. Set up nuXmv binary (optional, for spec checking)

The nuXmv binary is **not included** in this repository due to license restrictions. To enable specification checking and the interactive terminal:

1. Download nuXmv from https://nuxmv.fbk.eu/download.html (v2.1.0 or later)
2. Extract the archive
3. Place the binary at:
   ```
   smvis/bin/nuxmv/nuXmv.exe      (Windows)
   smvis/bin/nuxmv/nuXmv           (Linux/macOS)
   ```

The directory structure should look like:
```
smvis/
  bin/
    nuxmv/
      nuXmv.exe
  src/
  tests/
  examples/
  ...
```

Without the nuXmv binary, the visualizer still works for state graph exploration and BDD visualization. The "Check All Specs" button and interactive terminal will show an error message.

### 3. Run

```bash
python -m smvis
```

Opens at http://localhost:8050. Load any `.smv` file or use the built-in examples dropdown.

## Running tests

```bash
pytest
```

Tests that require the nuXmv binary are automatically skipped if it is not found.

## Example models

| Model | Description |
|-------|-------------|
| `counter.smv` | Simple up/down counter with mode switching |
| `mutex.smv` | Peterson's mutual exclusion with fairness |
| `gcd_01.smv` | Euclid's GCD algorithm |
| `mult.smv` | Multiplication by repeated addition |
| `bubble_sort3.smv` | 3-element bubble sort |
| `traffic_light.smv` | Traffic light controller |
| `swap.smv` | Variable swap |
| `abs_diff.smv` | Absolute difference (nuXmv range error expected) |
| `fibonacci.smv` | Fibonacci sequence (nuXmv range error expected) |
