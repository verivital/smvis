# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**smvis** is an interactive educational visualizer for finite-state NuSMV/nuXmv models, built for EECS 6315 (Model Checking). It combines three model checking paradigms — explicit-state exploration, symbolic BDD computation, and automata-theoretic LTL checking — with web-based visualization via Dash and Cytoscape.js.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run the web app (opens http://localhost:8050)
python -m smvis

# Run all tests (554 tests, ~33s)
pytest

# Run a single test file
pytest tests/test_ltl_buchi.py

# Run a single test by name
pytest tests/test_ltl_buchi.py::TestPatternF::test_f_pattern_2_states -v

# Run tests matching a keyword
pytest -k "buchi" -v

# Quick smoke test (no browser)
python -c "from smvis.app import create_app; create_app(); print('OK')"
```

## Architecture

The system is a pipeline where each stage's output feeds the next, all converging in the Dash web UI (`app.py`):

```
.smv text
  → smv_parser.py (Lark LALR)  → SmvModel (IR)
  → explicit_engine.py          → ExplicitResult (states, transitions)
  → bdd_engine.py               → BddResult (symbolic reachability)
  → cycle_analysis.py           → CycleAnalysisResult (SCCs, R(n) sets)
  → graph_builder.py            → Cytoscape elements (nodes/edges/CSS)
  → app.py                      → Dash web UI at :8050
```

For LTL model checking, a parallel pipeline composes the system with a Buchi automaton:

```
LTLSPEC φ
  → ltl_buchi.py: negate → pattern match → BuchiAutomaton
  → product_model.py: compose(SmvModel, Buchi) → product SmvModel
  → explicit_engine.py: explore(product) → product ExplicitResult  [reused unchanged]
  → cycle_analysis.py: analyze(product) → product CycleAnalysisResult  [reused unchanged]
  → accepting_cycles.py: filter SCCs by acceptance + fairness → lasso counterexample
  → graph_builder.py: overlay lasso on original graph
```

The key architectural insight is that the **product is just another SmvModel** — `product_model.py` adds a `_buchi_q` enum variable and feeds it through the existing pipeline with zero changes to `explicit_engine.py` or `cycle_analysis.py`.

## Key Module Responsibilities

| Module | Role |
|--------|------|
| `smv_model.py` | IR dataclasses: `SmvModel`, expression AST nodes, `SpecDecl` |
| `smv_parser.py` | Lark LALR parser + `SmvTransformer`. Entry: `parse_smv(text)` |
| `smv_grammar.lark` | LALR grammar for the NuSMV subset (119 lines) |
| `explicit_engine.py` | BFS state exploration. Entry: `explore(model) → ExplicitResult` |
| `bdd_engine.py` | Binary encoding + dd.autoref BDDs. Entry: `build_from_explicit(...)` |
| `cycle_analysis.py` | Tarjan SCC, R(n) repeatable states. Entry: `analyze_cycles(result)` |
| `ltl_buchi.py` | LTL negation, 9 pattern templates, HOA parser, Spot WSL fallback |
| `product_model.py` | Synchronous product M × A with `_dead` sink state |
| `accepting_cycles.py` | Fairness-aware SCC filter, lasso extraction, trace projection |
| `graph_builder.py` | Cytoscape element builders + all CSS stylesheets |
| `app.py` | Dash layout, all callbacks, result caching (SHA-256 keyed, LRU 10) |
| `nuxmv_runner.py` | nuXmv subprocess: batch checking, interactive terminal, trace parsing |

## Critical Constraints

**Parser must use LALR, not Earley.** Earley's lexer doesn't prioritize string terminals over regex, so `"FALSE"` tokenizes as temporal op `"F"` + IDENT `"ALSE"`. LALR gets keyword priority right.

**NuSMV semantics for unconstrained variables:**
- No `init()` → any domain value is valid (not just first)
- No `next()` → fully non-deterministic over domain (NOT identity)
- `next(var)` can appear in case conditions of OTHER variables' `next()` assignments (see mutex.smv) — requires dependency-aware evaluation

**Buchi dead state (`_dead`):** When a Buchi state has only conditional outgoing transitions (no `true` self-loop), the product needs a `_dead` non-accepting absorbing sink. Without it, runs where no guard fires incorrectly continue in the current Buchi state, causing false counterexamples.

**GF/FG parsing:** The parser treats `GF(...)` as a single IDENT. Use `G F(...)` with a space in .smv files.

## Test Fixtures

`tests/conftest.py` provides session-scoped fixtures for the 4 core example models (`counter_model`, `mutex_model`, `gcd_model`, `mult_model`) and their pre-computed explicit/BDD results. Tests for nuXmv features are auto-skipped when `bin/nuxmv/nuXmv.exe` is absent.

## Dependencies

- `lark` — LALR parser generator (grammar in `smv_grammar.lark`)
- `dd` — BDD library (`dd.autoref` for Python-level BDD operations)
- `dash` + `dash-cytoscape` — Web UI framework + graph visualization
- nuXmv binary (optional, at `bin/nuxmv/nuXmv.exe`) — for spec checking and interactive terminal
