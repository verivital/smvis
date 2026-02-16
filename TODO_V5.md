# smvis V5: LTL Model Checking with Buchi Automata

## Status: COMPLETE (554 tests passing)

## Context

V4 is complete with 394 passing tests, adding repeatable states (R(n)), SCC visualization, and cycle analysis. V5 makes the **automata-theoretic approach to LTL model checking** concrete and visual — the hardest concept in EECS 6315 to understand from textbooks alone.

**The pedagogical problem**: Students learn that LTL model checking works by (1) negating the property, (2) converting to a Buchi automaton, (3) composing with the system, (4) searching for accepting cycles. But they never *see* these steps happen. V5 makes each step visible and interactive.

**The approach**: Rather than calling nuXmv as a black box, we perform the automata-theoretic construction explicitly:
1. Negate the LTL formula (show the negation)
2. Build a Buchi automaton for ¬φ (show the small automaton graph)
3. Compose the Buchi with the transition system as a new NuSMV model (model transformation)
4. Explore the composed product with our existing engine (reuse V1-V4 infrastructure)
5. Find accepting cycles = counterexamples (reuse V4 cycle analysis + fairness filter)
6. Extract and visualize lasso counterexamples (prefix + loop)

---

## Key Concepts

| Term | Definition |
|------|-----------|
| **Buchi automaton** | Finite automaton over infinite words; accepts if an accepting state is visited infinitely often |
| **¬φ Buchi** | Buchi automaton that accepts exactly the traces violating property φ |
| **Synchronous product** | M × A: product states (s,q), transitions when both M and A can move |
| **Accepting cycle** | A cycle in the product visiting at least one accepting (Buchi) state |
| **Lasso** | Counterexample shape: finite prefix + repeating cycle (u · v^ω) |
| **Fairness** | NuSMV `FAIRNESS expr` = Buchi acceptance: expr must hold infinitely often |
| **Dead state** | `_dead`: non-accepting absorbing sink for when no Buchi guard fires |

**Algorithm**: M |= φ iff L(M) ∩ L(A_¬φ) = ∅ iff the product M × A_¬φ has no reachable accepting cycle.

---

## Architecture: Reuse Everything

The key insight: **the product is just another SmvModel**. We programmatically construct a new `SmvModel` with the Buchi state as an additional variable, then feed it through the existing pipeline:

```
Original SmvModel + BuchiAutomaton
        ↓ compose()
Product SmvModel (original vars + _buchi_q)
        ↓ explore()           ← existing V1 engine, unchanged
Product ExplicitResult
        ↓ analyze_cycles()    ← existing V4 engine, unchanged
Product CycleAnalysisResult
        ↓ find_accepting_cycles()  ← NEW: filter SCCs by Buchi acceptance + fairness
AcceptingCycleResult
        ↓ extract_lasso()     ← NEW: BFS prefix + BFS cycle
Lasso Trace
        ↓ project_trace()     ← NEW: remove _buchi_q, keep original vars
        ↓ apply_lasso_overlay() ← NEW: prefix (orange) + cycle (red) on graph
Visualization
```

Zero modifications to `explicit_engine.py`, `cycle_analysis.py`, or `bdd_engine.py`.

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/smvis/ltl_buchi.py` | **NEW** (~750 lines) | LTL negation, 9 pattern templates, BuchiAutomaton, HOA parser, Spot fallback |
| `src/smvis/product_model.py` | **NEW** (~210 lines) | Compose SmvModel + BuchiAutomaton → product SmvModel with _dead state |
| `src/smvis/accepting_cycles.py` | **NEW** (~230 lines) | Fairness-aware SCC filter, lasso extraction, trace projection |
| `src/smvis/smv_grammar.lark` | Modified | Added `U` (Until) binary temporal operator |
| `src/smvis/smv_parser.py` | Modified | Added `until_op` transformer method |
| `src/smvis/graph_builder.py` | Modified | `build_buchi_elements()`, `apply_lasso_overlay()`, BUCHI_STYLESHEET |
| `src/smvis/app.py` | Modified | LTL Buchi Analysis UI, compose callback, Buchi graph panel |
| `tests/test_ltl_buchi.py` | **NEW** (70 tests) | Negation, simplification, all 9 patterns, HOA parsing, real models |
| `tests/test_product_model.py` | **NEW** (44 tests) | Composition, exploration, accepting cycles, lasso, projection |
| `examples/request_grant.smv` | **NEW** | Minimal 2-state arbiter (4 LTLSPEC, all VIOLATED) |
| `examples/two_bit_counter.smv` | **NEW** | 4-state deterministic counter (3 LTLSPEC) |
| `examples/traffic_light.smv` | Modified | Added 3 LTLSPEC declarations |

---

## Supported Buchi Patterns (9 templates)

| # | Negated formula | States | Example original φ |
|---|----------------|--------|-------------------|
| 1 | `G(p)` | 1: q0(acc)→q0[p] | `F !p` |
| 2 | `F(p)` | 2+dead: q0→q0[t], q0→q1[p], q1(acc)→q1[t] | `G p` |
| 3 | `FG(p)` | 2+dead: q0→q0[t], q0→q1[p], q1(acc)→q1[p] | `GF !p` |
| 4 | `GF(p)` | 2: q0→q0[t], q0→q1[p], q1(acc)→q0[t] | `FG !p` |
| 5 | `F(p & G(q))` | 2+dead: q0→q0[t], q0→q1[p&q], q1(acc)→q1[q] | `G(p → F !q)` |
| 6 | `F(p & F(q))` | 3+dead: q0→q0[t], q0→q1[p], q1→q1[t], q1→q2[q], q2(acc)→q2[t] | `G(done → G done)` |
| 7 | `p & G(q)` | 2+dead: q0→q1[p&q], q1(acc)→q1[q] | `(cond) → F(result)` |
| 8 | `G(p → G(q))` | 2: q0(acc)→q0[!p\|q], q0→q1[p&!q], q1→q1[t] | — |
| 9 | `FG(p) \| FG(q)` | 3+dead: q0→q0[t], q0→q1[p], q0→q2[q], q1(acc)→q1[p], q2(acc)→q2[q] | `GF(a) & GF(b)` |

These cover every LTLSPEC in all example models. For unsupported patterns, falls back to Spot via WSL (`ltl2tgba -B -S -d`) with automatic HOA parsing.

---

## Key Implementation Details

### Dead State (_dead)
When a Buchi state has only conditional outgoing transitions (no `true` self-loop), we add a `_dead` non-accepting absorbing sink state. This is critical: if no guard fires, the product run must die (not continue in the current Buchi state).

### Non-deterministic Buchi Transitions
Handled via power-set enumeration of guard combinations. For n conditional guards from same source, generates 2^n case branches ordered most-specific-first. In practice n is small (0-2).

### Fairness + Buchi Acceptance
Product fairness = original model fairness + one constraint per accepting Buchi state: `FAIRNESS _buchi_q = q_acc`. This ensures the Buchi accepting state must be visited infinitely often.

### Lasso Projection
Counterexample traces from the product are projected back to original model variables (removing `_buchi_q`) and overlaid on the original transition system graph.

---

## Verified Results

### counter.smv (24 reachable, FAIRNESS mode!=off)

| LTLSPEC | Result | Product Reachable |
|---------|--------|-------------------|
| `G(x <= count_max)` | HOLDS | 24 |
| `G(x = 0)` | VIOLATED | 48 |
| `G F(mode=off & x=0)` | HOLDS | 70 |
| `GF(mode=off) & GF(x=0)` | HOLDS | 90 |
| `GF(mode=on) & GF(x=count_max)` | HOLDS | 76 |
| `F G(mode=on)` | VIOLATED | 28 |

### two_bit_counter.smv (4 reachable, deterministic cycle)

| LTLSPEC | Result | Product Reachable |
|---------|--------|-------------------|
| `G F(b0 & b1)` | HOLDS | 11 |
| `F G(b0 & b1)` | VIOLATED | 7 |
| `G((b0 & b1) → F(!(b0) & !(b1)))` | HOLDS | 9 |

### request_grant.smv (4 reachable, no fairness)

| LTLSPEC | Result | Product Reachable |
|---------|--------|-------------------|
| `G(request → F grant)` | VIOLATED | 9 |

---

## Tests: 554 total (160 new, all passing)

- `test_ltl_buchi.py` (70 tests): negation, simplification, helpers, all 9 patterns, HOA parsing, real model specs, Buchi invariants
- `test_product_model.py` (44 tests): composition structure, product exploration, accepting cycles, lasso extraction, projection, full model battery
- All 394 existing V1-V4 tests pass unchanged
