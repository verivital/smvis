"""Flatten multi-module NuSMV models with process instantiation.

Converts a multi-module model (MODULE main with process instances + sub-modules)
into a single flat SmvModel with interleaving semantics:
  - Each process instance's local variables are prefixed (e.g., proc1.state)
  - A _pid scheduler variable selects which process runs each step
  - Transition assignments are guarded by _pid; non-running processes stutter
  - FAIRNESS running → FAIRNESS (_pid = idx)
"""
from __future__ import annotations
import copy
from smvis.smv_model import (
    SmvModel, VarDecl, BoolType, EnumType, RangeType,
    IntLit, BoolLit, VarRef, NextRef, BinOp, UnaryOp,
    CaseExpr, SetExpr, TemporalUnary, TemporalBinary, SpecDecl,
    Expr,
)


def flatten_modules(modules: list[dict]) -> SmvModel:
    """Flatten a multi-module model into a single SmvModel.

    Parameters
    ----------
    modules : list of dicts
        Each dict has keys: name, params, variables, defines, inits, nexts,
        fairness, specs, process_instances.  Produced by SmvTransformer.module().
    """
    # Separate main module from sub-modules
    main_mod = None
    sub_modules: dict[str, dict] = {}
    for mod in modules:
        if mod["name"] == "main":
            main_mod = mod
        else:
            sub_modules[mod["name"]] = mod

    if main_mod is None:
        raise ValueError("No MODULE main found")

    model = SmvModel()

    # Copy main module's own declarations
    model.variables = dict(main_mod["variables"])
    model.defines = dict(main_mod["defines"])
    model.inits = dict(main_mod["inits"])
    model.nexts = dict(main_mod["nexts"])
    model.fairness = list(main_mod["fairness"])
    model.specs = list(main_mod["specs"])

    instances = main_mod["process_instances"]
    if not instances:
        return model

    # Add scheduler variable: _pid ∈ {0 .. n-1}
    n_procs = len(instances)
    model.variables["_pid"] = VarDecl("_pid", RangeType(0, n_procs - 1))
    # _pid has no init() and no next() → fully non-deterministic (interleaving)

    # Collect guarded next() branches for shared (parameter) variables
    # key = target var name, value = list of (pid_guard, case_branches) pairs
    shared_nexts: dict[str, list[tuple[Expr, list]]] = {}

    for idx, proc_tuple in enumerate(instances):
        # proc_tuple: (_PROCESS_MARKER, inst_name, mod_name, args)
        _, inst_name, mod_name, actual_args = proc_tuple

        if mod_name not in sub_modules:
            raise ValueError(
                f"Process '{inst_name}' references undefined module '{mod_name}'"
            )
        sub_mod = sub_modules[mod_name]
        formal_params = sub_mod["params"]

        if len(formal_params) != len(actual_args):
            raise ValueError(
                f"Module '{mod_name}' expects {len(formal_params)} parameters "
                f"but '{inst_name}' provides {len(actual_args)}"
            )

        param_map = dict(zip(formal_params, actual_args))
        local_vars = set(sub_mod["variables"].keys())
        pid_guard = BinOp("=", VarRef("_pid"), IntLit(idx))

        # --- Add prefixed local variables ---
        for var_name, var_decl in sub_mod["variables"].items():
            prefixed = f"{inst_name}.{var_name}"
            model.variables[prefixed] = VarDecl(prefixed, copy.deepcopy(var_decl.var_type))

        # --- Add init assignments ---
        for var_name, init_expr in sub_mod["inits"].items():
            target = _resolve_target(var_name, inst_name, param_map, local_vars)
            subst = _subst_expr(init_expr, inst_name, param_map, local_vars)
            model.inits[target] = subst

        # --- Add next assignments (guarded by _pid) ---
        for var_name, next_expr in sub_mod["nexts"].items():
            target = _resolve_target(var_name, inst_name, param_map, local_vars)
            subst = _subst_expr(next_expr, inst_name, param_map, local_vars)
            branches = _to_guarded_branches(subst, pid_guard, VarRef(target))

            if target in shared_nexts:
                shared_nexts[target].append((pid_guard, branches))
            elif target in model.nexts:
                # Main module already has next() for this var — convert and merge
                main_branches = _expr_to_branches(model.nexts[target], VarRef(target))
                shared_nexts[target] = [
                    (BoolLit(True), main_branches),
                    (pid_guard, branches),
                ]
                del model.nexts[target]
            else:
                shared_nexts[target] = [(pid_guard, branches)]

        # --- Stutter for local vars with next() when not running ---
        for var_name in sub_mod["nexts"]:
            target = _resolve_target(var_name, inst_name, param_map, local_vars)
            # Stutter is handled by the final TRUE fallback in _build_merged_next

        # --- FAIRNESS ---
        for fair_expr in sub_mod["fairness"]:
            if isinstance(fair_expr, VarRef) and fair_expr.name == "running":
                model.fairness.append(pid_guard)
            else:
                model.fairness.append(
                    _subst_expr(fair_expr, inst_name, param_map, local_vars)
                )

        # --- Stutter for local vars that HAVE next() when process isn't running ---
        # (Handled via shared_nexts with TRUE : var fallback)

    # Build merged next() assignments for all shared/process variables
    for target, guard_branches_list in shared_nexts.items():
        model.nexts[target] = _build_merged_next(target, guard_branches_list)

    return model


def _resolve_target(var_name: str, inst_name: str,
                    param_map: dict[str, str],
                    local_vars: set[str]) -> str:
    """Resolve a variable name to its flattened target name."""
    if var_name in param_map:
        return param_map[var_name]
    if var_name in local_vars:
        return f"{inst_name}.{var_name}"
    return var_name


def _subst_expr(expr: Expr, inst_name: str,
                param_map: dict[str, str],
                local_vars: set[str]) -> Expr:
    """Recursively substitute variable names in an expression.

    - Formal parameters → actual argument names
    - Local variables → inst_name.var_name
    - Enum values / other names → unchanged
    """
    if isinstance(expr, VarRef):
        new_name = _subst_name(expr.name, inst_name, param_map, local_vars)
        return VarRef(new_name)
    if isinstance(expr, NextRef):
        new_name = _subst_name(expr.name, inst_name, param_map, local_vars)
        return NextRef(new_name)
    if isinstance(expr, BinOp):
        return BinOp(
            expr.op,
            _subst_expr(expr.left, inst_name, param_map, local_vars),
            _subst_expr(expr.right, inst_name, param_map, local_vars),
        )
    if isinstance(expr, UnaryOp):
        return UnaryOp(
            expr.op,
            _subst_expr(expr.operand, inst_name, param_map, local_vars),
        )
    if isinstance(expr, CaseExpr):
        return CaseExpr([
            (
                _subst_expr(cond, inst_name, param_map, local_vars),
                _subst_expr(val, inst_name, param_map, local_vars),
            )
            for cond, val in expr.branches
        ])
    if isinstance(expr, SetExpr):
        return SetExpr([
            _subst_expr(v, inst_name, param_map, local_vars)
            for v in expr.values
        ])
    if isinstance(expr, TemporalUnary):
        return TemporalUnary(
            expr.op,
            _subst_expr(expr.operand, inst_name, param_map, local_vars),
        )
    if isinstance(expr, TemporalBinary):
        return TemporalBinary(
            expr.op,
            _subst_expr(expr.left, inst_name, param_map, local_vars),
            _subst_expr(expr.right, inst_name, param_map, local_vars),
        )
    # IntLit, BoolLit — no substitution needed
    return expr


def _subst_name(name: str, inst_name: str,
                param_map: dict[str, str],
                local_vars: set[str]) -> str:
    """Substitute a single variable name."""
    if name in param_map:
        return param_map[name]
    if name in local_vars:
        return f"{inst_name}.{name}"
    return name  # enum value or global variable


def _to_guarded_branches(expr: Expr, pid_guard: Expr,
                         stutter_ref: VarRef) -> list[tuple[Expr, Expr]]:
    """Convert an expression to a list of case branches guarded by pid.

    For a CaseExpr, each branch condition gets AND-ed with pid_guard.
    The TRUE fallback becomes just the pid_guard (catching remaining cases).
    For other expressions, creates a single guarded branch.
    """
    if isinstance(expr, CaseExpr):
        branches = []
        for cond, val in expr.branches:
            if isinstance(cond, BoolLit) and cond.value:
                # TRUE : val → _pid = idx : val
                branches.append((pid_guard, val))
            else:
                branches.append((BinOp("&", pid_guard, cond), val))
        return branches
    else:
        # Simple expression: _pid = idx : expr
        return [(pid_guard, expr)]


def _expr_to_branches(expr: Expr, stutter_ref: VarRef) -> list[tuple[Expr, Expr]]:
    """Convert an expression to unguarded case branches."""
    if isinstance(expr, CaseExpr):
        return list(expr.branches)
    return [(BoolLit(True), expr)]


def _build_merged_next(target: str,
                       guard_branches_list: list[tuple[Expr, list]]) -> CaseExpr:
    """Build a merged CaseExpr from multiple processes' guarded branches.

    Each process contributes branches guarded by its pid. A final TRUE fallback
    provides stutter semantics (variable keeps its current value).
    """
    all_branches = []
    for _pid_guard, branches in guard_branches_list:
        all_branches.extend(branches)
    # Final stutter fallback
    all_branches.append((BoolLit(True), VarRef(target)))
    return CaseExpr(all_branches)
