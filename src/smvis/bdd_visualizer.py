"""BDD visualization helpers - convert BDD structure to displayable info."""
from __future__ import annotations
from smvis.bdd_engine import BddResult


def get_bdd_summary(result: BddResult) -> dict:
    """Get a summary of the BDD computation for display."""
    # Variable encoding summary
    encoding_summary = []
    total_bdd_vars = 0
    for var_name, enc in result.encoding.items():
        encoding_summary.append({
            "variable": var_name,
            "domain_size": len(enc.domain),
            "bits": enc.n_bits,
            "bdd_vars": ", ".join(enc.bit_vars),
        })
        total_bdd_vars += enc.n_bits

    return {
        "encoding": encoding_summary,
        "total_bdd_vars": total_bdd_vars,
        "total_bdd_vars_with_primed": total_bdd_vars * 2,
        "init_bdd_nodes": result.init_node_count,
        "trans_bdd_nodes": result.trans_node_count,
        "reached_bdd_nodes": result.reached_node_count,
        "total_reachable": result.total_reachable,
        "fixpoint_iterations": len(result.iterations),
        "iterations": [
            {
                "iteration": it.iteration,
                "new_states": it.new_states_count,
                "total_reachable": it.total_reachable,
                "bdd_nodes": it.bdd_node_count,
            }
            for it in result.iterations
        ],
    }


def format_encoding_table(result: BddResult) -> str:
    """Format the binary encoding as a readable table."""
    lines = ["Variable | Domain | Bits | BDD Variables",
             "---------|--------|------|-------------"]
    for var_name, enc in result.encoding.items():
        domain_str = f"{len(enc.domain)} values"
        if len(enc.domain) <= 6:
            domain_str = "{" + ",".join(str(v) for v in enc.domain[:6]) + "}"
        lines.append(
            f"{var_name:12s} | {domain_str:20s} | {enc.n_bits:4d} | "
            f"{', '.join(enc.bit_vars)}"
        )
    return "\n".join(lines)


def format_iteration_log(result: BddResult) -> str:
    """Format the fixpoint iteration log."""
    lines = ["Iter | New States | Total Reachable | BDD Nodes",
             "-----|------------|-----------------|----------"]
    for it in result.iterations:
        lines.append(
            f"{it.iteration:4d} | {it.new_states_count:10d} | "
            f"{it.total_reachable:15d} | {it.bdd_node_count:9d}"
        )
    return "\n".join(lines)
