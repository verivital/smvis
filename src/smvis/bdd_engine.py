"""BDD-based symbolic engine using dd.autoref for state encoding and reachability."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import dd.autoref as _bdd
from smvis.smv_model import SmvModel, VarDecl, BoolType, EnumType, RangeType, get_domain


@dataclass
class EncodingInfo:
    """Binary encoding info for one SMV variable."""
    var_name: str
    n_bits: int
    bit_vars: list[str]          # current-state BDD variable names
    bit_vars_prime: list[str]    # next-state (primed) BDD variable names
    domain: list
    val_to_code: dict            # domain value -> integer code
    code_to_val: dict            # integer code -> domain value


@dataclass
class FixpointIteration:
    """One iteration of the fixpoint computation."""
    iteration: int
    new_states_count: int
    total_reachable: int
    bdd_node_count: int


@dataclass
class BddResult:
    """Result of BDD-based symbolic computation."""
    bdd: _bdd.BDD
    encoding: dict[str, EncodingInfo]
    init_bdd: object             # BDD node for initial states
    trans_bdd: object            # BDD node for transition relation
    reached_bdd: object          # BDD node for all reachable states
    domain_constraint: object    # BDD node for valid domain
    iterations: list[FixpointIteration]
    total_reachable: int
    init_node_count: int
    trans_node_count: int
    reached_node_count: int
    iteration_bdds: list[object] = field(default_factory=list)  # reached BDD after each iteration


def _build_encoding(model: SmvModel) -> tuple[_bdd.BDD, dict[str, EncodingInfo]]:
    """Create BDD variables and encoding info for all SMV variables."""
    bdd = _bdd.BDD()
    encoding = {}

    for var_name, var_decl in model.variables.items():
        domain = get_domain(var_decl)
        n_bits = max(1, math.ceil(math.log2(max(len(domain), 2))))

        # Map domain values to integer codes
        vt = var_decl.var_type
        if isinstance(vt, RangeType):
            val_to_code = {v: v - vt.lo for v in domain}
        elif isinstance(vt, EnumType):
            val_to_code = {v: i for i, v in enumerate(domain)}
        elif isinstance(vt, BoolType):
            val_to_code = {True: 1, False: 0}
        else:
            raise ValueError(f"Unknown type: {vt}")

        code_to_val = {c: v for v, c in val_to_code.items()}

        bit_vars = [f"{var_name}_{i}" for i in range(n_bits)]
        bit_vars_prime = [f"{var_name}_{i}_p" for i in range(n_bits)]

        for bv in bit_vars + bit_vars_prime:
            bdd.declare(bv)

        encoding[var_name] = EncodingInfo(
            var_name=var_name,
            n_bits=n_bits,
            bit_vars=bit_vars,
            bit_vars_prime=bit_vars_prime,
            domain=domain,
            val_to_code=val_to_code,
            code_to_val=code_to_val,
        )

    return bdd, encoding


def _encode_value(bdd: _bdd.BDD, enc: EncodingInfo, value, primed: bool = False) -> object:
    """Encode a single variable assignment as a BDD (conjunction of bit literals)."""
    code = enc.val_to_code[value]
    bits = enc.bit_vars_prime if primed else enc.bit_vars
    result = bdd.true
    for i in range(enc.n_bits):
        bit_val = (code >> i) & 1
        bv = bdd.var(bits[i])
        if bit_val:
            result &= bv
        else:
            result &= ~bv
    return result


def _encode_state(bdd: _bdd.BDD, encoding: dict[str, EncodingInfo],
                  state_dict: dict, primed: bool = False) -> object:
    """Encode a full state as a BDD minterm."""
    result = bdd.true
    for var_name, value in state_dict.items():
        enc = encoding[var_name]
        result &= _encode_value(bdd, enc, value, primed)
    return result


def _build_domain_constraint(bdd: _bdd.BDD, encoding: dict[str, EncodingInfo]) -> object:
    """Build BDD constraining variables to valid domain values."""
    constraint = bdd.true
    for var_name, enc in encoding.items():
        max_code = (1 << enc.n_bits) - 1
        if len(enc.domain) <= max_code:
            # Some bit patterns are invalid; constrain to valid codes
            valid = bdd.false
            for code in range(len(enc.domain)):
                term = bdd.true
                for i in range(enc.n_bits):
                    bv = bdd.var(enc.bit_vars[i])
                    if (code >> i) & 1:
                        term &= bv
                    else:
                        term &= ~bv
                valid |= term
            constraint &= valid
    return constraint


def build_from_explicit(model: SmvModel,
                        initial_states: list[dict],
                        transitions: list[tuple[dict, dict]],
                        var_names: list[str],
                        state_to_dict: dict) -> BddResult:
    """Build BDD representation from explicit-state results.

    Args:
        model: The parsed SMV model
        initial_states: List of initial state keys (tuples)
        transitions: List of (src_key, dst_key) pairs
        var_names: Ordered list of variable names
        state_to_dict: Map from state key to state dict
    """
    bdd, encoding = _build_encoding(model)

    # Build domain constraint
    domain_constraint = _build_domain_constraint(bdd, encoding)

    # Encode initial states
    init_bdd = bdd.false
    for sk in initial_states:
        sd = dict(zip(var_names, sk))
        init_bdd |= _encode_state(bdd, encoding, sd)
    init_bdd &= domain_constraint

    # Encode transition relation
    trans_bdd = bdd.false
    seen_transitions = set()
    for src_key, dst_key in transitions:
        pair = (src_key, dst_key)
        if pair in seen_transitions:
            continue
        seen_transitions.add(pair)

        src_dict = dict(zip(var_names, src_key))
        dst_dict = dict(zip(var_names, dst_key))

        src_bdd = _encode_state(bdd, encoding, src_dict, primed=False)
        dst_bdd = _encode_state(bdd, encoding, dst_dict, primed=True)
        trans_bdd |= (src_bdd & dst_bdd)

    # Build rename maps for forward image computation
    prime_to_current = {}
    current_vars = set()
    for var_name, enc in encoding.items():
        for i in range(enc.n_bits):
            prime_to_current[enc.bit_vars_prime[i]] = enc.bit_vars[i]
            current_vars.add(enc.bit_vars[i])

    # Forward fixpoint reachability
    reached = init_bdd
    iterations = []
    iteration_bdds = [init_bdd]  # Store BDD at each iteration (index 0 = initial)
    iteration = 0

    while True:
        iteration += 1

        # Forward image: next_states(s') = exists s: reached(s) & T(s, s')
        conjoined = reached & trans_bdd
        img = bdd.exist(current_vars, conjoined)
        # Rename primed -> current
        img_renamed = bdd.let(prime_to_current, img)
        # Apply domain constraint
        img_renamed &= domain_constraint

        new_reached = reached | img_renamed

        # Count states via satisfying assignments
        all_current_vars = set()
        for enc in encoding.values():
            all_current_vars.update(enc.bit_vars)
        n_vars = len(all_current_vars)

        old_count = _count_states(reached, bdd, encoding)
        new_count = _count_states(new_reached, bdd, encoding)

        iterations.append(FixpointIteration(
            iteration=iteration,
            new_states_count=new_count - old_count,
            total_reachable=new_count,
            bdd_node_count=len(new_reached),
        ))

        if new_reached == reached:
            break
        reached = new_reached
        iteration_bdds.append(reached)

        # Safety limit
        if iteration > 1000:
            break

    total_reachable = _count_states(reached, bdd, encoding)

    return BddResult(
        bdd=bdd,
        encoding=encoding,
        init_bdd=init_bdd,
        trans_bdd=trans_bdd,
        reached_bdd=reached,
        domain_constraint=domain_constraint,
        iterations=iterations,
        total_reachable=total_reachable,
        init_node_count=len(init_bdd),
        trans_node_count=len(trans_bdd),
        reached_node_count=len(reached),
        iteration_bdds=iteration_bdds,
    )


def _count_states(bdd_node, bdd: _bdd.BDD, encoding: dict[str, EncodingInfo]) -> int:
    """Count the number of concrete states represented by a BDD node."""
    if bdd_node == bdd.false:
        return 0

    # Get all current-state BDD variables
    all_bits = []
    for enc in encoding.values():
        all_bits.extend(enc.bit_vars)

    # Count satisfying assignments
    count = 0
    for assignment in bdd.pick_iter(bdd_node, care_vars=set(all_bits)):
        # Check if this assignment corresponds to a valid domain value for each variable
        valid = True
        for var_name, enc in encoding.items():
            code = 0
            for i in range(enc.n_bits):
                bv = enc.bit_vars[i]
                if bv in assignment and assignment[bv]:
                    code |= (1 << i)
            if code not in enc.code_to_val:
                valid = False
                break
        if valid:
            count += 1
    return count


def decode_bdd_states(bdd_node, bdd: _bdd.BDD, encoding: dict[str, EncodingInfo],
                      var_names: list[str]) -> list[dict]:
    """Decode a BDD node into a list of concrete state dicts."""
    states = []
    all_bits = []
    for enc in encoding.values():
        all_bits.extend(enc.bit_vars)

    for assignment in bdd.pick_iter(bdd_node, care_vars=set(all_bits)):
        state = {}
        valid = True
        for var_name in var_names:
            enc = encoding[var_name]
            code = 0
            for i in range(enc.n_bits):
                bv = enc.bit_vars[i]
                if bv in assignment and assignment[bv]:
                    code |= (1 << i)
            if code in enc.code_to_val:
                state[var_name] = enc.code_to_val[code]
            else:
                valid = False
                break
        if valid:
            states.append(state)
    return states


def get_bdd_structure(bdd_node, bdd: _bdd.BDD, max_nodes: int = 200) -> list[dict]:
    """Extract BDD DAG structure with ROBDD hierarchical positions.

    Nodes are positioned by variable ordering level (y-axis) with
    horizontal spacing within each level. Terminal nodes go at the bottom.
    Returns list of node/edge dicts with ``position`` for preset layout.
    """
    elements = []
    visited = set()
    node_count = [0]
    # Collect nodes by their variable level for positioning
    level_nodes: dict[str, list[int]] = {}  # var_name -> [node_ids]
    node_vars: dict[int, str] = {}  # node_id -> var_name or "T"/"F"

    def _traverse(node):
        node_id = int(node)
        if node_id in visited or node_count[0] > max_nodes:
            return
        visited.add(node_id)
        node_count[0] += 1

        if node == bdd.true:
            elements.append({"data": {"id": str(node_id), "label": "1"},
                             "classes": "terminal-true"})
            node_vars[node_id] = "__terminal__"
            level_nodes.setdefault("__terminal__", []).append(node_id)
            return
        if node == bdd.false:
            elements.append({"data": {"id": str(node_id), "label": "0"},
                             "classes": "terminal-false"})
            node_vars[node_id] = "__terminal__"
            level_nodes.setdefault("__terminal__", []).append(node_id)
            return

        var = node.var
        elements.append({"data": {"id": str(node_id), "label": var},
                         "classes": "internal"})
        node_vars[node_id] = var
        level_nodes.setdefault(var, []).append(node_id)

        # High child (then)
        high = node.high
        high_id = int(high)
        _traverse(high)
        elements.append({"data": {"source": str(node_id), "target": str(high_id)},
                         "classes": "high-edge"})

        # Low child (else)
        low = node.low
        low_id = int(low)
        _traverse(low)
        elements.append({"data": {"source": str(node_id), "target": str(low_id)},
                         "classes": "low-edge"})

    try:
        _traverse(bdd_node)
    except (AttributeError, TypeError):
        elements = [{"data": {"id": "info",
                              "label": f"BDD: {len(bdd_node)} nodes"},
                     "classes": "info"}]
        return elements

    # Remove edges that reference nodes truncated by max_nodes limit
    visited_ids = {str(nid) for nid in visited}
    elements = [e for e in elements
                if "source" not in e.get("data", {})
                or (e["data"]["source"] in visited_ids and e["data"]["target"] in visited_ids)]

    # Compute ROBDD hierarchical positions
    # Get variable ordering from the BDD manager
    try:
        var_order = list(bdd.vars)
    except Exception:
        var_order = []

    # Build ordered list of levels that actually appear in the BDD
    ordered_levels = []
    for v in var_order:
        if v in level_nodes:
            ordered_levels.append(v)
    # Terminal level at the bottom
    if "__terminal__" in level_nodes:
        ordered_levels.append("__terminal__")

    # Assign y-coordinate by level, x by position within level
    y_spacing = 80
    x_spacing = 80
    node_positions: dict[str, dict] = {}

    for level_idx, var_name in enumerate(ordered_levels):
        nodes_at_level = level_nodes[var_name]
        n = len(nodes_at_level)
        total_width = (n - 1) * x_spacing
        start_x = -total_width / 2
        for i, nid in enumerate(nodes_at_level):
            node_positions[str(nid)] = {
                "x": round(start_x + i * x_spacing),
                "y": level_idx * y_spacing,
            }

    # Apply positions to node elements
    for elem in elements:
        nid = elem["data"].get("id")
        if nid and nid in node_positions and "source" not in elem["data"]:
            elem["position"] = node_positions[nid]

    return elements


def build_truth_table(bdd_node, bdd: _bdd.BDD, encoding: dict[str, EncodingInfo],
                      var_names: list[str], max_rows: int = 64) -> list[dict] | None:
    """Build a truth table showing satisfying assignments of a BDD.

    Returns a list of row dicts with SMV variable values and "result" column,
    or None if the BDD has too many variables (> 6 relevant vars = 64 rows).
    """
    # Get all current-state BDD variables involved
    all_bits = []
    for vn in var_names:
        enc = encoding[vn]
        all_bits.extend(enc.bit_vars)

    if len(all_bits) > 10:
        return None  # Too many BDD vars for a full truth table

    # Enumerate satisfying assignments
    rows = []
    count = 0
    for assignment in bdd.pick_iter(bdd_node, care_vars=set(all_bits)):
        if count >= max_rows:
            break
        row = {}
        valid = True
        for vn in var_names:
            enc = encoding[vn]
            code = 0
            for i in range(enc.n_bits):
                bv = enc.bit_vars[i]
                if bv in assignment and assignment[bv]:
                    code |= (1 << i)
            if code in enc.code_to_val:
                row[vn] = enc.code_to_val[code]
            else:
                valid = False
                break
        if valid:
            rows.append(row)
            count += 1

    return rows if rows else None
