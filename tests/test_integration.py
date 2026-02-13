"""End-to-end integration tests."""
import os
import pytest
from smvis.smv_parser import parse_smv_file
from smvis.explicit_engine import explore
from smvis.bdd_engine import build_from_explicit, get_bdd_structure
from smvis.graph_builder import build_elements
from smvis.bdd_visualizer import get_bdd_summary

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)

MODEL_FILES = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".smv")]


@pytest.mark.parametrize("smv_file", MODEL_FILES)
def test_full_pipeline(smv_file):
    """Parse -> explore -> BDD -> graph -> summary without error."""
    path = os.path.join(EXAMPLES_DIR, smv_file)
    model = parse_smv_file(path)
    explicit = explore(model)
    bdd = build_from_explicit(
        model, explicit.initial_states, explicit.transitions,
        explicit.var_names, explicit.state_to_dict,
    )
    elements = build_elements(explicit, reachable_only=True)
    summary = get_bdd_summary(bdd)

    assert bdd.total_reachable == len(explicit.reachable_states)
    assert len(elements) > 0
    assert summary["total_reachable"] > 0


@pytest.mark.parametrize("smv_file", MODEL_FILES)
def test_bdd_structure_extraction(smv_file):
    """BDD structure extraction works for all selector types."""
    path = os.path.join(EXAMPLES_DIR, smv_file)
    model = parse_smv_file(path)
    explicit = explore(model)
    bdd = build_from_explicit(
        model, explicit.initial_states, explicit.transitions,
        explicit.var_names, explicit.state_to_dict,
    )
    for bdd_node in [bdd.init_bdd, bdd.reached_bdd, bdd.trans_bdd]:
        elems = get_bdd_structure(bdd_node, bdd.bdd)
        assert isinstance(elems, list)
