"""Shared fixtures for smvis tests."""
from __future__ import annotations
import os
import sys
import pytest

# Ensure src/ is on the path so smvis is importable without install
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from smvis.smv_parser import parse_smv_file
from smvis.explicit_engine import explore
from smvis.bdd_engine import build_from_explicit

EXAMPLES_DIR = os.path.join(_ROOT, "examples")


def _model_path(name: str) -> str:
    return os.path.join(EXAMPLES_DIR, name)


# --------------- Parsed models ---------------

@pytest.fixture(scope="session")
def counter_model():
    return parse_smv_file(_model_path("counter.smv"))


@pytest.fixture(scope="session")
def gcd_model():
    return parse_smv_file(_model_path("gcd_01.smv"))


@pytest.fixture(scope="session")
def mult_model():
    return parse_smv_file(_model_path("mult.smv"))


@pytest.fixture(scope="session")
def mutex_model():
    return parse_smv_file(_model_path("mutex.smv"))


# --------------- Explicit results ---------------

@pytest.fixture(scope="session")
def counter_explicit(counter_model):
    return explore(counter_model)


@pytest.fixture(scope="session")
def gcd_explicit(gcd_model):
    return explore(gcd_model)


@pytest.fixture(scope="session")
def mult_explicit(mult_model):
    return explore(mult_model)


@pytest.fixture(scope="session")
def mutex_explicit(mutex_model):
    return explore(mutex_model)


# --------------- BDD results ---------------

def _run_bdd(model, explicit):
    return build_from_explicit(
        model,
        explicit.initial_states,
        explicit.transitions,
        explicit.var_names,
        explicit.state_to_dict,
    )


@pytest.fixture(scope="session")
def counter_bdd(counter_model, counter_explicit):
    return _run_bdd(counter_model, counter_explicit)


@pytest.fixture(scope="session")
def gcd_bdd(gcd_model, gcd_explicit):
    return _run_bdd(gcd_model, gcd_explicit)


@pytest.fixture(scope="session")
def mult_bdd(mult_model, mult_explicit):
    return _run_bdd(mult_model, mult_explicit)


@pytest.fixture(scope="session")
def mutex_bdd(mutex_model, mutex_explicit):
    return _run_bdd(mutex_model, mutex_explicit)
