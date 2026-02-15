"""Tests for nuXmv runner — trace parsing, batch checking, interactive sessions."""
import os
import time
import pytest

from smvis.nuxmv_runner import (
    parse_spec_results,
    parse_xml_traces,
    parse_text_traces,
    _match_traces_to_specs,
    run_batch_check,
    nuxmv_available,
    NuxmvSession,
    write_temp_model,
    Trace,
    SpecResult,
    _NUXMV_PATH,
)

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nuxmv_path():
    """Skip tests if nuXmv binary is not available."""
    if not os.path.isfile(_NUXMV_PATH):
        pytest.skip("nuXmv binary not found")
    return _NUXMV_PATH


# ---------------------------------------------------------------------------
# Unit tests: parse_spec_results
# ---------------------------------------------------------------------------

class TestParseSpecResults:
    def test_parse_true_spec(self):
        output = "-- specification AG (x <= count_max)  is true\n"
        results = parse_spec_results(output)
        assert len(results) == 1
        assert results[0].passed is True
        assert "AG" in results[0].spec_text
        assert "x <= count_max" in results[0].spec_text

    def test_parse_false_spec(self):
        output = "-- specification AG x = 0  is false\n"
        results = parse_spec_results(output)
        assert len(results) == 1
        assert results[0].passed is False

    def test_parse_invariant(self):
        output = "-- invariant x <= count_max  is true\n"
        results = parse_spec_results(output)
        assert len(results) == 1
        assert results[0].spec_kind == "INVARSPEC"
        assert results[0].passed is True

    def test_parse_false_invariant(self):
        output = "-- invariant x <= count_max / 2  is false\n"
        results = parse_spec_results(output)
        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].spec_kind == "INVARSPEC"

    def test_parse_multiple_specs(self):
        output = (
            "-- specification AG x = 0  is false\n"
            "-- specification EF x = count_max  is true\n"
            "-- specification AG (AF (mode = off & x = 0))  is true\n"
            "-- invariant x <= count_max  is true\n"
        )
        results = parse_spec_results(output)
        assert len(results) == 4
        assert results[0].passed is False
        assert results[1].passed is True
        assert results[2].passed is True
        assert results[3].passed is True
        assert results[3].spec_kind == "INVARSPEC"

    def test_parse_empty_output(self):
        results = parse_spec_results("")
        assert results == []

    def test_parse_output_with_noise(self):
        output = (
            "*** This is nuXmv 2.1.0\n"
            "*** Copyright blah blah\n"
            "nuXmv > nuXmv > -- specification AG x = 0  is false\n"
            "-- as demonstrated by the following execution sequence\n"
            "Trace Description: CTL Counterexample\n"
            "Trace Type: Counterexample\n"
            "  -> State: 1.1 <-\n"
            "    mode = off\n"
        )
        results = parse_spec_results(output)
        assert len(results) == 1
        assert results[0].passed is False


# ---------------------------------------------------------------------------
# Unit tests: parse_xml_traces
# ---------------------------------------------------------------------------

class TestParseXmlTraces:
    def test_simple_trace(self):
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
<counter-example type="0" id="1" desc="CTL Counterexample" >
    <node>
        <state id="1">
            <value variable="mode">off</value>
            <value variable="x">0</value>
        </state>
    </node>
    <node>
        <state id="2">
            <value variable="mode">on</value>
            <value variable="x">1</value>
        </state>
    </node>
    <loops> </loops>
</counter-example>'''
        traces = parse_xml_traces(xml)
        assert len(traces) == 1
        t = traces[0]
        assert len(t.states) == 2
        assert t.states[0]["mode"] == "off"
        assert t.states[0]["x"] == "0"
        assert t.states[1]["mode"] == "on"
        assert t.states[1]["x"] == "1"
        assert t.loop_start is None
        assert t.description == "CTL Counterexample"

    def test_trace_with_loop(self):
        xml = '''<counter-example type="0" id="1" desc="LTL Counterexample" >
    <node>
        <state id="1">
            <value variable="mode">off</value>
            <value variable="x">0</value>
        </state>
    </node>
    <node>
        <state id="2">
            <value variable="mode">on</value>
            <value variable="x">1</value>
        </state>
    </node>
    <node>
        <state id="3">
            <value variable="mode">off</value>
            <value variable="x">0</value>
        </state>
    </node>
    <loops> 1 </loops>
</counter-example>'''
        traces = parse_xml_traces(xml)
        assert len(traces) == 1
        t = traces[0]
        assert len(t.states) == 3
        assert t.loop_start == 0  # 1-based → 0-based
        assert t.description == "LTL Counterexample"

    def test_multiple_traces(self):
        xml = '''<counter-example type="0" id="1" desc="CTL Counterexample" >
    <node>
        <state id="1">
            <value variable="x">0</value>
        </state>
    </node>
    <loops> </loops>
</counter-example>
<counter-example type="0" id="2" desc="LTL Counterexample" >
    <node>
        <state id="1">
            <value variable="x">5</value>
        </state>
    </node>
    <loops> 1 </loops>
</counter-example>'''
        traces = parse_xml_traces(xml)
        assert len(traces) == 2
        assert traces[0].states[0]["x"] == "0"
        assert traces[1].states[0]["x"] == "5"
        assert traces[1].loop_start == 0

    def test_empty_xml(self):
        traces = parse_xml_traces("no xml here")
        assert traces == []

    def test_embedded_in_nuxmv_output(self):
        output = '''nuXmv > <?xml version="1.0" encoding="UTF-8"?>
<counter-example type="0" id="1" desc="CTL Counterexample" >
    <node>
        <state id="1">
            <value variable="process1">idle</value>
            <value variable="flag1">FALSE</value>
        </state>
    </node>
    <loops> </loops>
</counter-example>
nuXmv > '''
        traces = parse_xml_traces(output)
        assert len(traces) == 1
        assert traces[0].states[0]["process1"] == "idle"
        assert traces[0].states[0]["flag1"] == "FALSE"


# ---------------------------------------------------------------------------
# Unit tests: parse_text_traces
# ---------------------------------------------------------------------------

class TestParseTextTraces:
    def test_simple_trace(self):
        text = """Trace Description: CTL Counterexample
Trace Type: Counterexample
  -> State: 1.1 <-
    mode = off
    press = FALSE
    x = 0
  -> State: 1.2 <-
    press = TRUE
  -> State: 1.3 <-
    mode = on
    press = FALSE
-- specification EF x = count_max  is true"""
        traces = parse_text_traces(text)
        assert len(traces) == 1
        t = traces[0]
        assert len(t.states) == 3
        assert t.states[0]["mode"] == "off"
        assert t.states[0]["x"] == "0"
        # State 1.2 should inherit from 1.1 and update press
        assert t.states[1]["press"] == "TRUE"
        assert t.states[1]["mode"] == "off"  # inherited
        # State 1.3 updates mode and press
        assert t.states[2]["mode"] == "on"
        assert t.states[2]["press"] == "FALSE"
        assert t.loop_start is None

    def test_trace_with_loop(self):
        text = """Trace Description: LTL Counterexample
Trace Type: Counterexample
  -- Loop starts here
  -> State: 1.1 <-
    mode = off
    x = 0
  -> State: 1.2 <-
    mode = on
    x = 1
"""
        traces = parse_text_traces(text)
        assert len(traces) == 1
        assert traces[0].loop_start == 0
        assert len(traces[0].states) == 2

    def test_loop_in_middle(self):
        text = """Trace Description: LTL Counterexample
Trace Type: Counterexample
  -> State: 1.1 <-
    x = 0
  -> State: 1.2 <-
    x = 1
  -- Loop starts here
  -> State: 1.3 <-
    x = 2
  -> State: 1.4 <-
    x = 3
"""
        traces = parse_text_traces(text)
        assert len(traces) == 1
        assert traces[0].loop_start == 2
        assert len(traces[0].states) == 4

    def test_empty_output(self):
        traces = parse_text_traces("")
        assert traces == []


# ---------------------------------------------------------------------------
# Unit tests: _match_traces_to_specs
# ---------------------------------------------------------------------------

class TestMatchTracesToSpecs:
    def test_match_single_trace(self):
        specs = [
            SpecResult("AG x = 0", "SPEC", False),
            SpecResult("EF x = 1", "SPEC", True),
        ]
        traces = [
            Trace(states=[{"x": "0"}], description="CTL Counterexample"),
        ]
        _match_traces_to_specs(specs, traces)
        assert specs[0].trace is not None
        assert specs[0].spec_kind == "CTLSPEC"
        assert specs[1].trace is None

    def test_match_multiple_traces(self):
        specs = [
            SpecResult("AG x = 0", "SPEC", False),
            SpecResult("AG y = 1", "SPEC", True),
            SpecResult("G x = 0", "SPEC", False),
        ]
        traces = [
            Trace(states=[{"x": "0"}], description="CTL Counterexample"),
            Trace(states=[{"x": "5"}], description="LTL Counterexample"),
        ]
        _match_traces_to_specs(specs, traces)
        assert specs[0].trace is traces[0]
        assert specs[0].spec_kind == "CTLSPEC"
        assert specs[1].trace is None
        assert specs[2].trace is traces[1]
        assert specs[2].spec_kind == "LTLSPEC"

    def test_match_invariant_trace(self):
        specs = [
            SpecResult("x <= 5", "INVARSPEC", False),
        ]
        traces = [
            Trace(states=[{"x": "6"}], description="AG alpha Counterexample"),
        ]
        _match_traces_to_specs(specs, traces)
        assert specs[0].trace is not None
        assert specs[0].spec_kind == "INVARSPEC"


# ---------------------------------------------------------------------------
# Integration tests: run_batch_check (require nuXmv binary)
# ---------------------------------------------------------------------------

class TestBatchCheck:
    def test_counter_model(self, nuxmv_path):
        """counter.smv should have a mix of passing and failing specs."""
        path = os.path.join(EXAMPLES_DIR, "counter.smv")
        with open(path) as f:
            text = f.read()
        result = run_batch_check(text, nuxmv_path)
        assert result.error is None
        assert len(result.specs) > 0
        # Some should pass, some should fail
        passed = [s for s in result.specs if s.passed]
        failed = [s for s in result.specs if not s.passed]
        assert len(passed) > 0, "Expected some passing specs"
        assert len(failed) > 0, "Expected some failing specs"
        # Failed specs should have traces
        for s in failed:
            assert s.trace is not None, f"Failed spec '{s.spec_text}' missing trace"
            assert len(s.trace.states) > 0

    def test_mutex_model(self, nuxmv_path):
        """mutex.smv has specific known results."""
        path = os.path.join(EXAMPLES_DIR, "mutex.smv")
        with open(path) as f:
            text = f.read()
        result = run_batch_check(text, nuxmv_path)
        assert result.error is None
        assert len(result.specs) > 0

    def test_all_models(self, nuxmv_path):
        """Batch check runs without crashing on all example models."""
        # Some models (mult.smv) have range errors that nuXmv detects;
        # we just verify no Python exceptions are raised.
        # Models where nuXmv detects assignment out of declared range
        known_errors = {"abs_diff.smv", "fibonacci.smv"}
        for fname in sorted(os.listdir(EXAMPLES_DIR)):
            if not fname.endswith(".smv"):
                continue
            path = os.path.join(EXAMPLES_DIR, fname)
            with open(path) as f:
                text = f.read()
            result = run_batch_check(text, nuxmv_path)
            if fname in known_errors:
                assert result.error is not None, (
                    f"Expected error on {fname} but got none"
                )
            else:
                assert result.error is None, f"Error on {fname}: {result.error}"

    def test_trace_states_are_valid(self, nuxmv_path):
        """Trace states from nuXmv should match our explicit engine states."""
        from smvis.smv_parser import parse_smv_file
        from smvis.explicit_engine import explore

        path = os.path.join(EXAMPLES_DIR, "counter.smv")
        model = parse_smv_file(path)
        explicit = explore(model)
        var_names = set(explicit.var_names)

        with open(path) as f:
            text = f.read()
        result = run_batch_check(text, nuxmv_path)

        def _normalize(val):
            """Normalize values for comparison (nuXmv TRUE/FALSE vs Python True/False)."""
            s = str(val)
            if s in ("TRUE", "True"):
                return "True"
            if s in ("FALSE", "False"):
                return "False"
            return s

        # Check that trace states exist in our state space
        for spec in result.specs:
            if spec.trace:
                for i, ts in enumerate(spec.trace.states):
                    # Only compare variables (skip DEFINEs like count_max)
                    trace_vars = {k: v for k, v in ts.items() if k in var_names}
                    if not trace_vars:
                        continue
                    matched = False
                    for state, sd in explicit.state_to_dict.items():
                        if all(_normalize(sd.get(k)) == _normalize(v)
                               for k, v in trace_vars.items()):
                            matched = True
                            break
                    assert matched, (
                        f"Trace state {i} not found in explicit states: {trace_vars}"
                    )

    def test_invalid_model(self, nuxmv_path):
        """Invalid SMV text should produce an error or empty specs."""
        result = run_batch_check("not valid smv", nuxmv_path)
        # nuXmv should either error or produce no specs
        assert result.error is not None or len(result.specs) == 0

    def test_missing_binary(self):
        """Non-existent binary path should return error."""
        result = run_batch_check("MODULE main", "/nonexistent/nuxmv")
        assert result.error is not None


# ---------------------------------------------------------------------------
# Integration tests: NuxmvSession (require nuXmv binary)
# ---------------------------------------------------------------------------

class TestNuxmvSession:
    def test_start_stop(self, nuxmv_path):
        """Session starts and stops cleanly."""
        session = NuxmvSession(nuxmv_path)
        assert session.start()
        assert session.is_alive
        time.sleep(1)  # let banner output appear
        output = session.get_new_output()
        assert "nuXmv" in output
        session.stop()
        time.sleep(0.5)
        assert not session.is_alive

    def test_send_go_command(self, nuxmv_path):
        """Send 'go' with a model loaded."""
        path = os.path.join(EXAMPLES_DIR, "mutex.smv")
        session = NuxmvSession(nuxmv_path)
        assert session.start(path)
        time.sleep(1)
        session.get_new_output()  # clear banner

        session.send_command("go")
        time.sleep(2)
        output = session.get_new_output()
        # 'go' should complete without error (empty or just prompts)
        assert "Error" not in output or "error" not in output.lower()

        session.send_command("check_ctlspec")
        time.sleep(3)
        output = session.get_new_output()
        assert "specification" in output.lower() or "is true" in output or "is false" in output

        session.stop()

    def test_write_temp_model(self):
        """write_temp_model creates a file and returns its path."""
        text = "MODULE main\n  VAR x : boolean;\n"
        path = write_temp_model(text)
        try:
            assert os.path.isfile(path)
            with open(path) as f:
                assert f.read() == text
        finally:
            os.unlink(path)
