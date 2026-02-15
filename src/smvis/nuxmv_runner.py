"""nuXmv model checker integration — batch checking, trace parsing, interactive sessions."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import logging

log = logging.getLogger("smvis.nuxmv")

# Locate the nuXmv binary relative to this package
_NUXMV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "bin", "nuxmv", "nuXmv.exe",
)


def nuxmv_available() -> bool:
    """Check if the nuXmv binary exists."""
    return os.path.isfile(_NUXMV_PATH)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    """A counterexample trace from nuXmv."""
    states: list[dict[str, str]]   # [{var: value, ...}, ...]
    loop_start: int | None = None  # index of loop-back state (lasso traces)
    description: str = ""          # "CTL Counterexample" / "LTL Counterexample"


@dataclass
class SpecResult:
    """Result of checking a single specification."""
    spec_text: str       # "AG (x <= count_max)"
    spec_kind: str       # "CTLSPEC", "LTLSPEC", "INVARSPEC", "SPEC"
    passed: bool
    trace: Trace | None = None  # counterexample if failed


@dataclass
class NuxmvResult:
    """Full result of a batch nuXmv run."""
    specs: list[SpecResult] = field(default_factory=list)
    raw_output: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Regex for spec result lines
_RE_SPEC = re.compile(
    r"-- (specification|invariant)\s+(.+?)\s+is\s+(true|false)",
)


def parse_spec_results(output: str) -> list[SpecResult]:
    """Parse '-- specification/invariant ... is true/false' lines from nuXmv output."""
    results = []
    for m in _RE_SPEC.finditer(output):
        kind_word = m.group(1)  # "specification" or "invariant"
        spec_text = m.group(2).strip()
        passed = m.group(3) == "true"

        # Determine spec kind from context
        if kind_word == "invariant":
            spec_kind = "INVARSPEC"
        else:
            spec_kind = "SPEC"  # refined by parse_output_paired

        results.append(SpecResult(
            spec_text=spec_text,
            spec_kind=spec_kind,
            passed=passed,
        ))
    return results


def parse_output_paired(output: str) -> list[SpecResult]:
    """Parse nuXmv output pairing each spec result with its inline trace.

    This is the primary parser: it scans the output sequentially, matching
    each '-- specification/invariant ... is false' with the trace that
    immediately follows it.
    """
    results: list[SpecResult] = []
    lines = output.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        m = _RE_SPEC.match(line) or _RE_SPEC.search(line)
        if m:
            kind_word = m.group(1)
            spec_text = m.group(2).strip()
            passed = m.group(3) == "true"
            spec_kind = "INVARSPEC" if kind_word == "invariant" else "SPEC"

            trace = None
            if not passed:
                # Look ahead for inline trace
                j = i + 1
                trace_lines = []
                found_trace = False
                while j < len(lines):
                    tl = lines[j].strip()
                    if tl.startswith("Trace Description:"):
                        found_trace = True
                    if found_trace:
                        # Collect until next spec result or end
                        if (_RE_SPEC.search(tl) and
                                not tl.startswith("Trace") and
                                not tl.startswith("--")):
                            break
                        # Stop at next spec result line
                        if _RE_SPEC.search(tl):
                            break
                        trace_lines.append(lines[j])
                    elif tl.startswith("-- as demonstrated"):
                        pass  # skip this line, trace follows
                    elif not tl or tl.startswith("nuXmv >"):
                        if found_trace:
                            break
                    j += 1

                if trace_lines:
                    trace_text = "\n".join(trace_lines)
                    parsed = parse_text_traces(trace_text)
                    if parsed:
                        trace = parsed[0]
                        # Infer spec kind from trace description
                        desc = trace.description.lower()
                        if "ltl" in desc:
                            spec_kind = "LTLSPEC"
                        elif "ag alpha" in desc:
                            spec_kind = "INVARSPEC"
                        elif "ctl" in desc:
                            spec_kind = "CTLSPEC"

            results.append(SpecResult(
                spec_text=spec_text,
                spec_kind=spec_kind,
                passed=passed,
                trace=trace,
            ))
        i += 1

    return results


def parse_xml_traces(output: str) -> list[Trace]:
    """Parse XML counterexample blocks from 'show_traces -p 4' output."""
    traces = []

    # Extract all <counter-example>...</counter-example> blocks
    pattern = re.compile(
        r"<counter-example\b[^>]*>.*?</counter-example>",
        re.DOTALL,
    )

    for match in pattern.finditer(output):
        xml_text = match.group(0)
        try:
            root = ET.fromstring(xml_text)
            desc = root.get("desc", "")

            states = []
            for node in root.findall("node"):
                state_elem = node.find("state")
                if state_elem is not None:
                    state = {}
                    for val in state_elem.findall("value"):
                        var_name = val.get("variable", "")
                        var_val = (val.text or "").strip()
                        if var_name:
                            state[var_name] = var_val
                    states.append(state)

            # Parse loop info
            loop_start = None
            loops_elem = root.find("loops")
            if loops_elem is not None and loops_elem.text and loops_elem.text.strip():
                try:
                    # nuXmv uses 1-based indexing for loops
                    loop_val = int(loops_elem.text.strip())
                    loop_start = loop_val - 1  # convert to 0-based
                except ValueError:
                    pass

            traces.append(Trace(states=states, loop_start=loop_start, description=desc))
        except ET.ParseError as e:
            log.warning("Failed to parse XML trace: %s", e)
            continue

    return traces


def parse_text_traces(output: str) -> list[Trace]:
    """Parse text-format counterexample traces (fallback if XML unavailable)."""
    traces = []
    current_states: list[dict[str, str]] = []
    current_state: dict[str, str] = {}
    current_desc = ""
    loop_start: int | None = None
    in_trace = False

    for line in output.split("\n"):
        line = line.strip()

        # Trace header
        if line.startswith("Trace Description:"):
            # Save previous trace if any
            if in_trace and current_states:
                if current_state:
                    current_states.append(current_state)
                traces.append(Trace(
                    states=current_states,
                    loop_start=loop_start,
                    description=current_desc,
                ))
            current_states = []
            current_state = {}
            current_desc = line.split(":", 1)[1].strip()
            loop_start = None
            in_trace = True
            continue

        if not in_trace:
            continue

        # Loop marker
        if line == "-- Loop starts here":
            # Account for pending current_state not yet appended
            loop_start = len(current_states) + (1 if current_state else 0)
            continue

        # State header
        state_match = re.match(r"-> State: (\d+)\.(\d+) <-", line)
        if state_match:
            if current_state:
                current_states.append(current_state)
            # For incremental traces, copy previous state as base
            if current_states:
                current_state = dict(current_states[-1])
            else:
                current_state = {}
            continue

        # Variable assignment
        assign_match = re.match(r"(\w+)\s*=\s*(.+)", line)
        if assign_match and in_trace:
            current_state[assign_match.group(1)] = assign_match.group(2).strip()
            continue

        # End of trace (empty line or new spec result)
        if line.startswith("-- specification") or line.startswith("-- invariant"):
            if current_state:
                current_states.append(current_state)
            if current_states:
                traces.append(Trace(
                    states=current_states,
                    loop_start=loop_start,
                    description=current_desc,
                ))
            current_states = []
            current_state = {}
            in_trace = False

    # Final trace
    if in_trace and (current_states or current_state):
        if current_state:
            current_states.append(current_state)
        if current_states:
            traces.append(Trace(
                states=current_states,
                loop_start=loop_start,
                description=current_desc,
            ))

    return traces


def _match_traces_to_specs(specs: list[SpecResult], traces: list[Trace]):
    """Associate parsed traces with their corresponding failed specs."""
    # Traces appear in order of failed specs
    trace_idx = 0
    for spec in specs:
        if not spec.passed and trace_idx < len(traces):
            spec.trace = traces[trace_idx]
            # Infer spec kind from trace description
            desc = traces[trace_idx].description.lower()
            if "ltl" in desc:
                spec.spec_kind = "LTLSPEC"
            elif "ctl" in desc or "ag alpha" in desc:
                if "ag alpha" in desc:
                    spec.spec_kind = "INVARSPEC"
                else:
                    spec.spec_kind = "CTLSPEC"
            trace_idx += 1


# ---------------------------------------------------------------------------
# Batch model checking
# ---------------------------------------------------------------------------

def run_batch_check(smv_text: str, nuxmv_path: str | None = None) -> NuxmvResult:
    """Run all specs through nuXmv and parse results.

    Writes SMV text to a temp file, runs nuXmv in interactive mode with
    a command script, parses results and counterexample traces.
    """
    nuxmv = nuxmv_path or _NUXMV_PATH
    if not os.path.isfile(nuxmv):
        return NuxmvResult(error=f"nuXmv binary not found at: {nuxmv}")

    # Write model to temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".smv", delete=False, encoding="utf-8",
    )
    try:
        tmp.write(smv_text)
        tmp.close()

        # Build command script
        script = (
            "go\n"
            "check_ctlspec\n"
            "check_ltlspec\n"
            "check_invar\n"
            "show_traces -p 4\n"  # XML format for reliable parsing
            "quit\n"
        )

        # Run nuXmv
        proc = subprocess.run(
            [nuxmv, "-int", tmp.name],
            input=script,
            capture_output=True,
            text=True,
            timeout=60,
        )
        raw = proc.stdout + (proc.stderr or "")

        # Detect model-building errors (e.g., value out of range)
        error = None
        _error_patterns = [
            re.compile(r"cannot assign value .+ to variable .+"),
            re.compile(r"A model must be built before"),
            re.compile(r"type error"),
            re.compile(r"undefined.*variable", re.IGNORECASE),
            re.compile(r"(?:syntax|parse) error", re.IGNORECASE),
        ]
        for pat in _error_patterns:
            m = pat.search(raw)
            if m:
                # Extract the full error line for context
                for line in raw.split("\n"):
                    if pat.search(line):
                        error = line.strip()
                        break
                break

        # Parse spec results paired with inline traces
        specs = parse_output_paired(raw)

        # Enrich traces with XML data if available (XML has full state info)
        xml_traces = parse_xml_traces(raw)
        if xml_traces:
            trace_idx = 0
            for spec in specs:
                if spec.trace is not None and trace_idx < len(xml_traces):
                    # XML traces have complete state data (no incremental diffs)
                    spec.trace = xml_traces[trace_idx]
                    trace_idx += 1

        return NuxmvResult(specs=specs, raw_output=raw, error=error)

    except subprocess.TimeoutExpired:
        return NuxmvResult(error="nuXmv timed out after 60 seconds")
    except Exception as e:
        return NuxmvResult(error=str(e))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

class NuxmvSession:
    """Manages a long-lived nuXmv -int subprocess for interactive use."""

    def __init__(self, nuxmv_path: str | None = None):
        self.nuxmv_path = nuxmv_path or _NUXMV_PATH
        self.process: subprocess.Popen | None = None
        self._output_buffer: list[str] = []
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._running = False

    @property
    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, model_path: str | None = None) -> bool:
        """Start a nuXmv interactive session.

        Args:
            model_path: Optional path to .smv file to load.

        Returns:
            True if started successfully.
        """
        if self.is_alive:
            self.stop()

        if not os.path.isfile(self.nuxmv_path):
            with self._lock:
                self._output_buffer.append(f"Error: nuXmv not found at {self.nuxmv_path}\n")
            return False

        args = [self.nuxmv_path, "-int"]
        if model_path:
            args.append(model_path)

        try:
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffered
            )
            self._running = True
            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._reader_thread.start()
            return True
        except Exception as e:
            with self._lock:
                self._output_buffer.append(f"Error starting nuXmv: {e}\n")
            return False

    def _read_loop(self):
        """Background thread that reads stdout line by line."""
        try:
            while self._running and self.process and self.process.stdout:
                line = self.process.stdout.readline()
                if not line:
                    break
                with self._lock:
                    self._output_buffer.append(line)
        except Exception:
            pass
        finally:
            with self._lock:
                self._output_buffer.append("\n[Session ended]\n")
            self._running = False

    def send_command(self, cmd: str) -> bool:
        """Send a command to the nuXmv process.

        Returns:
            True if command was sent successfully.
        """
        if not self.is_alive or not self.process or not self.process.stdin:
            return False
        try:
            self.process.stdin.write(cmd.rstrip("\n") + "\n")
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def get_new_output(self) -> str:
        """Get and clear any new output from the subprocess."""
        with self._lock:
            if not self._output_buffer:
                return ""
            lines = list(self._output_buffer)
            self._output_buffer.clear()
        return "".join(lines)

    def stop(self):
        """Stop the nuXmv subprocess."""
        self._running = False
        if self.process:
            try:
                self.process.stdin.write("quit\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None


# ---------------------------------------------------------------------------
# Temp file helper for interactive sessions
# ---------------------------------------------------------------------------

def write_temp_model(smv_text: str) -> str:
    """Write SMV text to a temp file, return path. Caller must clean up."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".smv", delete=False, encoding="utf-8",
    )
    tmp.write(smv_text)
    tmp.close()
    return tmp.name
