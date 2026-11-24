"""Collecting tests must not load reference data for deselected tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest


# One all-variant corpus case exercises that corpus's pinned reference fixture
# exactly as all 174 do; running the rest again here only repeats Isovar work.
ONE_CORPUS_CASE = "not every_literal_allele or ABI3BP-chr3-100850671"

# Default deadlines per selection mode (#568). The full real-RNA Isovar
# selection runs ~100s on an idle single worker, so 180s left too little
# headroom under concurrent xdist workers (TEST_SH_MAX=4); 360s preserves a
# real upper bound without failing healthy runs under normal workstation load.
COLLECT_ONLY_TIMEOUT_SECONDS = 60.0
ISOVAR_SELECTION_TIMEOUT_SECONDS = 360.0

# Limit native thread pools inside nested test processes so concurrent xdist
# workers do not oversubscribe BLAS/OpenMP/Accelerate threads (#568).
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


def _decode_stream(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _nested_timeout_seconds(selection):
    override = os.environ.get("TOPIARY_NESTED_TEST_TIMEOUT")
    if override:
        timeout = float(override)
        if timeout <= 0:
            raise ValueError("TOPIARY_NESTED_TEST_TIMEOUT must be positive")
        return timeout
    if "--collect-only" in selection:
        return COLLECT_ONLY_TIMEOUT_SECONDS
    return ISOVAR_SELECTION_TIMEOUT_SECONDS


def summarize_nested_progress(progress_log: Path) -> str:
    """Summarize active and slowest test cases from a child JSONL progress log."""
    if not progress_log.exists():
        return "Progress log: absent (child timed out before pytest started running tests)"
    active_nodeid = None
    active_started = None
    completed: dict[str, float] = {}
    last_elapsed = 0.0
    malformed = 0
    for line in progress_log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        elapsed = float(event.get("elapsed", 0.0))
        last_elapsed = max(last_elapsed, elapsed)
        kind = event.get("event")
        nodeid = event.get("nodeid")
        if kind == "start" and nodeid:
            active_nodeid = nodeid
            active_started = elapsed
        elif kind == "report" and nodeid:
            completed[nodeid] = completed.get(nodeid, 0.0) + float(event.get("duration", 0.0))
            if event.get("when") == "teardown" and active_nodeid == nodeid:
                active_nodeid = None
                active_started = None
    parts = [
        f"Progress log: {progress_log} ({len(completed)} tests reported; last event at {last_elapsed:.2f}s)",
    ]
    if active_nodeid is not None:
        running_for = max(0.0, last_elapsed - (active_started or 0.0))
        parts.append(
            f"Active test at timeout: {active_nodeid} (started at {active_started:.2f}s, "
            f"last event +{running_for:.2f}s)"
        )
    if completed:
        slowest = sorted(completed.items(), key=lambda item: item[1], reverse=True)[:10]
        formatted = "\n".join(f"  {dur:6.2f}s  {nid}" for nid, dur in slowest)
        parts.append(f"Slowest completed tests:\n{formatted}")
    if malformed:
        parts.append(f"(ignored {malformed} partial/malformed progress line(s))")
    return "\n".join(parts)


def run_nested_subprocess(command, *, cwd, env, timeout, selection, progress_log: Path):
    """Run a nested test subprocess, retaining output and timing diagnostics on timeout."""
    started = time.monotonic()
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout = _decode_stream(exc.stdout)
        stderr = _decode_stream(exc.stderr)
        progress = summarize_nested_progress(progress_log)
        raise AssertionError(
            f"Nested pytest subprocess timed out after {elapsed:.2f}s "
            f"(limit={timeout:.1f}s) for selection={selection!r}.\n"
            f"{progress}\n"
            f"--- child stdout (last 4000 chars) ---\n{stdout[-4000:]}\n"
            f"--- child stderr (last 4000 chars) ---\n{stderr[-4000:]}"
        ) from exc


@pytest.mark.parametrize("selection", [
    ["--collect-only"],
    ["-m", "isovar", "--strict-markers", "-k", ONE_CORPUS_CASE],
])
def test_collection_and_marker_selection_without_reference_data(tmp_path, selection):
    """Run collection and selected tests with an empty annotation cache."""
    if "--collect-only" not in selection:
        # Selected real-read tests need their published read bundle. Prepare it
        # before the child forbids networking, independent of other tests or a
        # restored CI cache. Collection alone must never need this prerequisite.
        import osteosarc
        from .sid_data import SHARED_BUNDLE
        osteosarc.fetch_bundle(SHARED_BUNDLE)
    cache = tmp_path / "ensembl"
    cache.mkdir()
    progress_log = tmp_path / "nested-progress.jsonl"
    code = textwrap.dedent("""
        import gzip
        import hashlib
        import json
        import os
        import sys
        import time
        from pathlib import Path

        def forbid_network(event, args):
            if event in {"socket.connect", "socket.getaddrinfo"}:
                raise AssertionError("Network access during collection/Isovar tests")

        sys.addaudithook(forbid_network)
        import pyensembl

        def forbid_annotation(*args, **kwargs):
            raise AssertionError("Unselected Ensembl annotation was accessed")

        # Selected real-read tests may index their checked-in, checksum-pinned
        # subset in a temporary directory. No reference access at all is
        # allowed during collection, and full-genome lookup/download stays
        # forbidden in both modes.
        references = {
            "GRCh38-osteosarc-six-transcript-subset":
                Path("tests/data/osteosarc/protein_reference/reference.gtf.gz").resolve(),
            "GRCh38-osteosarc-two-indel-subset":
                Path("tests/data/osteosarc_indels/protein_reference/reference.gtf.gz").resolve(),
            "GRCh38-osteosarc-all-" + hashlib.sha256(
                Path("tests/data/osteosarc_all_variants/reference/manifest.json").read_bytes()
            ).hexdigest()[:16]:
                Path("tests/data/osteosarc_all_variants/reference/reference.gtf.gz").resolve(),
        }
        original_db = pyensembl.Genome.db.fget
        original_index = pyensembl.Genome.index
        databases = {}
        # The continuity workflow actually builds a regional reference. Its
        # output is content-pinned too, though its path is a new tmp directory.
        # Pin uncompressed records: valid gzip headers/streams vary by platform.
        generated = json.loads(Path("tests/data/osteosarc_shared/continuity-reference.json").read_text())

        def require_pinned_reference(genome):
            assert "--collect-only" not in sys.argv, "Reference access during collection"
            if genome.reference_name in references:
                assert Path(genome.to_dict()["gtf_path_or_url"]).resolve() == references[genome.reference_name]
                return
            root = Path(genome.to_dict()["gtf_path_or_url"]).parent
            manifest_bytes = (root / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            assert genome.reference_name == "GRCh38-osteosarc-all-" + hashlib.sha256(manifest_bytes).hexdigest()[:16]
            assert {k: v for k, v in manifest.items() if k != "files"} == {
                k: v for k, v in generated.items() if k != "files"}
            assert manifest["files"].keys() == generated["files"].keys()
            for name, receipt in manifest["files"].items():
                packed = (root / name).read_bytes()
                assert receipt["sha256"] == hashlib.sha256(packed).hexdigest()
                assert receipt["source_url"] == generated["files"][name]["source_url"]
                assert hashlib.sha256(gzip.decompress(packed)).hexdigest() == generated["files"][name]["content_sha256"]

        def pinned_db(genome):
            require_pinned_reference(genome)
            database = original_db(genome)
            databases[id(database)] = database
            return database

        def pinned_index(genome, *args, **kwargs):
            require_pinned_reference(genome)
            return original_index(genome, *args, **kwargs)

        pyensembl.Genome.db = property(pinned_db)
        pyensembl.Genome.download = forbid_annotation
        pyensembl.Genome.index = pinned_index

        class ClosePinnedDatabases:
            def pytest_sessionfinish(self):
                # These references belong to this test process. Keep them
                # alive until they can be closed explicitly, before pytest's
                # final cyclic GC reports unclosed SQLite handles on 3.14.
                for database in databases.values():
                    database.close()
                databases.clear()

        class NestedProgressRecorder:
            def __init__(self, path):
                self._path = Path(path)
                self._started = time.monotonic()

            def _record(self, payload):
                payload["elapsed"] = round(time.monotonic() - self._started, 4)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload) + "\\n")
                    handle.flush()

            def pytest_runtest_logstart(self, nodeid, location):
                del location
                self._record({"event": "start", "nodeid": nodeid})

            def pytest_runtest_logreport(self, report):
                self._record({
                    "event": "report",
                    "nodeid": report.nodeid,
                    "when": report.when,
                    "outcome": report.outcome,
                    "duration": round(float(report.duration), 4),
                })

        import pytest
        plugins = [ClosePinnedDatabases()]
        progress_path = os.environ.get("TOPIARY_NESTED_PROGRESS_LOG")
        if progress_path:
            plugins.append(NestedProgressRecorder(progress_path))
        raise SystemExit(pytest.main(
            ["tests", "-q", *sys.argv[1:]], plugins=plugins))
    """)
    env = dict(
        os.environ,
        PYENSEMBL_CACHE_DIR=str(cache),
        TOPIARY_NESTED_PROGRESS_LOG=str(progress_log),
        **SINGLE_THREAD_ENV,
    )
    # The child selects tests explicitly; do not inherit xdist/coverage options
    # or pytest-cov's subprocess instrumentation from its parent test run.
    for name in list(env):
        if name in {"PYTEST_ADDOPTS", "COVERAGE_PROCESS_START"} or name.startswith("COV_CORE_"):
            env.pop(name)
    timeout = _nested_timeout_seconds(selection)
    result = run_nested_subprocess(
        [sys.executable, "-c", code, *selection],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        timeout=timeout,
        selection=selection,
        progress_log=progress_log,
    )
    assert result.returncode == 0, (
        f"{result.stdout}{result.stderr}\n{summarize_nested_progress(progress_log)}"
    )
    assert not list(cache.rglob("*")), "Collection/Isovar selection wrote reference data"
    expected = "collected" if "--collect-only" in selection else "deselected"
    assert expected in result.stdout


def test_nested_subprocess_timeout_retains_output_and_timing_diagnostics(tmp_path):
    """#568: timeout failures report child output, active test, and slowest timings."""
    progress_log = tmp_path / "progress.jsonl"
    child = textwrap.dedent("""
        import json, sys, time
        from pathlib import Path

        log = Path(sys.argv[1])
        events = [
            {"event": "start", "nodeid": "tests/test_fast.py::test_fast", "elapsed": 0.01},
            {"event": "report", "nodeid": "tests/test_fast.py::test_fast", "when": "call", "outcome": "passed", "duration": 0.02, "elapsed": 0.03},
            {"event": "report", "nodeid": "tests/test_fast.py::test_fast", "when": "teardown", "outcome": "passed", "duration": 0.001, "elapsed": 0.031},
            {"event": "start", "nodeid": "tests/test_slow.py::test_hang", "elapsed": 0.04},
        ]
        log.write_text("\\n".join(json.dumps(e) for e in events) + "\\n", encoding="utf-8")
        print("child-stdout-marker", flush=True)
        print("child-stderr-marker", file=sys.stderr, flush=True)
        time.sleep(5)
    """)
    with pytest.raises(AssertionError) as raised:
        run_nested_subprocess(
            [sys.executable, "-c", child, str(progress_log)],
            cwd=tmp_path,
            env=dict(os.environ, **SINGLE_THREAD_ENV),
            timeout=0.3,
            selection=["-m", "isovar"],
            progress_log=progress_log,
        )
    message = str(raised.value)
    assert "timed out" in message
    assert "selection=['-m', 'isovar']" in message
    assert "Active test at timeout: tests/test_slow.py::test_hang" in message
    assert "tests/test_fast.py::test_fast" in message
    assert "child-stdout-marker" in message
    assert "child-stderr-marker" in message


def test_data_path_import_does_not_import_annotation_libraries():
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            from pathlib import Path
            from tests.data import data_path
            assert "varcode" not in sys.modules
            assert "pyensembl" not in sys.modules
            assert Path(data_path("")) == Path("tests/data").resolve()
        """)],
        cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
