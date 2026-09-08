"""Exercise CI downloads, complete coverage aggregation, and upload retries."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from threading import Thread
import xml.etree.ElementTree as ET

from coverage import CoverageData
import pytest


@pytest.fixture
def download_server():
    """Serve controlled HTTP failures to the real curl used by the script."""
    responses = []
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            status, body, length = responses[min(len(requests) - 1, len(responses) - 1)]
            self.send_response(status)
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/reporter", responses, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("failure", ["none", "transient", "http", "not-found", "partial", "checksum"])
def test_verified_download_retries_and_fails_closed(tmp_path, download_server, failure):
    url, responses, requests = download_server
    body = b"verified reporter bytes\n"
    success = (200, body, len(body))
    responses.extend({
        "none": [success],
        "transient": [(503, b"unavailable", 11), success],
        "http": [(503, b"unavailable", 11)],
        "not-found": [(404, b"missing", 7)],
        "partial": [(200, body[:4], len(body))],
        "checksum": [(200, b"wrong binary", 12)],
    }[failure])
    destination = tmp_path / "reporter with spaces"
    destination.write_bytes(b"previous verified installation")
    script = Path(__file__).resolve().parents[1] / "scripts/download_verified.sh"
    result = subprocess.run(
        ["bash", str(script), url, hashlib.sha256(body).hexdigest(), str(destination)],
        capture_output=True, text=True, timeout=20,
    )
    if failure in {"none", "transient"}:
        assert result.returncode == 0, result.stderr
        assert destination.read_bytes() == body
        assert len(requests) == (2 if failure == "transient" else 1)
    else:
        assert result.returncode != 0
        assert destination.read_bytes() == b"previous verified installation"
        assert len(requests) == (1 if failure == "checksum" else 4)
        if failure in {"http", "not-found"}:
            assert str(responses[0][0]) in result.stderr
        if failure == "checksum":
            assert "FAILED" in result.stdout
    assert not list(tmp_path.glob("*.download.*"))


@pytest.fixture
def coverage_matrix(tmp_path):
    source = tmp_path / "example.py"
    source.write_text("first = 1\nsecond = 2\nthird = 3\n")
    directory = tmp_path / "coverage-data"
    paths = []
    for line, version in enumerate(["3.10", "3.11", "3.12"], 1):
        path = directory / f"coverage-python-{version}" / ".coverage"
        path.parent.mkdir(parents=True)
        data = CoverageData(basename=str(path))
        data.add_lines({str(source): {line}})
        data.write()
        paths.append(path)
    return directory, source, paths


def _combine(directory, output):
    script = Path(__file__).with_name("combine_coverage.py")
    return subprocess.run(
        [sys.executable, str(script), str(directory), "3", str(output)],
        capture_output=True, text=True, timeout=30,
    )


def test_combine_covers_union_and_reuses_successful_jobs_on_retry(tmp_path, coverage_matrix):
    directory, source, paths = coverage_matrix
    output = tmp_path / "coverage.xml"
    result = _combine(directory, output)
    assert result.returncode == 0, result.stderr
    assert ET.parse(output).getroot().attrib["line-rate"] == "1"
    assert all(path.is_file() for path in paths)

    # Only the retried member changes. Other matrix artifacts are reused,
    # and the old combined result must not leak into the next upload.
    retried = CoverageData(basename=str(paths[-1]))
    retried.erase()
    retried.add_lines({str(source): {2}})
    retried.write()
    result = _combine(directory, output)
    assert result.returncode == 0, result.stderr
    lines = ET.parse(output).findall(".//line")
    assert {int(line.attrib["number"]): int(line.attrib["hits"]) for line in lines} == {
        1: 1, 2: 1, 3: 0,
    }


@pytest.mark.parametrize("failure", ["missing", "extra", "empty", "corrupt", "incompatible"])
def test_combine_refuses_incomplete_or_invalid_matrix(tmp_path, coverage_matrix, failure):
    directory, source, paths = coverage_matrix
    if failure == "missing":
        paths[-1].unlink()
    elif failure == "extra":
        extra = directory / "coverage-python-unexpected" / ".coverage"
        extra.parent.mkdir()
        extra.write_bytes(paths[0].read_bytes())
    elif failure in {"empty", "corrupt"}:
        paths[-1].write_bytes(b"" if failure == "empty" else b"not a coverage database")
    else:
        data = CoverageData(basename=str(paths[-1]))
        data.erase()
        data.add_arcs({str(source): {(-1, 1), (1, -1)}})
        data.write()
    output = tmp_path / "coverage.xml"
    result = _combine(directory, output)
    assert result.returncode != 0
    assert not output.exists()
    assert paths[-1].name in result.stderr or "coverage" in result.stderr.lower()


def test_workflow_separates_coverage_and_preserves_matrix_artifacts():
    workflow = Path(".github/workflows/tests.yml").read_text()
    build = workflow.split("  build:\n", 1)[1].split("  pirlygenes-integration:\n", 1)[0]
    upload = workflow.split("  coverage:\n", 1)[1]
    assert "coverallsapp/github-action" not in workflow
    assert "parallel-finished" not in workflow
    assert "coveralls" not in build
    assert "coverage-jobs: ${{ strategy.job-total }}" in build
    assert "include-hidden-files: true" in build
    assert "overwrite: true" in build
    assert "if-no-files-found: error" in build
    assert "needs: build" in upload
    assert "always()" not in upload
    assert "continue-on-error" not in upload
    assert "needs.build.outputs.coverage-jobs" in upload
    assert "tests/combine_coverage.py" in upload
    assert "scripts/download_verified.sh" in upload


@pytest.mark.parametrize("status", [0, 42])
def test_actual_upload_step_has_retry_identity_and_propagates_errors(tmp_path, status):
    workflow = Path(".github/workflows/tests.yml").read_text()
    step = workflow.split("      - name: Publish coverage to Coveralls\n", 1)[1]
    command = textwrap.dedent(step.split("        run: |\n", 1)[1])
    reporter = tmp_path / "coveralls"
    reporter.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\nexit "$REPORT_STATUS"\n')
    reporter.chmod(0o755)
    for attempt in (1, 2):
        result = subprocess.run(
            ["bash", "-eo", "pipefail", "-c", command],
            env=dict(os.environ, RUNNER_TEMP=str(tmp_path), GITHUB_RUN_ID="12345",
                     GITHUB_RUN_ATTEMPT=str(attempt), REPORT_STATUS=str(status)),
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == status
        assert result.stdout.splitlines() == [
            "report", "coverage.xml", "--format=cobertura",
            f"--build-number=12345-{attempt}",
        ]
