"""The exact-release lookup must distinguish absence from uncertainty."""

from http.client import IncompleteRead
from io import BytesIO
import json
from urllib.error import HTTPError, URLError

import pytest

from topiary import pypi_release_exists
from topiary import release


def response(payload, status=200):
    result = BytesIO(json.dumps(payload).encode())
    result.status = status
    return result


@pytest.mark.parametrize("version, reported", [
    ("5.53", "5.53.0"), ("5.53.0.0", "5.53.0"),
    ("v5.53.0", "5.53.0"), ("5.53.0rc1", "5.53.0rc1"),
    ("5.53.0.post1", "5.53.0.post1"), ("5.53.0.dev1", "5.53.0.dev1"),
])
def test_release_exists_uses_standard_versions_and_names(monkeypatch, version, reported):
    requested = []

    def get(request, timeout):
        requested.append(request)
        assert timeout == 3
        return response({"info": {"name": "Example.Project", "version": reported,
                                   "yanked": True, "requires_python": ">=100"}, "urls": []})

    monkeypatch.setattr(release, "urlopen", get)
    monkeypatch.setenv("PIP_INDEX_URL", "https://wrong-index.example/simple")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://another-index.example/simple")
    assert pypi_release_exists("Example_Project", version, timeout=3) is True
    assert requested[0].full_url.startswith("https://pypi.org/pypi/example-project/")
    assert requested[0].get_header("Accept") == "application/json"
    assert requested[0].get_header("Cache-control") == "no-cache"


@pytest.mark.parametrize("project, version", [
    ("", "1.0"), ("../topiary", "1.0"), ("topiary@elsewhere", "1.0"),
    ("topiary", ""), ("topiary", "not-a-version"),
])
def test_invalid_identifiers_fail_before_request(monkeypatch, project, version):
    def unexpected_request(*args, **kwargs):
        pytest.fail("Invalid input reached the network")

    monkeypatch.setattr(release, "urlopen", unexpected_request)
    with pytest.raises(ValueError):
        pypi_release_exists(project, version)


@pytest.mark.parametrize("error", [
    URLError("DNS failure"), TimeoutError("timed out"),
    ConnectionResetError("connection reset"), IncompleteRead(b"{", 20),
    HTTPError("https://pypi.org", 503, "Service Unavailable", {}, None),
])
def test_lookup_failure_preserves_diagnostic_and_cause(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(release, "urlopen", fail)
    with pytest.raises(RuntimeError, match="Could not verify topiary 5.53.0 on PyPI") as raised:
        pypi_release_exists("topiary", "5.53.0")
    assert raised.value.__cause__ is error
    assert str(error) in str(raised.value)


def test_only_404_means_absent(monkeypatch):
    def absent(request, **kwargs):
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(release, "urlopen", absent)
    assert pypi_release_exists("topiary", "5.53.0") is False


@pytest.mark.parametrize("status", [201, 202, 204, 206, 304])
def test_nonstandard_success_status_is_not_release_evidence(monkeypatch, status):
    monkeypatch.setattr(release, "urlopen", lambda *a, **kw: response({}, status=status))
    with pytest.raises(RuntimeError, match=f"unexpected HTTP status {status}"):
        pypi_release_exists("topiary", "5.53.0")


def test_invalid_encoding_is_not_absence(monkeypatch):
    result = BytesIO(b"\xff")
    result.status = 200
    monkeypatch.setattr(release, "urlopen", lambda *a, **kw: result)
    with pytest.raises(RuntimeError, match="Could not verify"):
        pypi_release_exists("topiary", "5.53.0")
