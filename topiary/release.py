"""Read-only release checks against the PyPI upload destination."""

import json
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from packaging.utils import canonicalize_name
from packaging.version import Version


def pypi_release_exists(project, version, *, timeout=10):
    """Check whether an exact package version already exists on PyPI.

    Parameters
    ----------
    project : str
        Valid distribution name. Case, hyphens, underscores and dots are
        normalized using standard packaging rules.
    version : str
        PEP 440 version. Equivalent spellings, such as ``1.0`` and ``1.0.0``,
        identify the same release; substrings and prereleases do not.
    timeout : float, optional
        Timeout in seconds for the HTTPS request; defaults to 10.

    Returns
    -------
    bool
        True for matching release metadata, including yanked releases or
        releases with no remaining files. False only when PyPI returns HTTP
        404 (the project or version does not exist).

    Raises
    ------
    ValueError
        The project name or version is invalid, including empty input.
    RuntimeError
        PyPI cannot be reached, returns another HTTP error, or returns
        malformed/mismatched metadata. An unverifiable release is never
        treated as an available version.

    Notes
    -----
    Queries PyPI's release-specific JSON API, not pip's configured index or
    interpreter-filtered list of installable releases. This is a read-only
    preflight, not a reservation: the upload must still reject duplicates.
    """
    project = canonicalize_name(project, validate=True)
    version = Version(version)
    url = f"https://pypi.org/pypi/{quote(project, safe='')}/{quote(str(version), safe='')}/json"
    request = Request(url, headers={"Accept": "application/json", "Cache-Control": "no-cache"})
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ValueError(f"unexpected HTTP status {response.status}")
            payload = json.load(response)
        if not isinstance(payload, dict) or not isinstance(payload.get("info"), dict):
            raise ValueError("response has no release metadata")
        info = payload["info"]
        if not isinstance(info.get("name"), str) or canonicalize_name(info["name"]) != project:
            raise ValueError("response names a different project")
        if not isinstance(info.get("version"), str) or Version(info["version"]) != version:
            raise ValueError("response names a different version")
        return True
    except HTTPError as error:
        error.close()
        if error.code == 404:
            return False
        raise RuntimeError(f"Could not verify {project} {version} on PyPI: {error}") from error
    except (OSError, HTTPException, ValueError) as error:
        raise RuntimeError(f"Could not verify {project} {version} on PyPI: {error}") from error
