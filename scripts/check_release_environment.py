"""Fail-closed release environment preflight: python scripts/check_release_environment.py.

Validates before expensive lint and test gates that:

1. No stale build-generated ``topiary.egg-info`` directory in the repository
   root shadows a refreshed editable ``.dist-info`` installation (nor are
   multiple ``topiary`` distributions visible on ``sys.path``).
2. The selected ``importlib.metadata`` distribution version matches
   ``topiary/__init__.py`` in the current checkout.
3. The selected distribution's complete runtime dependency contract matches
   ``requirements.txt`` in the current checkout.
4. Importing ``topiary`` resolves to the current editable source checkout.

Release tooling, not library API (#565): invoked by ``deploy.sh`` before
``./lint.sh`` and ``./test.sh``.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


PROJECT_NAME = "topiary"


@dataclass(frozen=True)
class DistributionInfo:
    """Resolved ``importlib.metadata`` distribution visible to the interpreter."""

    name: str
    version: str
    path: Path
    is_egg_info: bool
    requirements: tuple[Requirement, ...]


@dataclass(frozen=True)
class ReleaseEnvironmentReport:
    """Verified release-environment metadata summary."""

    repo_root: Path
    source_version: str
    selected: DistributionInfo
    all_distributions: tuple[DistributionInfo, ...]
    expected_requirements: tuple[Requirement, ...]


def read_source_version(repo_root: Path | str) -> str:
    """Read ``__version__`` from ``topiary/__init__.py`` without importing it.

    Parameters
    ----------
    repo_root : path-like
        Root directory of the Topiary source checkout.

    Returns
    -------
    str
        The literal ``__version__`` string declared in ``topiary/__init__.py``.
    """
    init_path = Path(repo_root) / "topiary" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        version_text = node.value.value.strip()
                        if version_text:
                            Version(version_text)
                            return version_text
    raise ValueError(f"Could not find string literal __version__ in {init_path}")


def parse_requirements_file(path: Path | str) -> tuple[Requirement, ...]:
    """Parse runtime PEP 508 requirements from a ``requirements.txt`` file.

    Parameters
    ----------
    path : path-like
        Path to ``requirements.txt``.

    Returns
    -------
    tuple of packaging.requirements.Requirement
        Non-empty, non-comment requirement lines in file order.
    """
    requirements: list[Requirement] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(Requirement(line))
    return tuple(requirements)


def _runtime_requirements(raw_requires: Sequence[str] | None) -> tuple[Requirement, ...]:
    """Keep non-extra runtime requirements from ``Distribution.requires``."""
    result: list[Requirement] = []
    for text in raw_requires or ():
        req = Requirement(text)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        result.append(req)
    return tuple(result)


def _canonical_requirement_key(req: Requirement) -> tuple[str, str, str]:
    """Canonical comparison tuple ``(name, specifier, marker)``."""
    return (
        canonicalize_name(req.name),
        str(req.specifier),
        "" if req.marker is None else str(req.marker),
    )


def find_project_distributions(
    project: str = PROJECT_NAME,
    *,
    search_paths: Sequence[str] | None = None,
) -> list[DistributionInfo]:
    """Return every visible distribution matching *project* in resolution order.

    Parameters
    ----------
    project : str, optional
        Distribution name to search for; defaults to ``"topiary"``.
    search_paths : sequence of str, optional
        Paths searched by ``importlib.metadata.distributions``; defaults to
        ``sys.path``.

    Returns
    -------
    list of DistributionInfo
        Matching distributions in the order Python's metadata finder sees them
        (the first entry is the one ``importlib.metadata.distribution`` selects),
        deduplicated by resolved metadata path.
    """
    canonical = canonicalize_name(project)
    kwargs = {} if search_paths is None else {"path": list(search_paths)}
    found: list[DistributionInfo] = []
    seen_paths: set[Path] = set()
    for dist in importlib_metadata.distributions(**kwargs):
        name = dist.metadata.get("Name")
        if not name or canonicalize_name(name) != canonical:
            continue
        raw_path = getattr(dist, "_path", None)
        dist_path = Path(raw_path).resolve() if raw_path is not None else Path("<unknown>")
        if dist_path in seen_paths:
            continue
        seen_paths.add(dist_path)
        found.append(
            DistributionInfo(
                name=name,
                version=str(dist.version),
                path=dist_path,
                is_egg_info=dist_path.name.endswith(".egg-info"),
                requirements=_runtime_requirements(dist.requires),
            )
        )
    return found


def check_release_environment(
    repo_root: Path | str | None = None,
    *,
    search_paths: Sequence[str] | None = None,
    imported_file: Path | str | None = None,
) -> ReleaseEnvironmentReport:
    """Validate that installed editable metadata matches the source checkout.

    Detects stale build-generated ``topiary.egg-info`` directories shadowing
    refreshed PEP 660 ``.dist-info`` metadata, verifies that only one
    ``topiary`` distribution is visible, verifies the selected version against
    ``topiary/__init__.py``, verifies the complete runtime dependency contract
    against ``requirements.txt``, and (when *imported_file* is provided)
    verifies that imports resolve inside *repo_root*.

    Parameters
    ----------
    repo_root : path-like, optional
        Topiary repository checkout root. Defaults to the parent of ``scripts/``.
    search_paths : sequence of str, optional
        Explicit ``sys.path`` entries for metadata discovery. Defaults to
        ``sys.path``.
    imported_file : path-like, optional
        Resolved ``topiary.__file__`` path from the active interpreter. When
        supplied, verifies that it lies inside ``repo_root / "topiary"``.

    Returns
    -------
    ReleaseEnvironmentReport
        Summary of the verified source checkout and selected distribution.

    Raises
    ------
    RuntimeError
        If stale ``topiary.egg-info`` or duplicate distributions exist, if no
        distribution is installed, if the selected version or any runtime
        requirement disagrees with source, or if *imported_file* points outside
        the editable checkout.
    """
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    source_version = read_source_version(root)
    expected_reqs = parse_requirements_file(root / "requirements.txt")
    distributions = find_project_distributions(PROJECT_NAME, search_paths=search_paths)
    local_egg_info = (root / f"{PROJECT_NAME}.egg-info").resolve()

    if not distributions:
        raise RuntimeError(
            f"No installed {PROJECT_NAME!r} distribution metadata found on sys.path. "
            'Refresh the editable install with: "$PYTHON" -m pip install -e . --no-deps --no-build-isolation'
        )

    selected = distributions[0]
    summary = ", ".join(f"{d.path} (version {d.version})" for d in distributions)

    if len(distributions) > 1:
        remediation = (
            f"Remove stale build metadata ({local_egg_info}) and refresh the editable install: "
            'rm -rf topiary.egg-info && "$PYTHON" -m pip install -e . --no-deps --no-build-isolation'
            if any(d.is_egg_info for d in distributions) or local_egg_info.exists()
            else 'Reinstall cleanly into a single environment: "$PYTHON" -m pip install -e . --no-deps --no-build-isolation'
        )
        raise RuntimeError(
            f"Multiple {PROJECT_NAME!r} distributions are visible on sys.path; "
            f"selected {selected.path} ({selected.version}) shadows later metadata "
            f"[{summary}]. {remediation}"
        )

    if local_egg_info.exists():
        raise RuntimeError(
            f"Stale build-generated {local_egg_info} is present in the checkout "
            f"(selected metadata: {selected.path}, version {selected.version}). "
            "Ignored egg-info directories can shadow refreshed PEP 660 editable metadata. "
            'Remove it and refresh the editable install: rm -rf topiary.egg-info && '
            '"$PYTHON" -m pip install -e . --no-deps --no-build-isolation'
        )

    if imported_file is not None:
        package_dir = (root / PROJECT_NAME).resolve()
        resolved_import = Path(imported_file).resolve()
        try:
            resolved_import.relative_to(package_dir)
        except ValueError as error:
            raise RuntimeError(
                f"Imported {PROJECT_NAME!r} from {resolved_import}, outside the "
                f"current checkout {package_dir}. Refresh the editable install: "
                '"$PYTHON" -m pip install -e . --no-deps --no-build-isolation'
            ) from error

    if Version(selected.version) != Version(source_version):
        raise RuntimeError(
            f"Selected {PROJECT_NAME!r} metadata at {selected.path} declares version "
            f"{selected.version!r}, but source checkout {root} declares "
            f"{source_version!r}. Refresh editable metadata with: "
            '"$PYTHON" -m pip install -e . --no-deps --no-build-isolation'
        )

    expected_by_name = {
        canonicalize_name(req.name): _canonical_requirement_key(req)
        for req in expected_reqs
    }
    actual_by_name = {
        canonicalize_name(req.name): _canonical_requirement_key(req)
        for req in selected.requirements
    }

    missing = sorted(set(expected_by_name) - set(actual_by_name))
    extra = sorted(set(actual_by_name) - set(expected_by_name))
    mismatched = sorted(
        name
        for name in set(expected_by_name) & set(actual_by_name)
        if expected_by_name[name] != actual_by_name[name]
    )
    if missing or extra or mismatched:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        if mismatched:
            diffs = [
                f"{name}: metadata has {actual_by_name[name][1]!r} vs "
                f"requirements.txt {expected_by_name[name][1]!r}"
                for name in mismatched
            ]
            details.append("mismatched=[" + "; ".join(diffs) + "]")
        raise RuntimeError(
            f"Selected {PROJECT_NAME!r} metadata at {selected.path} disagrees with "
            f"{root / 'requirements.txt'} ({', '.join(details)}). "
            'Remove any stale topiary.egg-info and refresh editable metadata: '
            'rm -rf topiary.egg-info && "$PYTHON" -m pip install -e . --no-deps --no-build-isolation'
        )

    return ReleaseEnvironmentReport(
        repo_root=root,
        source_version=source_version,
        selected=selected,
        all_distributions=tuple(distributions),
        expected_requirements=expected_reqs,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 only when selected editable metadata matches the source checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to the Topiary repository root (defaults to parent of scripts/).",
    )
    args = parser.parse_args(argv)
    try:
        import topiary

        imported_file = getattr(topiary, "__file__", None)
        report = check_release_environment(
            repo_root=args.repo_root,
            imported_file=imported_file,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"Verified release environment: {report.selected.name} {report.selected.version} "
        f"at {report.selected.path} ({len(report.expected_requirements)} runtime requirements match source)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
