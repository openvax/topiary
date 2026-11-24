"""Unit tests for release-environment metadata preflight (#565)."""

from pathlib import Path
import pytest

from scripts.check_release_environment import (
    check_release_environment,
    find_project_distributions,
    main,
    parse_requirements_file,
    read_source_version,
)


def _write_repo(tmp_path: Path, *, version: str, requirements: list[str]) -> Path:
    repo = tmp_path / "repo"
    (repo / "topiary").mkdir(parents=True)
    (repo / "topiary" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (repo / "requirements.txt").write_text(
        "# Core runtime requirements\n" + "\n".join(requirements) + "\n"
    )
    return repo


def _write_dist_info(
    site_packages: Path,
    *,
    version: str,
    requirements: list[str],
    extra_requirements: list[str] = (),
) -> Path:
    dist_dir = site_packages / f"topiary-{version}.dist-info"
    dist_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Metadata-Version: 2.1",
        "Name: topiary",
        f"Version: {version}",
        *[f"Requires-Dist: {req}" for req in requirements],
        *[f"Requires-Dist: {req}" for req in extra_requirements],
    ]
    (dist_dir / "METADATA").write_text("\n".join(lines) + "\n")
    return dist_dir


def _write_egg_info(
    root: Path, *, version: str, requirements: list[str],
) -> Path:
    egg_dir = root / "topiary.egg-info"
    egg_dir.mkdir(parents=True, exist_ok=True)
    (egg_dir / "PKG-INFO").write_text(
        f"Metadata-Version: 2.1\nName: topiary\nVersion: {version}\n"
    )
    (egg_dir / "requires.txt").write_text("\n".join(requirements) + "\n")
    return egg_dir


def test_clean_editable_dist_info_passes_preflight(tmp_path):
    reqs = ["isovar>=1.78.0,<2", "mhctools>=4.5.0", "packaging>=23.2"]
    repo = _write_repo(tmp_path, version="5.116.0", requirements=reqs)
    site_packages = tmp_path / "site-packages"
    dist_dir = _write_dist_info(
        site_packages,
        version="5.116.0",
        requirements=reqs,
        extra_requirements=['pytest>=8; extra == "dev"'],
    )

    report = check_release_environment(
        repo,
        search_paths=[str(repo), str(site_packages)],
        imported_file=repo / "topiary" / "__init__.py",
    )

    assert report.source_version == "5.116.0"
    assert report.selected.version == "5.116.0"
    assert report.selected.path == dist_dir.resolve()
    assert not report.selected.is_egg_info
    assert len(report.all_distributions) == 1
    assert len(report.expected_requirements) == 3


def test_detects_stale_local_egg_info_shadowing_refreshed_dist_info(tmp_path):
    """Exact #565 reproduction: ignored 5.108.1 egg-info shadows 5.108.2 dist-info."""
    repo = _write_repo(
        tmp_path,
        version="5.108.2",
        requirements=["isovar>=1.54.3,<2", "mhctools>=4.5.0"],
    )
    site_packages = tmp_path / "site-packages"
    egg_dir = _write_egg_info(
        repo,
        version="5.108.1",
        requirements=["isovar>=1.39.5,<2", "mhctools>=4.5.0"],
    )
    dist_dir = _write_dist_info(
        site_packages,
        version="5.108.2",
        requirements=["isovar>=1.54.3,<2", "mhctools>=4.5.0"],
    )

    found = find_project_distributions(search_paths=[str(repo), str(site_packages)])
    assert [d.path for d in found] == [egg_dir.resolve(), dist_dir.resolve()]
    assert found[0].is_egg_info and found[0].version == "5.108.1"
    assert not found[1].is_egg_info and found[1].version == "5.108.2"

    with pytest.raises(RuntimeError) as raised:
        check_release_environment(
            repo,
            search_paths=[str(repo), str(site_packages)],
            imported_file=repo / "topiary" / "__init__.py",
        )
    message = str(raised.value)
    assert "Multiple 'topiary' distributions" in message
    assert str(egg_dir.resolve()) in message
    assert str(dist_dir.resolve()) in message
    assert "topiary.egg-info" in message


def test_detects_leftover_local_egg_info_even_when_repo_not_on_search_path(tmp_path):
    reqs = ["isovar>=1.78.0,<2"]
    repo = _write_repo(tmp_path, version="5.116.0", requirements=reqs)
    site_packages = tmp_path / "site-packages"
    _write_egg_info(repo, version="5.116.0", requirements=reqs)
    _write_dist_info(site_packages, version="5.116.0", requirements=reqs)

    with pytest.raises(RuntimeError, match="Stale build-generated.*topiary.egg-info"):
        check_release_environment(
            repo,
            search_paths=[str(site_packages)],
            imported_file=repo / "topiary" / "__init__.py",
        )


def test_detects_version_mismatch(tmp_path):
    reqs = ["isovar>=1.78.0,<2"]
    repo = _write_repo(tmp_path, version="5.116.0", requirements=reqs)
    site_packages = tmp_path / "site-packages"
    _write_dist_info(site_packages, version="5.115.0", requirements=reqs)

    with pytest.raises(RuntimeError, match="declares version '5.115.0'.*'5.116.0'"):
        check_release_environment(
            repo,
            search_paths=[str(repo), str(site_packages)],
            imported_file=repo / "topiary" / "__init__.py",
        )


def test_detects_dependency_contract_mismatch(tmp_path):
    repo = _write_repo(
        tmp_path,
        version="5.116.0",
        requirements=["isovar>=1.78.0,<2", "mhctools>=4.5.0", "packaging>=23.2"],
    )
    site_packages = tmp_path / "site-packages"
    _write_dist_info(
        site_packages,
        version="5.116.0",
        requirements=["isovar>=1.39.5,<2", "mhctools>=4.5.0", "obsolete>=1.0"],
    )

    with pytest.raises(RuntimeError) as raised:
        check_release_environment(
            repo,
            search_paths=[str(repo), str(site_packages)],
            imported_file=repo / "topiary" / "__init__.py",
        )
    message = str(raised.value)
    assert "missing=['packaging']" in message
    assert "extra=['obsolete']" in message
    assert "isovar" in message


def test_detects_non_editable_import_outside_checkout(tmp_path):
    reqs = ["isovar>=1.78.0,<2"]
    repo = _write_repo(tmp_path, version="5.116.0", requirements=reqs)
    site_packages = tmp_path / "site-packages"
    _write_dist_info(site_packages, version="5.116.0", requirements=reqs)
    other_install = site_packages / "topiary" / "__init__.py"
    other_install.parent.mkdir(parents=True)
    other_install.write_text('__version__ = "5.116.0"\n')

    with pytest.raises(RuntimeError, match="outside the current checkout"):
        check_release_environment(
            repo,
            search_paths=[str(site_packages)],
            imported_file=other_install,
        )


def test_detects_missing_installed_metadata(tmp_path):
    repo = _write_repo(tmp_path, version="5.116.0", requirements=["isovar>=1.78.0,<2"])
    empty_site = tmp_path / "empty-site"
    empty_site.mkdir()

    with pytest.raises(RuntimeError, match="No installed 'topiary' distribution metadata"):
        check_release_environment(repo, search_paths=[str(empty_site)])


def test_helpers_and_cli_exit_status(tmp_path, monkeypatch, capsys):
    reqs = ["isovar>=1.78.0,<2"]
    repo = _write_repo(tmp_path, version="5.116.0", requirements=reqs)
    assert read_source_version(repo) == "5.116.0"
    assert [r.name for r in parse_requirements_file(repo / "requirements.txt")] == ["isovar"]

    # CLI returns 1 on check failure with clear stderr diagnostic.
    monkeypatch.setattr(
        "scripts.check_release_environment.check_release_environment",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stale egg-info found")),
    )
    assert main(["--repo-root", str(repo)]) == 1
    assert "ERROR: stale egg-info found" in capsys.readouterr().err
