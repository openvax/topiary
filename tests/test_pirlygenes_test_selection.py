"""Regression tests for explicit PirlyGenes integration selection."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.optional_dependencies as optional_dependencies


def _raise(error):
    def raising_import(module_name):
        assert module_name == "pirlygenes"
        raise error

    return raising_import


def test_absent_pirlygenes_skips_in_base_environment(monkeypatch):
    error = ModuleNotFoundError(
        "No module named 'pirlygenes'", name="pirlygenes",
    )
    monkeypatch.delenv(optional_dependencies.PIRLYGENES_REQUIRED_ENV, False)
    monkeypatch.setattr(optional_dependencies, "import_module", _raise(error))

    with pytest.raises(pytest.skip.Exception):
        optional_dependencies.require_pirlygenes()


def test_absent_pirlygenes_fails_in_required_environment(monkeypatch):
    error = ModuleNotFoundError(
        "No module named 'pirlygenes'", name="pirlygenes",
    )
    monkeypatch.setenv(optional_dependencies.PIRLYGENES_REQUIRED_ENV, "1")
    monkeypatch.setattr(optional_dependencies, "import_module", _raise(error))

    with pytest.raises(pytest.fail.Exception, match="required"):
        optional_dependencies.require_pirlygenes()


def test_transitively_missing_dependency_is_not_skipped(monkeypatch):
    error = ModuleNotFoundError(
        "No module named 'broken_transitive_dependency'",
        name="broken_transitive_dependency",
    )
    monkeypatch.delenv(optional_dependencies.PIRLYGENES_REQUIRED_ENV, False)
    monkeypatch.setattr(optional_dependencies, "import_module", _raise(error))

    with pytest.raises(ModuleNotFoundError) as raised:
        optional_dependencies.require_pirlygenes()

    assert raised.value is error


def test_broken_import_is_not_skipped(monkeypatch):
    error = ImportError("cannot import name 'therapy_evidence' from 'oncoref'")
    monkeypatch.delenv(optional_dependencies.PIRLYGENES_REQUIRED_ENV, False)
    monkeypatch.setattr(optional_dependencies, "import_module", _raise(error))

    with pytest.raises(ImportError) as raised:
        optional_dependencies.require_pirlygenes()

    assert raised.value is error


def test_usable_pirlygenes_is_returned(monkeypatch):
    module = SimpleNamespace()
    monkeypatch.setattr(
        optional_dependencies, "import_module", lambda module_name: module,
    )

    assert optional_dependencies.require_pirlygenes() is module


def test_no_pirlygenes_importorskip_calls_remain():
    deprecated_call = "importorskip(" + '"pirlygenes")'
    for path in Path("tests").glob("test_*.py"):
        assert deprecated_call not in path.read_text()


def test_ci_requires_both_absent_and_installed_environments():
    workflow = Path(".github/workflows/tests.yml").read_text()

    assert "Verify PirlyGenes is absent from the base environment" in workflow
    assert "pirlygenes-integration:" in workflow
    assert "python -m pip install -e '.[pirlygenes]'" in workflow
    assert 'TOPIARY_TEST_REQUIRE_PIRLYGENES: "1"' in workflow
    assert "./test.sh -m pirlygenes --strict-markers" in workflow


def test_external_predictors_are_not_created_during_test_collection():
    """Marker selection must not require unrelated licensed executables."""
    external_predictors = {"NetMHC", "NetMHCpan", "NetMHCIIpan"}
    offenders = []

    for path in Path("tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for statement in tree.body:
            value = getattr(statement, "value", None)
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in external_predictors
            ):
                offenders.append(f"{path}:{statement.lineno}")

    assert not offenders, "external predictor constructed at import time: " + ", ".join(
        offenders,
    )
