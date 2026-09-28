"""The same selection rules must apply to both optional integrations."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.optional_dependencies as optional_dependencies


@pytest.fixture(params=["isovar", "pirlygenes"])
def dependency(request):
    return request.param


def _raise(error, dependency):
    def raising_import(module_name):
        assert module_name == dependency
        raise error

    return raising_import


def test_absent_integration_skips_in_base_environment(monkeypatch, dependency):
    error = ModuleNotFoundError(
        f"No module named '{dependency}'", name=dependency,
    )
    monkeypatch.delenv(f"TOPIARY_TEST_REQUIRE_{dependency.upper()}", False)
    monkeypatch.setattr(optional_dependencies, "import_module", _raise(error, dependency))

    with pytest.raises(pytest.skip.Exception):
        optional_dependencies.require_integration(dependency)


def test_absent_integration_fails_in_required_environment(monkeypatch, dependency):
    error = ModuleNotFoundError(
        f"No module named '{dependency}'", name=dependency,
    )
    monkeypatch.setenv(f"TOPIARY_TEST_REQUIRE_{dependency.upper()}", "1")
    monkeypatch.setattr(optional_dependencies, "import_module", _raise(error, dependency))

    with pytest.raises(pytest.fail.Exception, match="required"):
        optional_dependencies.require_integration(dependency)


@pytest.mark.parametrize("required", ["0", "1"])
def test_transitively_missing_dependency_is_not_skipped(monkeypatch, dependency, required):
    error = ModuleNotFoundError(
        "No module named 'broken_transitive_dependency'",
        name="broken_transitive_dependency",
    )
    monkeypatch.setenv(f"TOPIARY_TEST_REQUIRE_{dependency.upper()}", required)
    monkeypatch.setattr(optional_dependencies, "import_module", _raise(error, dependency))

    with pytest.raises(ModuleNotFoundError) as raised:
        optional_dependencies.require_integration(dependency)

    assert raised.value is error


@pytest.mark.parametrize("required", ["0", "1"])
@pytest.mark.parametrize("error_type", [ImportError, RuntimeError])
def test_broken_import_is_not_skipped(monkeypatch, dependency, required, error_type):
    error = error_type("broken optional installation")
    monkeypatch.setenv(f"TOPIARY_TEST_REQUIRE_{dependency.upper()}", required)
    monkeypatch.setattr(optional_dependencies, "import_module", _raise(error, dependency))

    with pytest.raises(error_type) as raised:
        optional_dependencies.require_integration(dependency)

    assert raised.value is error


def test_usable_integration_is_returned(monkeypatch, dependency):
    module = SimpleNamespace()
    monkeypatch.setattr(
        optional_dependencies, "import_module", lambda module_name: module,
    )

    assert optional_dependencies.require_integration(dependency) is module


def test_no_optional_importorskip_calls_remain(dependency):
    deprecated_call = "importorskip(" + f'"{dependency}")'
    for path in Path("tests").glob("test_*.py"):
        assert deprecated_call not in path.read_text()


def test_ci_requires_both_absent_and_installed_environments(dependency):
    """CI tests each optional integration both without it and with it.

    Read from the workflow's structure, not its text (#316): rewording a
    step or routing an install through scripts/pip_install.sh leaves this
    passing, while dropping either environment fails it.
    """
    import yaml

    jobs = yaml.safe_load(Path(".github/workflows/tests.yml").read_text())["jobs"]
    display_name = {"isovar": "Isovar", "pirlygenes": "PirlyGenes"}[dependency]
    selection = "'isovar or osteosarc'" if dependency == "isovar" else "pirlygenes"

    def commands(job):
        return "\n".join(step.get("run", "") for step in job.get("steps", []))

    # The base build proves the dependency is absent there...
    assert f"Verify {display_name} is absent from the base environment" in [
        step.get("name") for step in jobs["build"]["steps"]]
    # ...and a dedicated job installs the extra, requires it, and runs its
    # marked tests.
    integration = jobs[f"{dependency}-integration"]
    assert f"-e '.[{dependency}]'" in commands(integration)
    assert integration["env"][f"TOPIARY_TEST_REQUIRE_{dependency.upper()}"] == "1"
    assert f"./test.sh -m {selection} --strict-markers" in commands(integration)


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
