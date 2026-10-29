"""Test collection must not require licensed predictor executables."""

import ast
from pathlib import Path


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
