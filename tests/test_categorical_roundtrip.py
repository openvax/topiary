"""Categorical API predicates retain their values through both DSL renderers."""

import numpy as np
import pandas as pd
import pytest

from topiary import Column, EvalContext, parse
from .test_twin_conformance import DSL_RENDER_TWINS


def frame(values, column="label"):
    return pd.DataFrame({
        "source_sequence_name": "synthetic",
        "peptide_offset": list(range(len(values))),
        "peptide": ["A" * (i + 1) for i in range(len(values))],
        "allele": "HLA-A*02:01",
        column: pd.Series(values, dtype=object),
    })


def assert_same(node, df, preserve_unknown=False):
    context = EvalContext(df, preserve_unknown=preserve_unknown)
    expected = node.eval(context)
    for render in DSL_RENDER_TWINS:
        text = render(node)
        restored = parse(text)
        pd.testing.assert_series_equal(restored.eval(context), expected, check_exact=True)
        # Parsing can make grouping explicit; subsequent replay must be stable.
        canonical = render(restored)
        assert render(parse(canonical)) == canonical


@pytest.mark.parametrize("value", [
    "yes", "", "None", "True", "1", "café β", "both ' and \" quotes",
    "slash\\quote'\"", "line\nbreak\ttab\x00", r"literal\n\t",
    True, False, 0, -7, 2**53 + 1, 0.25, -1.25e-20, None,
    float("nan"), float("inf"), float("-inf"),
    np.int64(2**53 + 1), np.float64(0.25), np.bool_(False), np.str_("quoted '\""),
])
@pytest.mark.parametrize("method", ["eq", "ne"])
@pytest.mark.parametrize("preserve_unknown", [False, True])
def test_scalar_categories_roundtrip(value, method, preserve_unknown):
    df = frame([value, "unmatched", None])
    assert_same(getattr(Column("label"), method)(value), df, preserve_unknown)


@pytest.mark.parametrize("values", [[], ["I", "II"], [None, float("nan")],
                                    [True, -3, 2.5, "2.5", "'\\\"", None]])
@pytest.mark.parametrize("negate", [False, True])
@pytest.mark.parametrize("preserve_unknown", [False, True])
def test_membership_categories_roundtrip(values, negate, preserve_unknown):
    df = frame(["I", "II", "other", True, -3, 2.5, "2.5", "'\\\"", None])
    node = Column("label").isin(values)
    assert_same(~node if negate else node, df, preserve_unknown)
    assert_same(node, df.iloc[:0], preserve_unknown)


@pytest.mark.parametrize("node", [
    Column("label").eq("yes"), Column("label").ne("no"),
    Column("label").isin(["yes", "maybe"]), ~Column("label").isin(["no", "never"]),
    Column("label").isin([]), ~Column("label").isin([]),
])
def test_categorical_composition_preserves_masks_and_scores(node):
    df = frame(["yes", "no", "maybe", None]).assign(amount=[4., 3., 2., 1.])
    for expression in (node, ~node, node & (Column("amount") > 2),
                       node | Column("label").eq("no"),
                       node * Column("amount") + 1, (node * 3 >= 2),
                       node.clip(0, 1)):
        assert_same(expression, df)


def test_categorical_numbers_keep_integer_precision_and_raw_types():
    df = frame([2**53, 2**53 + 1, str(2**53 + 1), "not a number"])
    node = Column("label").eq(2**53 + 1)
    assert node.eval(EvalContext(df)).tolist() == [False, True, False, False]
    assert_same(node, df)


@pytest.mark.parametrize("dtype, values, target", [
    ("string", ["yes", "no", None], "yes"),
    ("boolean", [True, False, None], True),
    ("Int64", [2**53 + 1, -2, None], 2**53 + 1),
    ("Float64", [0.25, -2., None], 0.25),
])
def test_nullable_column_dtypes_preserve_unknown_membership(dtype, values, target):
    df = frame(values)
    df["label"] = df.label.astype(dtype)
    node = Column("label").eq(target)
    assert node.eval(EvalContext(df)).tolist() == [True, False, False]
    for preserve_unknown in (False, True):
        assert_same(node, df, preserve_unknown)
        assert_same(~node, df, preserve_unknown)


@pytest.mark.parametrize("name", ["review label", "a'b\"c", "x) | y", "", "é", "e\u0301"])
def test_quoted_column_names_share_one_rendering(name):
    df = frame(["yes", "no"], column=name)
    for node in (Column(name).eq("yes"), ~Column(name).isin(["no", "never"]),
                 Column(name).includes("yes")):
        assert_same(node, df)
    assert_same(Column(name), frame([1., 2.], column=name))


@pytest.mark.parametrize("value", ["both ' and \" quotes", "line\nbreak", r"literal\n", "β\\"])
def test_string_equality_and_membership_decode_the_same_escaped_literal(value):
    df = frame([value, "other"])
    expected = Column("label").eq(value).eval(EvalContext(df))
    for source in (f"column(label).eq({value!r})", f"label == {value!r}"):
        pd.testing.assert_series_equal(parse(source).eval(EvalContext(df)), expected)
    assert_same(Column("label").includes(value), df)


@pytest.mark.parametrize("source", [
    "label.eq(other)", "label.eq(column(other))", "label.eq(1 + 2)",
    "label.eq()", "label.ne('x', 'y')", "label.eq(['x'])",
    "label.isin('x')", "label.isin(other)", "label.isin([['x']])",
    "label.isin(['x', other])", "label.isin(['x'], ['y'])",
    "affinity.eq('x')", "(column(label) + 1).isin(['x'])",
    "label.eq('unterminated)", "label.eq('trailing\\')", r"label.eq('\xZZ')",
])
def test_categorical_syntax_rejects_expressions_and_malformed_literals(source):
    with pytest.raises(ValueError):
        parse(source)


def test_categorical_keywords_are_literals_only_inside_arguments():
    for spelling in ("none", "NONE", "None"):
        assert parse(f"label.eq({spelling})").values == (None,)
    for spelling in ("true", "True", "TRUE"):
        assert parse(f"label.eq({spelling})").values == (True,)
    assert parse("column(True)").col_name == "True"
    assert parse("None").col_name == "None"
    assert parse("label.isin([1, -2.5, False,])").values == (1, -2.5, False)


def test_preexisting_negated_string_policy_keeps_its_saved_expansion():
    from topiary import SelectionPolicy

    policy = SelectionPolicy("existing", "1", filter_by='~(label == "no")')
    definition = policy.to_dict()
    # This form was emitted before categorical method calls could be parsed.
    # Changing it would reject existing policies at expanded-definition checks.
    assert definition["expanded"]["filter_by"]["expression"] == "~ ( label == 'no' )"
    assert SelectionPolicy.from_dict(definition).sha256 == policy.sha256
