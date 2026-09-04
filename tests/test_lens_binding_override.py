"""Caller overrides for read_lens's binding-column map (topiary #211).

`_BINDING_METRICS` is private and `read_lens` took no mapping argument, so a
consumer hitting an unmapped or mis-mapped binding column had no supported way
to correct it locally — the only route was a topiary release. That is why a
mapping gap (#208) blocked a downstream consumer outright rather than being a
local workaround.

Keys are `(tool, metric)`, deliberately without the version: a mapping keyed
on the raw column name would stop working the moment a file spelled the
version differently, which is the brittleness #206 removed.
"""

import pathlib
import warnings

import pytest

from topiary import read_lens

FIXTURE = pathlib.Path(__file__).parent / "data" / "lens" / "sample_v1_4.tsv"


def _file_with(column, tmp_path, value="0.42"):
    """The fixture plus one extra predictor-shaped column."""
    lines = FIXTURE.read_text().splitlines()
    header = lines[0].split("\t") + [column]
    rows = ["\t".join(header)]
    for line in lines[1:]:
        rows.append("\t".join(line.split("\t") + [value]))
    path = tmp_path / f"{column.replace('.', '_')}.tsv"
    path.write_text("\n".join(rows) + "\n")
    return path


UNMAPPED = "netmhcpan_4.1b.opaque_metric"


# ---------------------------------------------------------------------------
# Without an override: the gap is reported, and that is all topiary can do
# ---------------------------------------------------------------------------


def test_an_unmapped_column_warns(tmp_path):
    with pytest.warns(UserWarning, match="look like"):
        read_lens(_file_with(UNMAPPED, tmp_path))


@pytest.mark.parametrize(
    "column, expected",
    [
        ("calis_EL_1.0.MT_Score", "calis_presentation_score"),
        ("calis_BA_1.0.MT_Score", "calis_affinity_score"),
        (
            "futuremodel_1.0.immunogenicity_MT_percentile",
            "futuremodel_immunogenicity_rank",
        ),
    ],
)
def test_shared_prediction_vocabulary_maps_new_lens_columns(
    column, expected, tmp_path,
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = read_lens(_file_with(column, tmp_path))

    assert expected in result.df.columns
    assert result.df[expected].eq(0.42).all()


def test_underscore_digit_tool_name_is_not_split_as_version(tmp_path):
    column = "foo_2_1.0.MT_Presentation_Score"

    result = read_lens(_file_with(column, tmp_path))

    assert "foo_2_presentation_score" in result.df.columns
    assert result.df["foo_2_presentation_score"].eq(0.42).all()
    assert result.models["foo_2"] == "1.0"


def test_underscore_digit_tool_name_matches_override_key(tmp_path):
    column = "foo_2_1.0.opaque_metric"

    result = read_lens(
        _file_with(column, tmp_path),
        binding_metrics={("foo_2", "opaque_metric"): ("affinity", "score")},
    )

    assert "foo_2_affinity_score" in result.df.columns
    assert result.df["foo_2_affinity_score"].eq(0.42).all()


def test_lens_preserves_wt_predictions_it_cannot_represent(tmp_path):
    """A WT value must not be silently relabeled as a mutant value."""
    column = "calis_EL_1.0.WT_Score"

    with pytest.warns(UserWarning, match="look like"):
        result = read_lens(_file_with(column, tmp_path))

    assert column in result.df.columns
    assert "calis_presentation_score" not in result.df.columns


# ---------------------------------------------------------------------------
# With one: the caller closes the gap locally
# ---------------------------------------------------------------------------


def test_an_override_maps_the_column(tmp_path):
    result = read_lens(
        _file_with(UNMAPPED, tmp_path),
        binding_metrics={("netmhcpan", "opaque_metric"): ("immunogenicity", "score")},
    )

    assert "netmhcpan_immunogenicity_score" in result.df.columns


def test_an_override_silences_the_warning(tmp_path):
    path = _file_with(UNMAPPED, tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        read_lens(
            path,
            binding_metrics={
                ("netmhcpan", "opaque_metric"): ("immunogenicity", "score"),
            },
        )


def test_the_mapped_column_reaches_the_long_form(tmp_path):
    """Landing in the wide frame is not enough — it has to survive to_long."""
    result = read_lens(
        _file_with(UNMAPPED, tmp_path),
        binding_metrics={("netmhcpan", "opaque_metric"): ("immunogenicity", "score")},
    )

    long_df = result.to_long().df
    rows = long_df[long_df["kind"] == "immunogenicity"]

    assert len(rows) > 0
    assert rows["score"].dropna().eq(0.42).all()


def test_overrides_merge_over_the_builtin_table(tmp_path):
    """Patch one column without restating the other thirteen."""
    result = read_lens(
        _file_with(UNMAPPED, tmp_path),
        binding_metrics={("netmhcpan", "opaque_metric"): ("immunogenicity", "score")},
    )

    assert "netmhcpan_affinity_value" in result.df.columns
    assert "mhcflurry_presentation_score" in result.df.columns


def test_an_override_can_correct_a_builtin_mapping():
    """Mis-mapped, not just unmapped — the issue names both.

    Redirected to a field of a *different* kind, since two metrics of one
    tool cannot share a (kind, field) — see the collision tests below.
    """
    result = read_lens(
        FIXTURE,
        binding_metrics={("netmhcpan", "aff_nm"): ("immunogenicity", "value")},
    )

    assert "netmhcpan_immunogenicity_value" in result.df.columns
    assert "netmhcpan_affinity_value" not in result.df.columns
    # Column presence is not enough: a frame can hold the name and still
    # be unusable. The corrected column has to survive to long form.
    long_df = result.to_long().df
    assert result.df.columns.is_unique
    assert (long_df["kind"] == "immunogenicity").any()


# ---------------------------------------------------------------------------
# An override must not be able to build a frame that cannot be read back
# ---------------------------------------------------------------------------


def test_two_metrics_claiming_one_output_are_refused():
    """#208's shape, and an override is the one way left to cause it.

    ``score_ba`` already maps to (affinity, score) for netmhcpan, so
    redirecting ``aff_nm`` there too puts two columns of the same name in
    the frame: pandas allows it, one set of values becomes unreachable,
    and ``to_long()`` raises "Expected a 1D array".
    """
    with pytest.raises(ValueError, match="more than one column to the same"):
        read_lens(
            FIXTURE,
            binding_metrics={("netmhcpan", "aff_nm"): ("affinity", "score")},
        )


def test_the_collision_error_names_both_columns():
    """So the caller can see which two, not just that there were two."""
    with pytest.raises(ValueError) as excinfo:
        read_lens(
            FIXTURE,
            binding_metrics={("netmhcpan", "aff_nm"): ("affinity", "score")},
        )

    message = str(excinfo.value)
    assert "netmhcpan_4.1b.aff_nm" in message
    assert "netmhcpan_4.1b.score_ba" in message
    assert "netmhcpan_affinity_score" in message


def test_two_overrides_colliding_with_each_other_are_refused(tmp_path):
    """Not only override-versus-builtin."""
    path = _file_with(UNMAPPED, tmp_path)

    with pytest.raises(ValueError, match="more than one column to the same"):
        read_lens(path, binding_metrics={
            ("netmhcpan", "opaque_metric"): ("immunogenicity", "score"),
            ("netmhcpan", "score_el"): ("immunogenicity", "score"),
        })


def test_a_collision_can_be_resolved_by_declaring_one_a_non_prediction():
    """The escape the error message points at."""
    result = read_lens(FIXTURE, binding_metrics={
        ("netmhcpan", "aff_nm"): ("affinity", "score"),
        ("netmhcpan", "score_ba"): None,
    })

    assert result.df.columns.is_unique
    assert "netmhcpan_affinity_score" in result.df.columns


def test_the_key_is_version_free(tmp_path):
    """One mapping covers a tool however the file spells its version."""
    for version in ("4.1b", "4.2", "9.9"):
        result = read_lens(
            _file_with(f"netmhcpan_{version}.opaque_metric", tmp_path),
            binding_metrics={
                ("netmhcpan", "opaque_metric"): ("immunogenicity", "score"),
            },
        )
        assert "netmhcpan_immunogenicity_score" in result.df.columns


# ---------------------------------------------------------------------------
# None: "not a prediction" — acknowledged, not deleted
# ---------------------------------------------------------------------------


def test_none_silences_the_warning(tmp_path):
    path = _file_with(UNMAPPED, tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        read_lens(path, binding_metrics={("netmhcpan", "opaque_metric"): None})


def test_none_leaves_the_column_alone(tmp_path):
    """Overriding a mapping must not delete the caller's data."""
    result = read_lens(
        _file_with(UNMAPPED, tmp_path),
        binding_metrics={("netmhcpan", "opaque_metric"): None},
    )

    assert UNMAPPED in result.df.columns
    assert "netmhcpan_immunogenicity_score" not in result.df.columns


# ---------------------------------------------------------------------------
# Validation, because a bad mapping fails silently downstream
# ---------------------------------------------------------------------------


def test_an_unknown_kind_is_refused():
    """It would emit a name to_long cannot read, losing the data quietly."""
    with pytest.raises(ValueError, match="unknown kind"):
        read_lens(
            FIXTURE,
            binding_metrics={("netmhcpan", "opaque_metric"): ("bogus", "score")},
        )


def test_an_unknown_field_is_refused():
    with pytest.raises(ValueError, match="unknown field"):
        read_lens(
            FIXTURE,
            binding_metrics={("netmhcpan", "opaque_metric"): ("affinity", "nope")},
        )


def test_a_non_pair_key_is_refused():
    with pytest.raises(ValueError, match=r"\(tool, metric\) string"):
        read_lens(
            FIXTURE,
            binding_metrics={"netmhcpan.opaque_metric": ("affinity", "value")},
        )


def test_a_non_pair_value_is_refused():
    with pytest.raises(ValueError, match="must be a .kind, field. pair"):
        read_lens(
            FIXTURE,
            binding_metrics={("netmhcpan", "opaque_metric"): "affinity"},
        )


def test_a_non_mapping_is_refused():
    with pytest.raises(TypeError, match="must be a mapping"):
        read_lens(FIXTURE, binding_metrics=[("netmhcpan", "opaque_metric")])


def test_keys_are_case_and_space_insensitive(tmp_path):
    """The warning prints lowercase; a caller copying it out should work."""
    result = read_lens(
        _file_with(UNMAPPED, tmp_path),
        binding_metrics={
            (" NetMHCpan ", " Opaque_Metric "): (
                "immunogenicity", "score",
            ),
        },
    )

    assert "netmhcpan_immunogenicity_score" in result.df.columns


def test_no_override_is_unchanged():
    """The default path must be exactly what it was."""
    with_none = read_lens(FIXTURE, binding_metrics=None)
    without = read_lens(FIXTURE)

    assert list(with_none.df.columns) == list(without.df.columns)


def test_an_unhashable_field_is_refused():
    """A list field must raise ValueError, not TypeError from a set lookup."""
    with pytest.raises(ValueError, match="unknown field"):
        read_lens(
            FIXTURE,
            binding_metrics={("netmhcpan", "opaque_metric"): ("affinity", ["value"])},
        )


def test_values_are_case_and_space_insensitive_like_keys(tmp_path):
    """The key half normalizes; the value half has no reason not to."""
    result = read_lens(
        _file_with(UNMAPPED, tmp_path),
        binding_metrics={
            ("netmhcpan", "opaque_metric"): (" Immunogenicity ", " Score "),
        },
    )

    assert "netmhcpan_immunogenicity_score" in result.df.columns


def test_a_key_no_column_could_match_is_refused():
    """A tool that does not start with a letter is a silent no-op."""
    with pytest.raises(ValueError, match="no LENS column can match"):
        read_lens(
            FIXTURE,
            binding_metrics={("_netmhc", "opaque_metric"): ("affinity", "value")},
        )


def test_an_empty_metric_is_refused():
    with pytest.raises(ValueError, match="empty metric"):
        read_lens(
            FIXTURE, binding_metrics={("netmhcpan", "  "): ("affinity", "value")},
        )


def test_two_keys_meaning_the_same_pair_are_refused():
    """They normalize together, so one would be discarded in silence."""
    with pytest.raises(ValueError, match="two keys that mean the same"):
        read_lens(FIXTURE, binding_metrics={
            ("NetMHCpan", "aff_nm"): ("affinity", "score"),
            ("netmhcpan", "aff_nm"): ("presentation", "rank"),
        })


def test_the_warning_names_the_override_key(tmp_path):
    """The design claim: what topiary tells you is what you pass back."""
    with pytest.warns(UserWarning, match=r"\('netmhcpan', 'opaque_metric'\)"):
        read_lens(_file_with(UNMAPPED, tmp_path))


def test_overrides_are_recorded_as_provenance(tmp_path):
    """A frame built with a corrected map is distinguishable from one without."""
    result = read_lens(
        _file_with(UNMAPPED, tmp_path),
        binding_metrics={("netmhcpan", "opaque_metric"): ("immunogenicity", "score")},
    )

    assert result.metadata.extra["lens_binding_metrics"] == {
        "netmhcpan.opaque_metric": ["immunogenicity", "score"],
    }


def test_no_override_records_no_provenance():
    assert "lens_binding_metrics" not in read_lens(FIXTURE).metadata.extra


def test_validation_happens_before_the_file_is_read():
    """A malformed override should not cost a parse of a large TSV first."""
    with pytest.raises(ValueError, match="unknown kind"):
        read_lens(
            "/nonexistent/path.tsv",
            binding_metrics={("netmhcpan", "opaque_metric"): ("bogus", "score")},
        )
