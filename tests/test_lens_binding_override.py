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
import tempfile
import warnings

import pytest

from topiary import read_lens

FIXTURE = pathlib.Path(__file__).parent / "data" / "lens" / "sample_v1_4.tsv"


def _file_with(column, value="0.42", tmp_path=None):
    """The fixture plus one extra predictor-shaped column."""
    lines = FIXTURE.read_text().splitlines()
    header = lines[0].split("\t") + [column]
    rows = ["\t".join(header)]
    for line in lines[1:]:
        rows.append("\t".join(line.split("\t") + [value]))
    base = pathlib.Path(tmp_path or tempfile.mkdtemp())
    path = base / "extra_column.tsv"
    path.write_text("\n".join(rows) + "\n")
    return path


UNMAPPED = "netmhcpan_4.1b.el_score"


# ---------------------------------------------------------------------------
# Without an override: the gap is reported, and that is all topiary can do
# ---------------------------------------------------------------------------


def test_an_unmapped_column_warns(tmp_path):
    with pytest.warns(UserWarning, match="look like"):
        read_lens(_file_with(UNMAPPED, tmp_path=tmp_path))


# ---------------------------------------------------------------------------
# With one: the caller closes the gap locally
# ---------------------------------------------------------------------------


def test_an_override_maps_the_column(tmp_path):
    result = read_lens(
        _file_with(UNMAPPED, tmp_path=tmp_path),
        binding_metrics={("netmhcpan", "el_score"): ("immunogenicity", "score")},
    )

    assert "netmhcpan_immunogenicity_score" in result.df.columns


def test_an_override_silences_the_warning(tmp_path):
    path = _file_with(UNMAPPED, tmp_path=tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        read_lens(
            path,
            binding_metrics={
                ("netmhcpan", "el_score"): ("immunogenicity", "score"),
            },
        )


def test_the_mapped_column_reaches_the_long_form(tmp_path):
    """Landing in the wide frame is not enough — it has to survive to_long."""
    result = read_lens(
        _file_with(UNMAPPED, tmp_path=tmp_path),
        binding_metrics={("netmhcpan", "el_score"): ("immunogenicity", "score")},
    )

    long_df = result.to_long().df
    rows = long_df[long_df["kind"] == "immunogenicity"]

    assert len(rows) > 0
    assert rows["score"].dropna().eq(0.42).all()


def test_overrides_merge_over_the_builtin_table(tmp_path):
    """Patch one column without restating the other thirteen."""
    result = read_lens(
        _file_with(UNMAPPED, tmp_path=tmp_path),
        binding_metrics={("netmhcpan", "el_score"): ("immunogenicity", "score")},
    )

    assert "netmhcpan_affinity_value" in result.df.columns
    assert "mhcflurry_presentation_score" in result.df.columns


def test_an_override_can_correct_a_builtin_mapping(tmp_path):
    """Mis-mapped, not just unmapped — the issue names both."""
    result = read_lens(
        FIXTURE,
        binding_metrics={("netmhcpan", "aff_nm"): ("affinity", "score")},
    )

    assert "netmhcpan_affinity_score" in result.df.columns
    assert "netmhcpan_affinity_value" not in result.df.columns


def test_the_key_is_version_free(tmp_path):
    """One mapping covers a tool however the file spells its version."""
    for version in ("4.1b", "4.2", "9.9"):
        result = read_lens(
            _file_with(f"netmhcpan_{version}.el_score", tmp_path=tmp_path),
            binding_metrics={
                ("netmhcpan", "el_score"): ("immunogenicity", "score"),
            },
        )
        assert "netmhcpan_immunogenicity_score" in result.df.columns


# ---------------------------------------------------------------------------
# None: "not a prediction" — acknowledged, not deleted
# ---------------------------------------------------------------------------


def test_none_silences_the_warning(tmp_path):
    path = _file_with(UNMAPPED, tmp_path=tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        read_lens(path, binding_metrics={("netmhcpan", "el_score"): None})


def test_none_leaves_the_column_alone(tmp_path):
    """Overriding a mapping must not delete the caller's data."""
    result = read_lens(
        _file_with(UNMAPPED, tmp_path=tmp_path),
        binding_metrics={("netmhcpan", "el_score"): None},
    )

    assert UNMAPPED in result.df.columns
    assert "netmhcpan_immunogenicity_score" not in result.df.columns


# ---------------------------------------------------------------------------
# Validation, because a bad mapping fails silently downstream
# ---------------------------------------------------------------------------


def test_an_unknown_kind_is_refused(tmp_path):
    """It would emit a name to_long cannot read, losing the data quietly."""
    with pytest.raises(ValueError, match="unknown kind"):
        read_lens(
            FIXTURE,
            binding_metrics={("netmhcpan", "el_score"): ("bogus", "score")},
        )


def test_an_unknown_field_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown field"):
        read_lens(
            FIXTURE,
            binding_metrics={("netmhcpan", "el_score"): ("affinity", "nope")},
        )


def test_a_non_pair_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"\(tool, metric\) string"):
        read_lens(
            FIXTURE,
            binding_metrics={"netmhcpan.el_score": ("affinity", "value")},
        )


def test_a_non_pair_value_is_refused(tmp_path):
    with pytest.raises(ValueError, match="must be a .kind, field. pair"):
        read_lens(
            FIXTURE,
            binding_metrics={("netmhcpan", "el_score"): "affinity"},
        )


def test_a_non_mapping_is_refused(tmp_path):
    with pytest.raises(TypeError, match="must be a mapping"):
        read_lens(FIXTURE, binding_metrics=[("netmhcpan", "el_score")])


def test_keys_are_case_and_space_insensitive(tmp_path):
    """The warning prints lowercase; a caller copying it out should work."""
    result = read_lens(
        _file_with(UNMAPPED, tmp_path=tmp_path),
        binding_metrics={
            (" NetMHCpan ", " EL_Score "): ("immunogenicity", "score"),
        },
    )

    assert "netmhcpan_immunogenicity_score" in result.df.columns


def test_no_override_is_unchanged():
    """The default path must be exactly what it was."""
    with_none = read_lens(FIXTURE, binding_metrics=None)
    without = read_lens(FIXTURE)

    assert list(with_none.df.columns) == list(without.df.columns)
