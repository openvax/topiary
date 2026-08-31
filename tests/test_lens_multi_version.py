"""One LENS file, two versions of one tool (topiary #208).

Stripping the version from a binding column name is lossy when a table carries
`netmhcpan_4.1b.aff_nm` *and* `netmhcpan_4.2.aff_nm`: both became
`netmhcpan_affinity_value`, one set of values was dropped, and the only signal
was a pandas duplicate-column warning later — which doesn't name the predictor
that lost its values. `to_long()` then raised, so a consumer couldn't route
around it either.

Multi-version tables are a real input shape, not a constructed edge case.
"""

import pathlib
import warnings

import pandas as pd
import pytest

from topiary import Affinity, from_wide, read_lens, to_wide
from topiary.ranking import evaluate_scores

FIXTURE = pathlib.Path(__file__).parent / "data" / "lens" / "sample_v1_4.tsv"


def _two_version_file(tmp_path, source="netmhcpan_4.1b.aff_nm",
                      added="netmhcpan_4.2.aff_nm", offset=45.0):
    """The fixture plus a second version of one tool, with distinct values."""
    lines = FIXTURE.read_text().splitlines()
    header = lines[0].split("\t")
    index = header.index(source)
    header.append(added)
    rows = ["\t".join(header)]
    for line in lines[1:]:
        fields = line.split("\t")
        cell = fields[index].strip()
        # A blank or NA affinity is a legitimate cell; carry it through as
        # one rather than crashing the fixture builder on float("NA").
        if cell in ("", "NA", "nan"):
            fields.append(cell)
        else:
            fields.append(str(float(cell) + offset))
        rows.append("\t".join(fields))
    path = tmp_path / "two_versions.tsv"
    path.write_text("\n".join(rows) + "\n")
    return path


COLLISION_WARNING = r".*appears with versions"


def _drifted_version_file(tmp_path, column="netmhcpan_4.1b.score_ba",
                          drifted="netmhcpan_4.1.score_ba"):
    """The fixture with one column's version spelled differently.

    One run of one tool, inconsistently labelled across its metrics --
    which is drift, not a second version, since no two columns claim the
    same output name.
    """
    lines = FIXTURE.read_text().splitlines()
    header = lines[0].split("\t")
    header[header.index(column)] = drifted
    lines[0] = "\t".join(header)
    path = tmp_path / "drifted_version.tsv"
    path.write_text("\n".join(lines) + "\n")
    return path


def _read(path):
    """Read, muting only the collision warning these tests provoke."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", COLLISION_WARNING, UserWarning)
        return read_lens(path)


# ---------------------------------------------------------------------------
# Nothing is lost, and nothing collides
# ---------------------------------------------------------------------------


def test_columns_do_not_collide(tmp_path):
    result = _read(_two_version_file(tmp_path))

    assert result.df.columns.is_unique


def test_both_versions_get_their_own_column(tmp_path):
    result = _read(_two_version_file(tmp_path))

    assert "netmhcpan_4.1b_affinity_value" in result.df.columns
    assert "netmhcpan_4.2_affinity_value" in result.df.columns


def test_both_versions_keep_their_values(tmp_path):
    result = _read(_two_version_file(tmp_path))
    df = result.df

    # The second version was written as the first plus 45.
    assert (
        df["netmhcpan_4.2_affinity_value"]
        - df["netmhcpan_4.1b_affinity_value"]
    ).round(6).eq(45.0).all()


def test_metadata_keeps_its_documented_shape(tmp_path):
    """models stays method -> version; it cannot express two versions."""
    result = _read(_two_version_file(tmp_path))

    models = dict(result.metadata.models)
    assert "netmhcpan" in models
    assert models["netmhcpan"] in ("4.1b", "4.2")
    # Keys stay bare method names -- a caller reading models["netmhcpan"]
    # gets an answer for every file, multi-version or not.
    assert not any(key.startswith("netmhcpan_") for key in models)


def test_metadata_records_what_each_key_was_built_from(tmp_path):
    """The full truth lives in the explicit key map, not in models."""
    result = _read(_two_version_file(tmp_path))

    keys = result.metadata.extra["topiary_model_keys"]
    assert keys["netmhcpan_4.1b"] == ["netmhcpan", "4.1b"]
    assert keys["netmhcpan_4.2"] == ["netmhcpan", "4.2"]


def test_the_collision_is_announced(tmp_path):
    path = _two_version_file(tmp_path)

    with pytest.warns(UserWarning, match=r"netmhcpan appears with versions"):
        read_lens(path)


def test_a_single_version_file_is_unchanged(tmp_path):
    """No collision, no qualification, no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = read_lens(FIXTURE)

    assert "netmhcpan_affinity_value" in result.df.columns
    assert dict(result.metadata.models)["netmhcpan"] == "4.1b"


# ---------------------------------------------------------------------------
# The long form is usable, with the version addressable
# ---------------------------------------------------------------------------


def test_to_long_works(tmp_path):
    """It raised on a duplicate column, so there was no consumer escape."""
    result = _read(_two_version_file(tmp_path))

    long_result = result.to_long()

    assert len(long_result.df) > 0


def test_the_method_is_the_method_and_the_version_is_the_version(tmp_path):
    """Not a method named 'netmhcpan_4.1b' with no version."""
    long_df = _read(_two_version_file(tmp_path)).to_long().df
    netmhcpan = long_df[long_df["prediction_method_name"] == "netmhcpan"]

    assert sorted(netmhcpan["predictor_version"].dropna().unique()) == [
        "4.1b", "4.2",
    ]


def test_a_version_qualified_reference_resolves(tmp_path):
    """affinity['netmhcpan', '4.2'] is an advertised capability."""
    long_df = _read(_two_version_file(tmp_path)).to_long().df

    scores = evaluate_scores(long_df, Affinity["netmhcpan", "4.2"].value)

    assert scores.notna().any()


def test_the_two_versions_score_differently(tmp_path):
    long_df = _read(_two_version_file(tmp_path)).to_long().df

    first = evaluate_scores(long_df, Affinity["netmhcpan", "4.1b"].value)
    second = evaluate_scores(long_df, Affinity["netmhcpan", "4.2"].value)

    assert not first.equals(second)


# ---------------------------------------------------------------------------
# Qualifying is for real collisions only
# ---------------------------------------------------------------------------


def test_version_drift_across_metrics_does_not_split_the_predictor(tmp_path):
    """One run, two spellings of its version, still one predictor.

    A file that writes `netmhcpan_4.1b.aff_nm` beside
    `netmhcpan_4.1.score_ba` is inconsistent, not multi-version: no two
    columns claim the same output name. Qualifying on "two version
    strings appeared for this tool" would split the affinity axis in
    half and undo what #206 fixed.
    """
    path = _drifted_version_file(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = read_lens(path)

    assert "netmhcpan_affinity_value" in result.df.columns
    assert "netmhcpan_affinity_score" in result.df.columns

    long_df = result.to_long().df
    netmhcpan = long_df[long_df["prediction_method_name"] == "netmhcpan"]
    affinity = netmhcpan[netmhcpan["kind"] == "pMHC_affinity"]
    assert affinity["predictor_version"].nunique(dropna=False) == 1
    # Value and score reached the same rows rather than separate ones.
    assert affinity["value"].notna().any()
    assert affinity["score"].notna().any()
    assert (affinity["value"].notna() & affinity["score"].notna()).any()


# ---------------------------------------------------------------------------
# The same round-trip gap, independent of LENS
# ---------------------------------------------------------------------------


def test_to_wide_from_wide_recovers_method_and_version():
    """to_wide qualifies the key on collision; from_wide must reverse it."""
    long_df = pd.DataFrame([
        dict(source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
             allele="HLA-A*02:01", kind="pMHC_affinity", value=value,
             score=0.5, percentile_rank=1.0,
             prediction_method_name="netmhcpan", predictor_version=version)
        for value, version in ((75.0, "4.1b"), (120.0, "4.2"))
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        restored = from_wide(to_wide(long_df))

    pairs = sorted(zip(restored["prediction_method_name"],
                       restored["predictor_version"]))
    assert pairs == [("netmhcpan", "4.1b"), ("netmhcpan", "4.2")]


def test_a_method_whose_name_ends_in_a_version_survives():
    """'netmhcpan_4.1b' as a method name is not an encoded key.

    Guessing at the encoding by stripping a trailing '_{version}' cannot
    tell the two apart and renames this method to 'netmhcpan'.
    """
    long_df = pd.DataFrame([
        dict(source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
             allele="HLA-A*02:01", kind="pMHC_affinity", value=75.0,
             score=0.5, percentile_rank=1.0,
             prediction_method_name="netmhcpan_4.1b",
             predictor_version="4.1b"),
    ])

    restored = from_wide(to_wide(long_df))

    assert restored["prediction_method_name"].tolist() == ["netmhcpan_4.1b"]
    assert restored["predictor_version"].tolist() == ["4.1b"]


def test_a_single_version_round_trip_is_unchanged():
    long_df = pd.DataFrame([
        dict(source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
             allele="HLA-A*02:01", kind="pMHC_affinity", value=75.0,
             score=0.5, percentile_rank=1.0,
             prediction_method_name="netmhcpan", predictor_version="4.1b"),
    ])

    restored = from_wide(to_wide(long_df))

    assert restored["prediction_method_name"].tolist() == ["netmhcpan"]
    assert restored["predictor_version"].tolist() == ["4.1b"]
