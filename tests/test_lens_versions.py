"""read_lens must not hinge on a predictor's version spelling (topiary #206).

A LENS binding column is named `<tool>_<version>.<metric>`. The mapping table
was keyed on the whole name, so a version topiary hadn't seen passed through
unmapped — and a consumer reading normalized names cannot tell "this tool
emitted nothing" from "this tool emitted something I don't recognize". The
predictor's entire axis just wasn't there.

`aff_nm` is an IC50 whichever NetMHCpan produced it, so the table is keyed on
(tool, metric) and the version is recorded instead.
"""

import pathlib
import warnings

import pytest

from topiary import read_lens
from topiary.io_lens import detect_lens_version

FIXTURE = pathlib.Path(__file__).parent / "data" / "lens" / "sample_v1_4.tsv"

BINDING_COLUMNS = (
    "netmhcpan_presentation_score", "netmhcpan_presentation_rank",
    "netmhcpan_affinity_score", "netmhcpan_affinity_rank",
    "netmhcpan_affinity_value", "mhcflurry_affinity_value",
    "mhcflurry_affinity_rank", "mhcflurry_antigen_processing_score",
    "mhcflurry_presentation_score", "mhcflurry_presentation_rank",
    "netmhcstabpan_stability_score", "netmhcstabpan_stability_value",
    "netmhcstabpan_stability_rank",
)


def _respun(tmp_path, *substitutions):
    """The real fixture with its version segments rewritten."""
    text = FIXTURE.read_text()
    for old, new in substitutions:
        text = text.replace(old, new)
    path = tmp_path / "lens.tsv"
    path.write_text(text)
    return path


def _frame(result):
    return result.df if hasattr(result, "df") else result


@pytest.mark.parametrize("substitution", [
    (),
    (("netmhcpan_4.1b.", "netmhcpan_4.1."),),
    (("netmhcpan_4.1b.", "netmhcpan_4.2."),),
    (("mhcflurry_2.1.1.", "mhcflurry_3.0."),),
    (("netmhcstabpan_1.0.", "netmhcstabpan_1.1."),),
    (("netmhcpan_4.1b.", "netmhcpan_5."),
     ("mhcflurry_2.1.1.", "mhcflurry_2.2.0."),),
])
def test_every_binding_column_maps_whatever_the_version(tmp_path, substitution):
    df = _frame(read_lens(_respun(tmp_path, *substitution)))

    missing = [c for c in BINDING_COLUMNS if c not in df.columns]
    assert not missing


def test_the_version_is_recorded_rather_than_matched(tmp_path):
    result = read_lens(_respun(tmp_path, ("netmhcpan_4.1b.", "netmhcpan_4.2.")))

    assert result.metadata.models["netmhcpan"] == "4.2"


def test_values_are_unchanged_by_the_version_spelling(tmp_path):
    """Same file, different version segment: identical numbers."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()

    original = _frame(read_lens(_respun(first)))
    respun = _frame(read_lens(
        _respun(second, ("netmhcpan_4.1b.", "netmhcpan_4.9.")),
    ))

    for column in BINDING_COLUMNS:
        assert original[column].equals(respun[column]), column


# ---------------------------------------------------------------------------
# What topiary doesn't know, it says
# ---------------------------------------------------------------------------


def test_an_unknown_tool_is_reported_not_dropped(tmp_path):
    path = _respun(
        tmp_path,
        ("netmhcpan_4.1b.aff_nm", "brandnewtool_1.0.opaque_metric"),
    )

    with pytest.warns(UserWarning, match="brandnewtool_1.0.opaque_metric"):
        df = _frame(read_lens(path))

    # Still in the frame under its own name — reported, not discarded.
    assert "brandnewtool_1.0.opaque_metric" in df.columns


def test_an_unknown_metric_is_reported(tmp_path):
    path = _respun(tmp_path, ("netmhcpan_4.1b.aff_nm", "netmhcpan_4.1b.brand_new"))

    with pytest.warns(UserWarning, match="netmhcpan_4.1b.brand_new"):
        read_lens(path)


def test_a_known_file_says_nothing(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        read_lens(_respun(tmp_path))


def test_non_predictor_columns_are_not_reported(tmp_path):
    """Ordinary annotation columns aren't predictor-shaped."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read_lens(_respun(tmp_path))

    assert not [w for w in caught if "predictor output" in str(w.message)]


# ---------------------------------------------------------------------------
# Version detection has the same brittleness, and the same fix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["1.0", "1.1", "2.0"])
def test_v1_4_is_detected_whatever_netmhcstabpan_version(version):
    columns = [f"netmhcstabpan_{version}.stab_pred_score", "peptide", "allele"]

    assert detect_lens_version(columns) == "v1.4"


def test_later_versions_still_detect_by_their_own_markers():
    assert detect_lens_version(["lohhla_allele_loss_pval"]) == "v1.9"
    assert detect_lens_version(["snaf_exp"]) == "v1.5.1"


def test_an_unrecognizable_file_is_still_none():
    assert detect_lens_version(["peptide", "allele"]) is None
