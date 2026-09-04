"""Shared prediction-column vocabulary used by report readers."""

import pytest

from topiary import parse_prediction_metric


# Base names from mhctools.cli.args.mhc_predictors. Mode-specific entries such
# as netmhcpan41-el and bigmhc-im reduce to these bases before a suffix is
# applied. Keeping the list here makes a dropped integration name visible.
MHCTOOLS_MODEL_BASES = (
    "bigmhc",
    "calis",
    "deepimmuno",
    "deeptap",
    "eramer",
    "mhcflurry",
    "mixmhc2pred",
    "mixmhcpred",
    "netchop",
    "netcleave",
    "netcleave-i",
    "netcleave-ii",
    "netmhc",
    "netmhc3",
    "netmhc4",
    "netmhccons",
    "netmhccons-iedb",
    "netmhciipan",
    "netmhciipan-iedb",
    "netmhciipan3",
    "netmhciipan4",
    "netmhciipan43",
    "netmhcpan",
    "netmhcpan-iedb",
    "netmhcpan28",
    "netmhcpan3",
    "netmhcpan4",
    "netmhcpan41",
    "netmhcpan42",
    "netmhcstabpan",
    "pepsickle",
    "prime",
    "random",
    "smm-iedb",
    "smm-pmbec-iedb",
    "tlimmuno2",
)


@pytest.mark.parametrize(
    ("model", "metric", "method", "kind", "field", "sequence"),
    [
        (
            "NetMHCpanEL", "MT Score", "netmhcpan",
            "pMHC_presentation", "score", "mt",
        ),
        (
            "BigMHC_EL", "WT Score", "bigmhc_el",
            "pMHC_presentation", "score", "wt",
        ),
        (
            "NetMHCpan_BA", "Score WT", "netmhcpan",
            "pMHC_affinity", "score", "wt",
        ),
        (
            "NetMHCpan-Aff", "MT Percentile", "netmhcpan",
            "pMHC_affinity", "percentile_rank", "mt",
        ),
        (
            "NetMHCpanAffinity", "Percentile WT", "netmhcpan",
            "pMHC_affinity", "percentile_rank", "wt",
        ),
        (
            "MHCflurryEL", "Processing WT Percentile", "mhcflurry",
            "antigen_processing", "percentile_rank", "wt",
        ),
        (
            "BigMHC_IM", "MT Score", "bigmhc_im",
            "immunogenicity", "score", "mt",
        ),
        (
            "PRIME", "WT Percentile", "prime",
            "immunogenicity", "percentile_rank", "wt",
        ),
        (
            "NetMHCpan", "MT IC50 Score", "netmhcpan",
            "pMHC_affinity", "value", "mt",
        ),
        (
            "new-model", "WT Presentation Percentile", "new_model",
            "pMHC_presentation", "percentile_rank", "wt",
        ),
        (
            "new-model", "Endolysosomal Cleavage MT Score", "new_model",
            "endolysosomal_cleavage", "score", "mt",
        ),
    ],
)
def test_model_metric_vocabulary(
    model, metric, method, kind, field, sequence,
):
    parsed = parse_prediction_metric(model, metric)

    assert parsed is not None
    assert parsed.prediction_method_name == method
    assert parsed.kind == kind
    assert parsed.field == field
    assert parsed.sequence == sequence


@pytest.mark.parametrize("quantity,kind", [
    ("Affinity", "pMHC_affinity"),
    ("Processing", "antigen_processing"),
    ("Presentation", "pMHC_presentation"),
    ("Immunogenicity", "immunogenicity"),
])
def test_every_requested_quantity_can_carry_a_percentile(quantity, kind):
    parsed = parse_prediction_metric(
        "future-predictor", f"Percentile WT {quantity}",
    )

    assert parsed is not None
    assert parsed.kind == kind
    assert parsed.field == "percentile_rank"
    assert parsed.sequence == "wt"


@pytest.mark.parametrize("model", MHCTOOLS_MODEL_BASES)
@pytest.mark.parametrize(("suffix", "kind"), [
    ("EL", "pMHC_presentation"),
    ("_EL", "pMHC_presentation"),
    ("BA", "pMHC_affinity"),
    ("_BA", "pMHC_affinity"),
    ("Aff", "pMHC_affinity"),
    ("_aff", "pMHC_affinity"),
    ("Affinity", "pMHC_affinity"),
])
def test_mhctools_model_vocabulary_accepts_mode_suffixes(
    model, suffix, kind,
):
    parsed = parse_prediction_metric(f"{model}{suffix}", "MT Score")

    assert parsed is not None
    assert parsed.kind == kind
    assert parsed.sequence == "mt"


def test_explicit_quantity_overrides_model_mode():
    parsed = parse_prediction_metric(
        "MHCflurryEL", "Immunogenicity MT Score",
    )
    assert parsed.kind == "immunogenicity"


def test_unknown_bare_score_is_not_guessed():
    assert parse_prediction_metric("future-predictor", "MT Score") is None


def test_conflicting_quantities_are_not_guessed():
    assert parse_prediction_metric(
        "MHCflurryEL", "Processing Presentation Score",
    ) is None
