import pytest

import topiary.sources as sources


# PirlyGenes has shipped per-tissue normalized-TPM columns under two
# spellings; topiary must read tissue names from either without pinning to
# whichever release happens to be installed.


def _fake_expression(naming):
    import pandas as pd

    def column(tissue):
        return f"nTPM_{tissue}" if naming == "prefix" else f"{tissue}_nTPM"

    return pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG1", "ENSG2"],
        "Symbol": ["A", "B"],
        column("lung"): [50.0, 0.1],
        column("testis"): [0.2, 80.0],
    })


@pytest.mark.parametrize("naming", ["prefix", "suffix"])
def test_available_tissues_reads_either_naming_form(naming):
    tissues = sources._tissue_column_map(_fake_expression(naming))

    assert sorted(tissues) == ["lung", "testis"]


@pytest.mark.parametrize("naming", ["prefix", "suffix"])
def test_tissue_columns_resolve_under_either_naming_form(naming):
    pce = _fake_expression(naming)

    cols = sources._tissue_columns(pce, ["testis"])

    assert pce[cols].iloc[1, 0] == 80.0


@pytest.mark.parametrize("naming", ["prefix", "suffix"])
def test_unknown_tissue_names_the_tissue_not_the_column(naming):
    pce = _fake_expression(naming)

    with pytest.raises(ValueError, match=r"Unknown tissue\(s\): \['brain'\]"):
        sources._tissue_columns(pce, ["brain"])


def test_non_string_columns_are_ignored():
    pce = _fake_expression("suffix")
    pce[7] = 1

    assert sorted(sources._tissue_column_map(pce)) == ["lung", "testis"]
