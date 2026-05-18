"""Direct unit tests for TopiaryPredictor private methods."""

import numpy as np
import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import TopiaryPredictor
from topiary.predictor import _attach_expression_data


# ---------------------------------------------------------------------------
# _format_prediction_df
# ---------------------------------------------------------------------------


def _make_predictor():
    return TopiaryPredictor(models=RandomBindingPredictor, alleles=["A0201"])


def _raw_df(**overrides):
    """Minimal DataFrame mimicking mhctools output before formatting."""
    row = dict(
        peptide="SIINFEKL",
        allele="HLA-A*02:01",
        kind="pMHC_affinity",
        score=0.8,
        value=120.0,
        percentile_rank=0.5,
        offset=10,
        predictor_name="random",
    )
    row.update(overrides)
    return pd.DataFrame([row])


def test_format_renames_columns():
    df = _raw_df()
    result = _make_predictor()._format_prediction_df(df)
    assert "peptide_offset" in result.columns
    assert "prediction_method_name" in result.columns
    assert "offset" not in result.columns
    assert "predictor_name" not in result.columns


def test_format_adds_peptide_length():
    df = _raw_df(peptide="SIINFEKLAA")
    result = _make_predictor()._format_prediction_df(df)
    assert result.iloc[0]["peptide_length"] == 10


def test_format_adds_source_sequence_name_if_missing():
    df = _raw_df()
    df = df.drop(columns=["source_sequence_name"], errors="ignore")
    result = _make_predictor()._format_prediction_df(df)
    assert "source_sequence_name" in result.columns


def test_format_adds_peptide_offset_if_missing():
    df = _raw_df()
    df = df.drop(columns=["offset"], errors="ignore")
    df = df.drop(columns=["peptide_offset"], errors="ignore")
    result = _make_predictor()._format_prediction_df(df)
    assert "peptide_offset" in result.columns
    assert result.iloc[0]["peptide_offset"] == 0


def test_format_affinity_for_affinity_kind():
    df = _raw_df(kind="pMHC_affinity", value=85.0)
    result = _make_predictor()._format_prediction_df(df)
    assert result.iloc[0]["affinity"] == 85.0


def test_format_affinity_nan_for_non_affinity_kind():
    df = _raw_df(kind="pMHC_presentation", value=0.95)
    result = _make_predictor()._format_prediction_df(df)
    assert np.isnan(result.iloc[0]["affinity"])


def test_format_preserves_existing_affinity():
    df = _raw_df()
    df["affinity"] = 999.0
    result = _make_predictor()._format_prediction_df(df)
    assert result.iloc[0]["affinity"] == 999.0


def test_format_empty_df():
    df = pd.DataFrame()
    result = _make_predictor()._format_prediction_df(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_format_value_backfilled_from_score_when_nan():
    df = _raw_df(kind="pMHC_presentation", value=np.nan, score=0.42)
    result = _make_predictor()._format_prediction_df(df)
    assert result.iloc[0]["value"] == 0.42
    assert result.iloc[0]["score"] == 0.42
    assert np.isnan(result.iloc[0]["affinity"])


def test_format_value_preserved_when_already_set():
    df = _raw_df(kind="pMHC_affinity", value=120.0, score=0.8)
    result = _make_predictor()._format_prediction_df(df)
    assert result.iloc[0]["value"] == 120.0
    assert result.iloc[0]["affinity"] == 120.0


def test_format_value_backfill_mixed_kinds():
    df = pd.DataFrame([
        dict(
            peptide="SIINFEKL", allele="HLA-A*02:01", kind="pMHC_affinity",
            score=0.8, value=120.0, percentile_rank=0.5, offset=0,
            predictor_name="random",
        ),
        dict(
            peptide="SIINFEKL", allele="HLA-A*02:01", kind="pMHC_presentation",
            score=0.42, value=np.nan, percentile_rank=2.0, offset=0,
            predictor_name="random",
        ),
    ])
    result = _make_predictor()._format_prediction_df(df)
    aff = result[result["kind"] == "pMHC_affinity"].iloc[0]
    pres = result[result["kind"] == "pMHC_presentation"].iloc[0]
    assert aff["value"] == 120.0
    assert aff["affinity"] == 120.0
    assert pres["value"] == 0.42
    assert np.isnan(pres["affinity"])


def test_format_nan_value_on_affinity_row_not_backfilled():
    # A NaN ``value`` on an affinity/stability row means the unit is
    # genuinely unknown — must NOT be silently rewritten as ``score``,
    # which would falsely advertise a normalized [0, 1] score as an
    # IC50 in nM.
    df = _raw_df(kind="pMHC_affinity", value=np.nan, score=0.8)
    result = _make_predictor()._format_prediction_df(df)
    assert np.isnan(result.iloc[0]["value"])
    assert np.isnan(result.iloc[0]["affinity"])

    df = _raw_df(kind="pMHC_stability", value=np.nan, score=0.8)
    result = _make_predictor()._format_prediction_df(df)
    assert np.isnan(result.iloc[0]["value"])


class _AffinityPlusPresentationModel(RandomBindingPredictor):
    """Wraps RandomBindingPredictor so each predicted peptide yields
    both a pMHC_affinity row (carries IC50 from the random predictor)
    and a pMHC_presentation row (score only, value left None)."""

    def predict_proteins_dataframe(self, name_to_sequence_dict):
        affinity_df = super().predict_proteins_dataframe(name_to_sequence_dict)
        if affinity_df.empty:
            return affinity_df
        presentation_df = affinity_df.copy()
        presentation_df["kind"] = "pMHC_presentation"
        presentation_df["score"] = 0.7
        presentation_df["value"] = np.nan
        return pd.concat([affinity_df, presentation_df], ignore_index=True)


class _ToyVersionModel:
    default_peptide_lengths = [9]
    supported_kinds = ("pMHC_affinity",)

    def __init__(self, name, version, row_version):
        self.prediction_method_name = name
        self.predictor_version = version
        self.row_version = row_version

    def predict_dataframe(self, peptides):
        rows = []
        for peptide in peptides:
            rows.append({
                "peptide": peptide,
                "allele": "HLA-A*02:01",
                "kind": "pMHC_affinity",
                "score": 0.5,
                "value": 100.0,
                "percentile_rank": 1.0,
                "predictor_name": self.prediction_method_name,
            })
        df = pd.DataFrame(rows)
        if self.row_version == "missing":
            return df
        if self.row_version == "blank":
            df["predictor_version"] = ""
        elif self.row_version == "na":
            df["predictor_version"] = pd.NA
        else:
            df["predictor_version"] = self.row_version
        return df


def test_end_to_end_presentation_value_populated():
    # Public-API check: presentation rows produced through
    # predict_from_named_sequences must report value == score (not NaN),
    # while affinity rows keep IC50 in value/affinity.
    model = _AffinityPlusPresentationModel(alleles=["HLA-A*02:01"])
    predictor = TopiaryPredictor(models=[model])
    df = predictor.predict_from_named_sequences({"t": "SIINFEKLAAAAA"})
    pres = df[df["kind"] == "pMHC_presentation"]
    aff = df[df["kind"] == "pMHC_affinity"]
    assert not pres.empty
    assert not aff.empty
    assert not pres["value"].isna().any()
    assert (pres["value"] == pres["score"]).all()
    assert pres["affinity"].isna().all()
    assert not aff["value"].isna().any()
    assert (aff["value"] == aff["affinity"]).all()


@pytest.mark.parametrize("missing_state", ["missing", "blank", "na"])
def test_result_attrs_fill_versions_per_missing_method(tmp_path, missing_state):
    from topiary import TopiaryResult, read_tsv

    predictor = TopiaryPredictor(models=[
        _ToyVersionModel("with_rows", "1.0", "1.0"),
        _ToyVersionModel("from_model", "2.0", missing_state),
    ])

    df = predictor.predict_from_named_peptides({"pep": "SIINFEKLA"})

    assert df.attrs["topiary_models"] == {
        "with_rows": "1.0",
        "from_model": "2.0",
    }
    assert TopiaryResult(df).models == {
        "with_rows": "1.0",
        "from_model": "2.0",
    }

    path = tmp_path / "predictions.tsv"
    TopiaryResult(df).to_tsv(path)

    assert read_tsv(path).models == {
        "with_rows": "1.0",
        "from_model": "2.0",
    }


# ---------------------------------------------------------------------------
# _attach_expression_data: additional edge cases
# ---------------------------------------------------------------------------


def _make_prediction_df():
    return pd.DataFrame([
        dict(
            source_sequence_name="var1", peptide="SIINFEKL", peptide_offset=10,
            allele="HLA-A*02:01", kind="pMHC_affinity",
            score=0.8, value=120.0, percentile_rank=0.5,
            gene_id="ENSG00000157764", transcript_id="ENST00000288602",
            variant="chr7 g.140753336A>T",
        ),
    ])


def test_attach_empty_expression_noop():
    df = _make_prediction_df()
    n_cols = len(df.columns)
    result = _attach_expression_data(df, {"gene": [], "transcript": [], "variant": []})
    assert len(result.columns) == n_cols


def test_attach_duplicate_gene_ids_summed():
    df = _make_prediction_df()
    expr_df = pd.DataFrame({
        "gene_id": ["ENSG00000157764", "ENSG00000157764"],
        "TPM": [20.0, 22.5],
    })
    expr_data = {"gene": [("gene", "gene_id", expr_df)], "transcript": [], "variant": []}
    result = _attach_expression_data(df, expr_data)
    assert "gene_tpm" in result.columns
    assert result.iloc[0]["gene_tpm"] == pytest.approx(42.5)


def test_attach_prefix_lowercasing():
    df = _make_prediction_df()
    expr_df = pd.DataFrame({
        "gene_id": ["ENSG00000157764"],
        "TPM": [42.5],
        "NumReads": [1000],
    })
    expr_data = {"gene": [("salmon", "gene_id", expr_df)], "transcript": [], "variant": []}
    result = _attach_expression_data(df, expr_data)
    assert "salmon_tpm" in result.columns
    assert "salmon_numreads" in result.columns


def test_attach_no_prefix():
    """When name_prefix is None, columns keep original (lowercased) names."""
    df = _make_prediction_df()
    expr_df = pd.DataFrame({
        "variant": ["chr7 g.140753336A>T"],
        "num_alt_reads": [15],
    })
    expr_data = {"gene": [], "transcript": [], "variant": [(None, "variant", expr_df)]}
    result = _attach_expression_data(df, expr_data)
    assert "num_alt_reads" in result.columns
