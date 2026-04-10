"""Tests for validation, edge cases, and bugs found during code review.

Covers:
- predict_from_mutation_effects returns DataFrame (not list) on empty input
- --regions rejected with peptide inputs
- _parse_regions validates bounds and coords
- No-input-at-all gives a clear error
- Empty gene/tissue lists rejected
- FASTA parser rejects data before header and empty headers
- predict_from_named_peptides with duplicate peptides
- CLI _get_direct_input for all input paths
- sequences_from_transcript_names and non_cta_sequences coverage
"""

import os
import tempfile
from types import SimpleNamespace

import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import TopiaryPredictor
from topiary.cli.args import (
    _get_direct_input,
    _parse_regions,
    _validate_input_modes,
    arg_parser,
    predict_epitopes_from_args,
    predictors_from_args,
)
from topiary.inputs import read_fasta, read_peptide_fasta
from topiary.sources import (
    sequences_from_gene_ids,
    sequences_from_gene_names,
    sequences_from_transcript_ids,
    sequences_from_transcript_names,
)


def _tmpfile(content, suffix=".csv"):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Bug #1: predict_from_mutation_effects returns DataFrame on empty input
# ---------------------------------------------------------------------------


def test_predict_from_mutation_effects_empty_returns_dataframe():
    """predict_from_mutation_effects must return a DataFrame, not a list."""
    from unittest.mock import MagicMock

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    # Create a mock EffectCollection that filters to empty
    mock_effects = MagicMock()
    mock_effects.groupby_variant.return_value = {}
    # Patch filter_silent_and_noncoding_effects and apply_effect_expression_filters
    # to pass through the mock
    import topiary.predictor as pred_module
    original_filter = pred_module.filter_silent_and_noncoding_effects
    original_expr = pred_module.apply_effect_expression_filters
    try:
        pred_module.filter_silent_and_noncoding_effects = lambda e: e
        pred_module.apply_effect_expression_filters = lambda e, **kw: e
        result = predictor.predict_from_mutation_effects(mock_effects)
    finally:
        pred_module.filter_silent_and_noncoding_effects = original_filter
        pred_module.apply_effect_expression_filters = original_expr

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Bug #3: --regions rejected with peptide inputs
# ---------------------------------------------------------------------------


def test_regions_with_peptide_csv_raises():
    path = _tmpfile("name,peptide\npep1,SIINFEKLA\n")
    try:
        args = arg_parser.parse_args([
            "--mhc-predictor", "random",
            "--mhc-alleles", "A0201",
            "--peptide-csv", path,
            "--regions", "pep1:0-5",
        ])
        with pytest.raises(ValueError, match="--regions cannot be used with peptide"):
            predict_epitopes_from_args(args)
    finally:
        os.unlink(path)


def test_regions_with_peptide_fasta_raises():
    path = _tmpfile(">pep1\nSIINFEKLA\n", ".fasta")
    try:
        args = arg_parser.parse_args([
            "--mhc-predictor", "random",
            "--mhc-alleles", "A0201",
            "--peptide-fasta", path,
            "--regions", "pep1:0-5",
        ])
        with pytest.raises(ValueError, match="--regions cannot be used with peptide"):
            predict_epitopes_from_args(args)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Bug #4: _parse_regions validates bounds
# ---------------------------------------------------------------------------


def test_parse_regions_start_greater_than_end():
    with pytest.raises(ValueError, match="start must be less than end"):
        _parse_regions(["spike:10-5"])


def test_parse_regions_equal_start_and_end():
    with pytest.raises(ValueError, match="start must be less than end"):
        _parse_regions(["spike:5-5"])


def test_parse_regions_empty_interval():
    with pytest.raises(ValueError, match="START-END"):
        _parse_regions(["spike:"])


def test_parse_regions_empty_start():
    with pytest.raises(ValueError, match="coordinates"):
        _parse_regions(["spike:-10"])


def test_parse_regions_non_integer_coords():
    with pytest.raises(ValueError, match="coordinates must be integers"):
        _parse_regions(["spike:abc-def"])


def test_parse_regions_missing_colon():
    with pytest.raises(ValueError, match="NAME:START-END"):
        _parse_regions(["spike319-541"])


def test_parse_regions_missing_dash():
    with pytest.raises(ValueError, match="START-END"):
        _parse_regions(["spike:319"])


def test_parse_regions_valid():
    result = _parse_regions(["spike:319-541", "nuc:0-50"])
    assert result == {"spike": [(319, 541)], "nuc": [(0, 50)]}


def test_parse_regions_multiple_for_same_name():
    result = _parse_regions(["prot:0-10", "prot:20-30"])
    assert result == {"prot": [(0, 10), (20, 30)]}


# ---------------------------------------------------------------------------
# Bug #5: no input at all → clear error
# ---------------------------------------------------------------------------


def test_no_input_raises_clear_error():
    args = arg_parser.parse_args([
        "--mhc-predictor", "random",
        "--mhc-alleles", "A0201",
    ])
    with pytest.raises(ValueError, match="No input specified"):
        predict_epitopes_from_args(args)


def test_multiple_predictors_parse_from_one_cli_invocation():
    args = arg_parser.parse_args([
        "--mhc-predictor", "random", "random",
        "--mhc-alleles", "A0201",
    ])
    predictors = predictors_from_args(args)
    assert len(predictors) == 2


# ---------------------------------------------------------------------------
# Bug #7: empty gene/tissue lists rejected
# ---------------------------------------------------------------------------


def test_sequences_from_gene_names_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        sequences_from_gene_names([])


def test_sequences_from_gene_ids_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        sequences_from_gene_ids([])


def test_sequences_from_transcript_ids_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        sequences_from_transcript_ids([])


def test_sequences_from_transcript_names_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        sequences_from_transcript_names([])


def test_tissue_expressed_gene_ids_empty_raises():
    pytest.importorskip("pirlygenes")
    from topiary.sources import tissue_expressed_gene_ids
    with pytest.raises(ValueError, match="non-empty"):
        tissue_expressed_gene_ids([])


# ---------------------------------------------------------------------------
# Bug #8: FASTA parser rejects malformed input
# ---------------------------------------------------------------------------


def test_fasta_data_before_header_raises():
    path = _tmpfile("SIINFEKL\n>pep1\nELAGIGIL\n", ".fasta")
    try:
        with pytest.raises(ValueError, match="Sequence data before first header"):
            read_fasta(path)
    finally:
        os.unlink(path)


def test_fasta_empty_header_raises():
    path = _tmpfile(">\nSIINFEKL\n", ".fasta")
    try:
        with pytest.raises(ValueError, match="Empty FASTA header"):
            read_fasta(path)
    finally:
        os.unlink(path)


def test_peptide_fasta_data_before_header_raises():
    path = _tmpfile("SIINFEKL\n>pep1\nELAGIGIL\n", ".fasta")
    try:
        with pytest.raises(ValueError, match="Sequence data before first header"):
            read_peptide_fasta(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# predict_from_named_peptides with duplicate peptides
# ---------------------------------------------------------------------------


def test_predict_named_peptides_duplicate_peptides_expanded():
    """Same peptide with different names → each name gets predictions."""
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor(alleles=["A0201"], default_peptide_lengths=[9]),
    )
    peptides = {"name_A": "SIINFEKLA", "name_B": "SIINFEKLA", "name_C": "ELAGIGILT"}
    df = predictor.predict_from_named_peptides(peptides)
    assert len(df) > 0
    names = set(df["source_sequence_name"].unique())
    assert "name_A" in names
    assert "name_B" in names
    assert "name_C" in names


def test_predict_named_peptides_single():
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor(alleles=["A0201"], default_peptide_lengths=[9]),
    )
    df = predictor.predict_from_named_peptides({"pep1": "SIINFEKLA"})
    assert len(df) > 0
    assert set(df["source_sequence_name"].unique()) == {"pep1"}
    # Peptides scored as-is: offset should be 0
    assert (df["peptide_offset"] == 0).all()


def test_predict_named_peptides_empty():
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor(alleles=["A0201"], default_peptide_lengths=[9]),
    )
    df = predictor.predict_from_named_peptides({})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# CLI _get_direct_input for all input paths
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Build a minimal args namespace for _get_direct_input."""
    defaults = dict(
        peptide_csv=None,
        sequence_csv=None,
        fasta=None,
        peptide_fasta=None,
        ensembl_proteome=False,
        gene_names=None,
        gene_ids=None,
        transcript_ids=None,
        cta=False,
        ensembl_release=None,
        regions=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_get_direct_input_peptide_csv():
    path = _tmpfile("name,peptide\npep1,SIINFEKL\n")
    try:
        args = _make_args(peptide_csv=path)
        seqs, is_peptides = _get_direct_input(args)
        assert seqs == {"pep1": "SIINFEKL"}
        assert is_peptides is True
    finally:
        os.unlink(path)


def test_get_direct_input_sequence_csv():
    path = _tmpfile("name,sequence\nbraf,MASIINFEKLGGG\n")
    try:
        args = _make_args(sequence_csv=path)
        seqs, is_peptides = _get_direct_input(args)
        assert seqs == {"braf": "MASIINFEKLGGG"}
        assert is_peptides is False
    finally:
        os.unlink(path)


def test_get_direct_input_fasta():
    path = _tmpfile(">prot1\nMASIINFEKL\n", ".fasta")
    try:
        args = _make_args(fasta=path)
        seqs, is_peptides = _get_direct_input(args)
        assert seqs == {"prot1": "MASIINFEKL"}
        assert is_peptides is False
    finally:
        os.unlink(path)


def test_get_direct_input_peptide_fasta():
    path = _tmpfile(">pep1\nSIINFEKL\n", ".fasta")
    try:
        args = _make_args(peptide_fasta=path)
        seqs, is_peptides = _get_direct_input(args)
        assert seqs == {"pep1": "SIINFEKL"}
        assert is_peptides is True
    finally:
        os.unlink(path)


def test_get_direct_input_gene_names():
    args = _make_args(gene_names=["BRAF"])
    seqs, is_peptides = _get_direct_input(args)
    assert len(seqs) == 1
    assert any("BRAF" in k for k in seqs)
    assert is_peptides is False


def test_get_direct_input_gene_ids():
    args = _make_args(gene_ids=["ENSG00000157764"])  # BRAF
    seqs, is_peptides = _get_direct_input(args)
    assert len(seqs) == 1
    assert is_peptides is False


def test_get_direct_input_transcript_ids():
    args = _make_args(transcript_ids=["ENST00000496384"])  # BRAF
    seqs, is_peptides = _get_direct_input(args)
    assert len(seqs) == 1
    assert is_peptides is False


def test_get_direct_input_none():
    args = _make_args()
    seqs, is_peptides = _get_direct_input(args)
    assert seqs is None
    assert is_peptides is None


def test_get_direct_input_multiple_sources_raises():
    path1 = _tmpfile("name,peptide\npep1,SIINFEKL\n")
    path2 = _tmpfile("name,sequence\nbraf,MASIINFEKLGGG\n")
    try:
        args = _make_args(peptide_csv=path1, sequence_csv=path2)
        with pytest.raises(ValueError, match="Only one sequence source"):
            _get_direct_input(args)
    finally:
        os.unlink(path1)
        os.unlink(path2)


def test_get_direct_input_fasta_with_regions():
    path = _tmpfile(">spike\n" + "A" * 100 + "\n", ".fasta")
    try:
        args = _make_args(fasta=path, regions=["spike:10-20"])
        seqs, is_peptides = _get_direct_input(args)
        assert "spike:10-20" in seqs
        assert len(seqs["spike:10-20"]) == 10
        assert is_peptides is False
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# sources: sequences_from_transcript_names, non_cta_sequences
# ---------------------------------------------------------------------------


def test_sequences_from_transcript_names_valid():
    seqs = sequences_from_transcript_names(["BRAF-204"])
    assert len(seqs) >= 1
    assert any("BRAF" in k for k in seqs)


def test_sequences_from_transcript_names_unknown():
    seqs = sequences_from_transcript_names(["NOTAREAL_TRANSCRIPT_XYZ"])
    assert len(seqs) == 0


def test_non_cta_sequences_excludes_cta_genes():
    """non_cta_sequences should return proteins whose genes are NOT in the CTA set."""
    pytest.importorskip("pirlygenes")
    from topiary.sources import non_cta_sequences, _pirlygenes_cta_gene_ids
    from pyensembl import ensembl_grch38

    cta_ids = _pirlygenes_cta_gene_ids()
    seqs = non_cta_sequences()

    # Should have many proteins (most of the proteome minus CTAs)
    assert len(seqs) > 10000

    # Spot-check: no returned protein should belong to a CTA gene
    for protein_id in list(seqs.keys())[:200]:
        try:
            gene_id = ensembl_grch38.gene_id_of_protein_id(protein_id)
        except ValueError:
            continue
        assert gene_id not in cta_ids, (
            f"Protein {protein_id} (gene {gene_id}) is CTA but was not excluded"
        )
