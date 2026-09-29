"""Native Exacto schema fixtures and explicit fragment conversion."""

from copy import deepcopy
import csv
import gzip
import hashlib
from io import StringIO
import json
from pathlib import Path

import pandas as pd
import pytest

from topiary import (
    EXACTO_SCHEMA_COMMIT, EXACTO_SCHEMAS, combine_sources, read_exacto,
    read_exacto_fragments, read_tsv, reconcile_evidence,
)

ROOT = Path(__file__).parent / "data" / "exacto"


def table(rows, schema):
    stream = StringIO()
    writer = csv.DictWriter(stream, EXACTO_SCHEMAS[schema], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    return stream


def primary_rows():
    # M-SIINFEKL-stop. One producer-annotated changed residue, with shared
    # DNA/RNA identifiers carried on its complete codon.
    coding = "ATGTCTATTATTAATTTTGAAAAACTGTAA"
    protein = "MSIINFEKL*"
    rows = []
    for index, base in enumerate(coding):
        row = dict.fromkeys(EXACTO_SCHEMAS["primary_structures"], "")
        row.update(peptide_id="01", primary_structure_index=index, type="base",
                   amino_acid=protein[index // 3], amino_acid_index=index // 3,
                   codon_index=index % 3, nucleotide=base, transcript_model_id="model-1",
                   reference_transcript_ids="ENST1,ENST2", transcript_structure_index=0,
                   read_start=100 + index, read_end=100 + index,
                   frameshift_state="inframe", net_variant_nucleotides_count=0,
                   amino_acid_change="mutant" if index // 3 == 4 else "reference",
                   rna_variant_call_ids="rna-1" if index // 3 == 4 else "",
                   dna_variant_call_ids="dna-1" if index // 3 == 4 else "")
        rows.append(row)
    return rows


def peptide_rows():
    return [dict(mutant_peptide_id="0001", peptide_id="01", mutant_peptide_sequence="SIINFEKL", k=8,
                 primary_structure_index_start=3, primary_structure_index_end=26,
                 rna_variant_call_ids="rna-1", dna_variant_call_ids="dna-1")]


def test_pinned_native_corpus_and_all_orfs_are_retained():
    manifest = json.loads((ROOT / "provenance.json").read_text())
    assert manifest["commit"] == EXACTO_SCHEMA_COMMIT
    for name, record in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record["sha256"]
    peptides = read_exacto(ROOT / "peptide-variants.tsv", sample_name="p")
    primary = read_exacto(ROOT / "primary-structures.tsv", sample_name="p")
    translations = read_exacto(ROOT / "translations.tsv.gz", sample_name="p")
    assert [len(r) for r in (peptides, primary, translations)] == [228, 6, 77]
    assert primary.df.protein_sequence.notna().all()
    assert translations.df.protein_sequence.notna().sum() == 73
    assert translations.df.peptide.isna().all()
    for result in (peptides, primary, translations):
        assert result.df.allele.isna().all()
        assert result.df.value.isna().all()
        assert result.df.tumor_specificity.isna().all()
        assert result.df.sequence_source.eq("predicted_from_observed_rna").all()
        reconciled = reconcile_evidence(combine_sources({"native": result}))
        assert reconciled.df.candidate_id.isna().all()


def test_primary_context_novelty_and_call_ids_are_preserved():
    result = read_exacto(table(peptide_rows(), "peptide_variants"), sample_name="p",
                         primary_structures=table(primary_rows(), "primary_structures"))
    row = result.df.iloc[0]
    assert row.peptide == "SIINFEKL"
    assert row.sequence == row.protein_sequence == "MSIINFEKL"
    assert (row.peptide_start, row.peptide_end) == (1, 9)
    assert row.n_flank == "M" and row.c_flank == ""
    assert row.mutation_intervals_in_peptide == [[3, 4]]
    assert row.target_intervals == [[4, 5]]
    assert row.exacto_rna_variant_call_ids == ["rna-1"]
    assert row.exacto_dna_variant_call_ids == ["dna-1"]
    assert row.exacto_reference_transcript_ids == ["ENST1", "ENST2"]
    assert row.exacto_record_id == "0001"
    assert result.extra["exacto"]["records"][0]["mutant_peptide_id"] == "0001"


def test_all_native_peptide_occurrences_match_primary_sequences():
    result = read_exacto(ROOT / "peptide-variants.tsv", sample_name="p",
                         primary_structures=ROOT / "primary-structures.tsv")
    assert len(result) == 228
    for row in result.df.itertuples():
        assert row.protein_hypothesis_sequence[row.peptide_start:row.peptide_end] == row.peptide
        assert row.mutation_intervals_in_peptide


def test_native_read_support_remains_transcript_scoped_and_deduplicates_names():
    support = [dict(transcript_model_id="model-1", read_name=name) for name in ("r1", "r2", "r1")]
    result = read_exacto(table(primary_rows(), "primary_structures"), sample_name="p",
                         transcript_read_support=table(support, "transcript_read_support"),
                         library_id="library", read_set_id="alignments", tag="run-1")
    measurement = result.df.iloc[0].rna_observations[0]
    assert measurement["entity_type"] == "transcript"
    assert measurement["value"] == 2
    assert measurement["evidence_unit_ids"] == ["r1", "r2"]
    assert "n_rna_alt" not in result.df and "transcript_expression" not in result.df
    with pytest.raises(ValueError, match="library_id"):
        read_exacto(table(primary_rows(), "primary_structures"), sample_name="p",
                    transcript_read_support=table(support, "transcript_read_support"), tag="run-1")
    unknown = read_exacto(table(primary_rows(), "primary_structures"), sample_name="p",
                          transcript_read_support=table([], "transcript_read_support"),
                          library_id="library", read_set_id="alignments", tag="run-1")
    assert unknown.df.iloc[0].rna_observations == []


def test_translation_coordinates_partial_products_and_originals_roundtrip(tmp_path):
    rows = [dict(peptide_id="p1", peptide_sequence="MA*", rna_id="rna1", rna_sequence="CCCATGGCTTAA",
                 orf_start=3, orf_end=11),
            dict(peptide_id="p2", peptide_sequence="MA", rna_id="rna1", rna_sequence="CCCATGGCT",
                 orf_start=3, orf_end=8)]
    result = read_exacto(table(rows, "translations"), sample_name="p", reference_name="GRCh38")
    assert result.df.orf_end.tolist() == [12, 9]
    assert result.df.orf_completeness.tolist() == ["start_to_stop", "partial_end"]
    assert result.df.protein_sequence.notna().tolist() == [True, False]
    path = tmp_path / "orfs.tsv"
    result.to_tsv(path)
    restored = read_tsv(path)
    assert restored.extra == result.extra
    assert restored.df.protein_hypothesis_sequence.tolist() == ["MA", "MA"]
    assert restored.df.value.isna().all()


@pytest.mark.parametrize("schema", ["peptide_variants", "translations", "primary_structures"])
def test_empty_valid_native_tables_have_a_stable_schema(schema):
    result = read_exacto(table([], schema), sample_name="p")
    assert result.empty
    assert "peptide" in result.df
    assert read_exacto_fragments(table([], schema), sample_name="p") == []
    assert combine_sources({"empty": result}).empty


@pytest.mark.parametrize("text,kwargs,message", [
    ("bogus\nvalue\n", {}, "Unrecognized"),
    ("peptide_id\tpeptide_id\n1\t2\n", {}, "Duplicate"),
    ("bogus\nvalue\n", {"schema": "legacy"}, "Unsupported"),
    ("bogus\nvalue\n", {"schema": "translations"}, "Missing required"),
])
def test_unsupported_native_schemas_raise(text, kwargs, message):
    with pytest.raises(ValueError, match=message):
        read_exacto(StringIO(text), sample_name="p", **kwargs)


@pytest.mark.parametrize("change,message", [
    ({"k": 9}, "disagrees with k"),
    ({"k": "nan"}, "nonnegative integer"),
    ({"peptide_id": ""}, "Empty"),
    ({"primary_structure_index_end": 0}, "reversed"),
])
def test_invalid_peptide_records(change, message):
    rows = peptide_rows()
    rows[0].update(change)
    with pytest.raises(ValueError, match=message):
        read_exacto(table(rows, "peptide_variants"), sample_name="p")


@pytest.mark.parametrize("index,change,message", [
    (0, {"type": "circular"}, "Unsupported.*type"),
    (0, {"type": "event"}, "Sequence-bearing"),
    (0, {"primary_structure_index": 5}, "contiguous"),
    (0, {"read_end": 102}, "one nucleotide"),
    (1, {"codon_index": 2}, "codon/amino-acid"),
    (1, {"read_start": 200, "read_end": 200}, "Gapped"),
    (1, {"amino_acid": "A"}, "disagree"),
    (1, {"amino_acid_change": "unknown"}, "amino_acid_change"),
    (1, {"reference_transcript_ids": "different"}, "differs"),
    (1, {"nucleotide": "Z"}, "Invalid.*coding"),
])
def test_invalid_primary_structures_fail_explicitly(index, change, message):
    rows = primary_rows()
    rows[index].update(change)
    with pytest.raises(ValueError, match=message):
        read_exacto(table(rows, "primary_structures"), sample_name="p")


@pytest.mark.parametrize("change,message", [
    ({"peptide_id": "other"}, "Missing Exacto primary"),
    ({"primary_structure_index_start": 999, "primary_structure_index_end": 1022}, "coding bases"),
    ({"primary_structure_index_start": 4}, "complete codons"),
    ({"mutant_peptide_sequence": "GILGFVFT", "k": 8}, "disagrees with its primary"),
])
def test_peptide_primary_join_is_validated(change, message):
    rows = peptide_rows()
    rows[0].update(change)
    with pytest.raises(ValueError, match=message):
        read_exacto(table(rows, "peptide_variants"), sample_name="p",
                    primary_structures=table(primary_rows(), "primary_structures"))


def test_path_stream_and_gzip_preserve_identifiers_and_missing_context(tmp_path):
    native = table(peptide_rows(), "peptide_variants").getvalue()
    path = tmp_path / "native.tsv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(native)
    from_path = read_exacto(path, sample_name="p")
    from_stream = read_exacto(StringIO(native), sample_name="p")
    pd.testing.assert_frame_equal(from_path.df, from_stream.df)
    assert from_path.df.exacto_record_id.iloc[0] == "0001"
    assert from_path.df.protein_sequence.isna().all()
    assert from_path.df.n_flank.isna().all()
    assert from_path.df.target_intervals.isna().all()
    with pytest.raises(ValueError, match="Ragged"):
        read_exacto(StringIO(native + "short\trow\n"), sample_name="p")
    with pytest.raises(ValueError, match="sample_name"):
        read_exacto(StringIO(native), sample_name="")


def test_fragment_conversion_is_explicit_and_preserves_scoped_provenance():
    native = primary_rows()
    before = deepcopy(native)
    fragments = read_exacto_fragments(table(native, "primary_structures"), sample_name="p")
    assert len(fragments) == 1
    fragment = fragments[0]
    assert fragment.sequence == "MSIINFEKL"
    assert fragment.target_intervals == [(4, 5)]
    assert fragment.reference_sequence is None and fragment.germline_sequence is None
    assert fragment.annotations["sequence_source"] == "predicted_from_observed_rna"
    assert fragment.annotations["tumor_specificity"] is None
    assert len(fragment.annotations["exacto"]["records"]) == len(native)
    assert native == before


def test_read_support_requires_resolvable_run_and_transcript_identity():
    support = [dict(transcript_model_id="model-1", read_name="r1")]
    kwargs = dict(sample_name="p", library_id="library", read_set_id="reads")
    with pytest.raises(ValueError, match="tag"):
        read_exacto(table(primary_rows(), "primary_structures"), **kwargs,
                    transcript_read_support=table(support, "transcript_read_support"))
    for schema, rows in (("peptide_variants", peptide_rows()), ("translations", [])):
        with pytest.raises(ValueError, match="primary-structure"):
            read_exacto(table(rows, schema), **kwargs, tag="run",
                        transcript_read_support=table(support, "transcript_read_support"))
    with pytest.raises(ValueError, match="companion"):
        read_exacto(table(support, "transcript_read_support"), sample_name="p")


def test_fragment_read_membership_retains_only_relevant_native_records():
    support = [dict(transcript_model_id=model, read_name=read)
               for model, read in (("model-1", "r1"), ("other", "r2"))]
    fragments = read_exacto_fragments(table(primary_rows(), "primary_structures"), sample_name="p",
        transcript_read_support=table(support, "transcript_read_support"), tag="run",
        library_id="library", read_set_id="reads")
    evidence = fragments[0].annotations["exacto"]
    assert evidence["transcript_read_support"]["records"] == support[:1]
    assert fragments[0].annotations["exacto_observation"]["rna_observations"][0]["value"] == 1
