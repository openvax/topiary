"""Every source reaches a ProteinFragment with the same core (topiary #102).

The premise of the multi-source abstraction is that a consumer reads one
shape and never branches on where the data came from. That only holds if
*every* path produces a fragment — so isovar has an adapter, the table
readers have one, and varcode already had one.

They differ only in which fields they can fill, and each says how real its
numbers are: isovar counts reads, so its counts are `measured`; pVACseq
derives them from depth x VAF and LENS counts something adjacent, so theirs
are `approximated`.
"""

import warnings

import pandas as pd
import pytest

from topiary import (
    SEMANTIC_CORE,
    ProteinFragment,
    fragment_from_effect,
    fragment_from_isovar_result,
    fragments_for_sample,
    fragments_from_dataframe,
    fragments_from_isovar_results,
    provenance_for_method,
    read_lens,
    read_pvacseq,
)
from topiary.evidence import RNA_ALIGNMENT, RNA_DEPTH_X_VAF

LENS = "tests/data/lens/sample_v1_4.tsv"
PVACSEQ = "tests/data/pvacseq/mhc_i_all_epitopes.tsv"


class _ProteinSequence:
    amino_acids = "MKTVRQERLKSIVRILE"
    mutation_start_idx = 4
    mutation_end_idx = 6
    gene_name = "BRAF"
    transcript_ids = ["ENST1", "ENST2"]
    transcript_names = ["BRAF-204"]
    num_supporting_fragments = 27
    num_supporting_reads = 52


class _IsovarResult:
    """Shaped like isovar.IsovarResult — the attributes actually read."""

    top_protein_sequence = _ProteinSequence()
    variant = "chr7 g.140453136 A>T"
    num_total_fragments = 61
    num_total_reads = 118
    num_alt_fragments = 30
    num_alt_reads = 58
    num_ref_fragments = 31
    num_ref_reads = 60
    num_other_fragments = 2
    num_other_reads = 4


class _NoRNASupport:
    top_protein_sequence = None
    variant = "chr1 g.1 A>T"


class _Effect:
    def __init__(self):
        self.mutant_protein_sequence = "MKTVRQERLK"
        self.original_protein_sequence = "MKTVAQERLK"
        self.aa_mutation_start_offset = 4
        self.aa_mutation_end_offset = 5
        self.gene_name = "BRAF"
        self.gene_id = "ENSG1"
        self.transcript_id = "ENST1"
        self.transcript_name = "BRAF-204"
        self.short_description = "p.A5R"
        self.variant = type("V", (), {"short_description": "chr7:1A>T"})()


def _frame(fn, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(path).df


# ---------------------------------------------------------------------------
# isovar
# ---------------------------------------------------------------------------


def test_isovar_builds_a_fragment():
    fragment = fragment_from_isovar_result(_IsovarResult())

    assert isinstance(fragment, ProteinFragment)
    assert fragment.sequence == "MKTVRQERLKSIVRILE"
    assert fragment.gene == "BRAF"


def test_isovar_counts_are_measured_not_derived():
    """The distinction the whole vocabulary exists for: isovar counted
    these, every other source estimates or counts something adjacent."""
    fragment = fragment_from_isovar_result(_IsovarResult())

    assert fragment.n_rna_alt_fragments == 30
    assert fragment.n_rna_alt_reads == 58
    assert fragment.n_rna_alt == 30        # fragments preferred
    assert fragment.rna_evidence_subject() == "fragments"
    assert not fragment.is_approximate("n_rna_alt_reads")
    assert fragment.is_usable_as_biology("n_rna_alt_reads")


def test_isovar_carries_other_allele_counts_in_both_units():
    fragment = fragment_from_isovar_result(_IsovarResult())

    assert fragment.n_rna_other_fragments == 2
    assert fragment.n_rna_other_reads == 4
    assert fragment.n_rna_other == 2


def test_the_mutated_span_lands_inside_the_sequence():
    fragment = fragment_from_isovar_result(_IsovarResult())

    low, high = fragment.target_intervals[0]
    assert 0 <= low <= high <= len(fragment.sequence)
    assert fragment.sequence[low:high] == "RQ"


def test_isovar_records_that_the_sequence_was_assembled():
    fragment = fragment_from_isovar_result(_IsovarResult())

    assert fragment.annotations["sequence_source"] == "isovar_assembly"
    assert fragment.annotations["rna_evidence_method"] == RNA_ALIGNMENT


def test_supporting_transcripts_are_carried():
    """A release mismatch downstream should be visible, not an empty list."""
    fragment = fragment_from_isovar_result(_IsovarResult())

    assert fragment.annotations["supporting_reference_transcripts"] == [
        "ENST1", "ENST2",
    ]


def test_a_variant_with_no_rna_support_yields_no_fragment():
    """An absence, not an error — a variant with no RNA support is normal."""
    assert fragment_from_isovar_result(_NoRNASupport()) is None


def test_the_list_helper_drops_unsupported_variants():
    fragments = fragments_from_isovar_results([
        _IsovarResult(), _NoRNASupport(), _IsovarResult(),
    ])

    assert len(fragments) == 2
    assert all(f is not None for f in fragments)


def test_isovar_is_not_imported_by_importing_topiary():
    """Optional in the strong sense: not importable, not installed, shape
    unchanged. A consumer reading LENS reports should not pay for it."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, topiary; print('isovar' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )

    assert result.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# The table readers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reader,path", [
    (read_lens, LENS), (read_pvacseq, PVACSEQ),
], ids=["lens", "pvacseq"])
def test_a_reader_frame_becomes_fragments(reader, path):
    fragments = fragments_from_dataframe(_frame(reader, path))

    assert len(fragments) > 0
    assert all(isinstance(f, ProteinFragment) for f in fragments)
    assert all(f.sequence for f in fragments)


def test_a_frame_carries_other_allele_support_into_a_fragment():
    fragment = fragments_from_dataframe(pd.DataFrame([{
        "peptide": "SIINFEKLA",
        "n_rna_other": 3,
        "rna_evidence_subject": "fragments",
    }]))[0]

    assert fragment.n_rna_other_fragments == 3


def test_pvacseq_counts_are_marked_derived():
    fragment = fragments_from_dataframe(_frame(read_pvacseq, PVACSEQ))[0]

    assert fragment.n_rna_alt_reads > 0
    assert fragment.is_approximate("n_rna_alt_reads")
    assert fragment.annotations["rna_evidence_method"] == RNA_DEPTH_X_VAF


def test_a_lens_cds_overlap_count_keeps_its_own_name():
    """LENS counts reads overlapping the peptide's CDS, which is not a
    count of reads supporting the assembled protein sequence. It stays
    under the source's own column rather than being emitted under a name
    that overstates it."""
    frame = _frame(read_lens, LENS)

    assert "lens_rna_reads_covering_genomic_origin_with_peptide_cds" in frame.columns
    assert "n_rna_alt_reads_supporting_protein_sequence" not in frame.columns


def test_a_reader_frame_records_its_sequence_source():
    lens = fragments_from_dataframe(_frame(read_lens, LENS))[0]
    pvac = fragments_from_dataframe(_frame(read_pvacseq, PVACSEQ))[0]

    assert lens.annotations["sequence_source"] == "lens_pep_context"
    assert pvac.annotations["sequence_source"] == "pvacseq_epitope"


def test_a_frame_with_no_sequence_column_says_so():
    with pytest.raises(ValueError, match="No sequence column"):
        fragments_from_dataframe(pd.DataFrame({"gene": ["BRAF"]}))


def test_an_empty_frame_yields_no_fragments():
    assert fragments_from_dataframe(pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# Identity: an ID derives from everything the rows are grouped by
# ---------------------------------------------------------------------------

CONTEXT = "MRKPAAGFLPSLLKVLLLPLAPAAAQ"


def _context_rows(*rows, **shared):
    """Reader-shaped rows sharing one context, one row per reported peptide."""
    return pd.DataFrame([
        {"source_sequence_name": "CTA:ACRBP", "pep_context": CONTEXT,
         "rna_evidence_subject": "reads", **shared, **row}
        for row in rows
    ])


@pytest.mark.parametrize("absent", [
    None, float("nan"), pd.NA, "", "  ", "nan", "None", "<NA>",
], ids=repr)
def test_an_unstated_variant_never_reaches_an_id(absent):
    """``NaN or "fallback"`` is NaN: the absence must be decided once, first."""
    frame = _context_rows(
        {"peptide": "FLPSLLKVL", "n_rna_overlapping": 47},
        {"peptide": "LLLPLAPAA", "n_rna_overlapping": 51},
        variant=absent,
    )

    fragments = fragments_from_dataframe(frame)

    assert len({f.fragment_id for f in fragments}) == 2
    assert all(f.fragment_id.startswith("CTA:ACRBP__") for f in fragments)
    assert all(f.variant is None for f in fragments)


def test_each_reported_peptide_keeps_its_own_evidence():
    """LENS reports evidence per peptide; a shared context must not merge it."""
    frame = _context_rows(
        {"peptide": "FLPSLLKVL", "n_rna_overlapping": 47, "allele": "A"},
        {"peptide": "FLPSLLKVL", "n_rna_overlapping": 47, "allele": "B"},
        {"peptide": "LLLPLAPAA", "n_rna_overlapping": 51, "allele": "A"},
    )

    fragments = fragments_from_dataframe(frame)

    assert {f.annotations["reported_peptide"]: f.n_rna_overlapping_reads
            for f in fragments} == {"FLPSLLKVL": 47, "LLLPLAPAA": 51}
    assert {f.sequence for f in fragments} == {CONTEXT}


def test_rows_describing_one_fragment_must_agree():
    frame = _context_rows(
        {"peptide": "FLPSLLKVL", "n_rna_overlapping": 47},
        {"peptide": "FLPSLLKVL", "n_rna_overlapping": 66},
    )

    with pytest.raises(ValueError, match="differ in n_rna_overlapping_reads"):
        fragments_from_dataframe(frame)


@pytest.mark.parametrize("column,value", [
    ("n_rna_alt", "abc"), ("n_rna_alt", 2.5), ("n_rna_alt", -1),
    ("gene_expression", "high"),
])
def test_a_stated_cell_that_is_not_a_number_is_refused(column, value):
    frame = _context_rows({"peptide": "FLPSLLKVL", column: value})

    with pytest.raises(ValueError, match=column):
        fragments_from_dataframe(frame)


def test_a_frame_sample_labels_its_fragments():
    frame = _context_rows(
        {"peptide": "FLPSLLKVL", "n_rna_overlapping": 47, "sample_name": "T1"},
        {"peptide": "FLPSLLKVL", "n_rna_overlapping": 12, "sample_name": "T2"},
    )

    fragments = fragments_from_dataframe(frame)

    assert {f.annotations["sample_name"]: f.n_rna_overlapping_reads
            for f in fragments} == {"T1": 47, "T2": 12}
    assert {f.fragment_id.split(":")[0] for f in fragments} == {"T1", "T2"}


# ---------------------------------------------------------------------------
# The property that makes the abstraction worth having
# ---------------------------------------------------------------------------


def test_every_path_produces_the_same_core():
    """One shape from four sources; only which fields are filled differs."""
    fragments = {
        "isovar": fragment_from_isovar_result(_IsovarResult()),
        "varcode": fragment_from_effect(_Effect(), padding_around_mutation=2),
        "lens": fragments_from_dataframe(_frame(read_lens, LENS))[0],
        "pvacseq": fragments_from_dataframe(_frame(read_pvacseq, PVACSEQ))[0],
    }

    for source, fragment in fragments.items():
        for name in SEMANTIC_CORE:
            assert hasattr(fragment, name), f"{source} missing {name}"


def test_a_consumer_reads_every_source_through_one_path():
    """No branching on where the data came from — the whole premise."""
    def rna_support(fragment):
        if not fragment.is_usable_as_biology("n_rna_alt_reads"):
            return None
        return (fragment.n_rna_alt_reads, fragment.is_approximate("n_rna_alt_reads"))

    isovar = rna_support(fragment_from_isovar_result(_IsovarResult()))
    pvacseq = rna_support(
        fragments_from_dataframe(_frame(read_pvacseq, PVACSEQ))[0]
    )
    varcode = rna_support(
        fragment_from_effect(_Effect(), padding_around_mutation=2)
    )

    assert isovar == (58, False)          # counted
    assert pvacseq[1] is True             # derived
    assert varcode is None                # no RNA data at all


def test_only_isovar_reports_counted_reads():
    """The property the vocabulary encodes, asserted across sources."""
    counted = {
        "isovar": fragment_from_isovar_result(_IsovarResult()),
        "pvacseq": fragments_from_dataframe(_frame(read_pvacseq, PVACSEQ))[0],
    }

    assert counted["isovar"].provenance_of("n_rna_alt_reads") == "measured"
    assert counted["pvacseq"].provenance_of("n_rna_alt_reads") == "approximated"


def test_the_method_to_provenance_map_is_single_valued():
    """One mapping, so a frame and a fragment cannot disagree about
    whether depth x VAF counts as measured. It does not."""
    assert provenance_for_method(RNA_ALIGNMENT) == "measured"
    assert provenance_for_method(RNA_DEPTH_X_VAF) == "approximated"


def test_a_sample_label_namespaces_the_id_and_is_idempotent():
    fragment = ProteinFragment(fragment_id="v__1", sequence="SIINFEKLL")

    labelled, = fragments_for_sample([fragment], "T2 short-read")
    again, = fragments_for_sample([labelled], "T2 short-read")

    assert labelled.fragment_id == "T2_short-read:v__1"
    assert labelled.annotations == {"sample_name": "T2 short-read"}
    assert again is labelled
    assert fragment.annotations == {}


@pytest.mark.parametrize("sample_name", [None, "", "  ", "nan", "///", 3])
def test_a_missing_sample_label_is_refused(sample_name):
    with pytest.raises(ValueError, match="sample_name"):
        fragments_for_sample([ProteinFragment(fragment_id="v", sequence="S")], sample_name)


def test_one_samples_evidence_is_never_relabelled_as_anothers():
    labelled = fragments_for_sample([ProteinFragment(fragment_id="v", sequence="S")], "T1")

    with pytest.raises(ValueError, match="already labelled with sample 'T1'"):
        fragments_for_sample(labelled, "T2")
