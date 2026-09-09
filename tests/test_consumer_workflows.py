"""End-to-end workflows, tested as wholes rather than parts.

Written after a specific failure: asked whether a downstream consumer was
unblocked, I checked that the four capabilities their design needed were
exported, and said yes. They were exported. They did not *compose* into the
operation being requested — writing a peptide-level row onto an allele to
mean "credit this evidence here" was silently discarded (#232), so every
attribution policy produced identical scores.

Checking that parts exist is not checking that the whole works. Each test
here walks a documented workflow from input to answer, so a claim that "X is
supported" has something that runs behind it.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from topiary import (
    DEFAULT_PROTEIN_SEQUENCE_LENGTH,
    EvalContext,
    Presentation,
    ProteinFragment,
    TopiaryPredictor,
    TopiaryResult,
    aggregate_evidence_across_samples,
    apply_filter,
    apply_sort,
    attach_dna_evidence,
    attach_rna_evidence,
    describe_read_evidence,
    evaluate_scores,
    fragment_from_effect,
    fragment_from_isovar_result,
    fragments_from_dataframe,
    fragments_from_variants,
    peptide_view,
    read_lens,
    read_pvacseq,
    resolve_default_methods,
    resolve_default_versions,
    read_fragments,
    stack_results,
    write_fragments,
)
from topiary.ranking import parse

LENS = "tests/data/lens/sample_v1_4.tsv"
PVACSEQ = "tests/data/pvacseq/mhc_i_all_epitopes.tsv"
PVACSEQ_PRESENTATION = (
    "tests/data/pvacseq/mhc_i_all_epitopes_presentation.tsv"
)


def _long(reader, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = reader(path)
        return result.to_long().df if result.metadata.form == "wide" else result.df


# ---------------------------------------------------------------------------
# Per-sample results, stacked and pooled without losing either view
# ---------------------------------------------------------------------------


def test_stacked_sample_evidence_can_be_pooled_as_a_separate_view():
    def result(sample_name, alt, overlapping):
        return TopiaryResult(pd.DataFrame([{
            "fragment_id": "fragment-1",
            "peptide": "SIINFEKL",
            "peptide_offset": 3,
            "allele": "HLA-A*02:01",
            "sample_name": sample_name,
            "kind": "pMHC_affinity",
            "prediction_method_name": "netmhcpan",
            "predictor_version": "4.1",
            "value": 50.0,
            "n_rna_alt": alt,
            "n_rna_overlapping": overlapping,
            "rna_vaf": alt / overlapping,
            "rna_evidence_subject": "reads",
            "rna_evidence_method": "rna_alignment",
        }]))

    stacked = stack_results([
        result("tumor_pre", 40, 100),
        result("tumor_post", 20, 80),
    ])

    pooled = aggregate_evidence_across_samples(stacked.df)

    assert list(stacked.df["sample_name"]) == ["tumor_pre", "tumor_post"]
    assert list(stacked.df["n_rna_alt"]) == [40, 20]
    assert len(pooled) == 1
    assert pooled.loc[0, "n_samples"] == 2
    assert pooled.loc[0, "n_rna_alt"] == 60
    assert pooled.loc[0, "n_rna_overlapping"] == 180
    assert pooled.loc[0, "rna_vaf"] == pytest.approx(60 / 180)


@pytest.mark.parametrize(
    ("attach", "argument", "assay"),
    [
        (attach_rna_evidence, "overlapping", "rna"),
        (attach_dna_evidence, "depth", "dna"),
    ],
)
def test_topiary_depth_only_evidence_can_be_stacked_and_pooled(
    attach, argument, assay,
):
    """Both evidence writers compose with the cross-sample aggregator."""
    base = pd.DataFrame([{
        "fragment_id": "fragment-1",
        "peptide": "SIINFEKL",
        "peptide_offset": 3,
        "allele": "HLA-A*02:01",
        "kind": "pMHC_affinity",
        "prediction_method_name": "netmhcpan",
        "predictor_version": "4.1",
        "value": 50.0,
    }])

    results = []
    for sample_name, depth in (("tumor_pre", 50), ("tumor_post", 70)):
        frame = attach(base, **{argument: pd.Series([depth])})
        frame["sample_name"] = sample_name
        results.append(TopiaryResult(frame))

    stacked = stack_results(results)
    pooled = aggregate_evidence_across_samples(stacked.df)

    assert pooled.loc[0, "n_samples"] == 2
    assert pooled.loc[0, f"n_{assay}_overlapping"] == 120
    assert pooled.loc[0, f"{assay}_evidence_subject"] == "reads"
    assert f"{assay}_evidence_method" not in pooled.columns


# ---------------------------------------------------------------------------
# A LENS report, read and scored
# ---------------------------------------------------------------------------


def test_a_lens_report_can_be_filtered_and_sorted_by_a_dsl_expression():
    """The documented shape of a run, not its ingredients."""
    df = _long(read_lens, LENS)
    expression = parse(
        "affinity['netmhcpan'].value.logistic_normalized(350, 150)"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        kept = apply_filter(df, parse("affinity['netmhcpan'].value <= 5000"))
        ordered = apply_sort(kept, [expression])
        scores = evaluate_scores(ordered, expression)

    assert len(kept) > 0
    assert len(ordered) == len(kept)
    assert scores.notna().any()


@pytest.mark.parametrize("expression", [
    "gene_tpm > 1",
    "lens_vaf > 0.1",
    "rna_vaf > 0.1",
    "n_rna_alt > 5",
    "affinity['netmhcpan'].value.logistic_normalized(350,150) * (gene_tpm > 1)",
])
def test_a_lens_annotation_is_addressable_from_the_dsl(expression):
    """The claim: LENS annotations reach the DSL. Run it, do not infer it.

    Note the name: `read_lens` renames `tpm` to `gene_tpm` (keeping the raw
    string in `gene_tpm_raw`, since LENS writes fusion rows as composites).
    An earlier assessment of this quoted `tpm` and would have failed.
    """
    df = _long(read_lens, LENS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scores = evaluate_scores(df, parse(expression))

    assert len(scores) == len(df)


def test_the_lens_renames_are_what_the_dsl_sees():
    df = _long(read_lens, LENS)

    for original, renamed in (
        ("tpm", "gene_tpm"), ("gene_name", "gene"),
        ("variant_coords", "variant"),
        # LENS's own fraction keeps LENS's name: unqualified `vaf` would
        # be unattributable next to another tool's VAF in a stacked frame.
        ("vaf", "lens_vaf"),
    ):
        assert renamed in df.columns, f"{original} should surface as {renamed}"
        assert original not in df.columns


# ---------------------------------------------------------------------------
# Multi-version and multi-method frames, resolved and scored
# ---------------------------------------------------------------------------


def test_the_resolver_output_actually_scores_the_frame():
    """resolve -> evaluate is the documented loop; run the loop."""
    df = _long(read_lens, LENS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scores = evaluate_scores(
            df, parse("affinity.value"),
            default_methods=resolve_default_methods(df),
            default_versions=resolve_default_versions(df),
        )

    assert scores.notna().any()


def test_an_unresolved_multi_method_frame_still_refuses():
    """The safety half of the same loop."""
    df = _long(read_lens, LENS)

    with pytest.raises(ValueError, match="Ambiguous"):
        evaluate_scores(df, parse("affinity.value"))


def test_a_pvacseq_presentation_report_can_be_resolved_and_scored():
    """The pVACseq -> Topiary -> Vaxrank-shaped scoring path is live."""
    df = _long(read_pvacseq, PVACSEQ_PRESENTATION)
    methods = resolve_default_methods(df)
    scores = evaluate_scores(
        df,
        Presentation.score,
        default_methods=methods,
    )

    assert methods["pMHC_presentation"] == "mhcflurry"
    assert scores.tolist() == pytest.approx([0.91] * len(df))


# ---------------------------------------------------------------------------
# Allele attribution — the composition that was missing
# ---------------------------------------------------------------------------


def _attribution_frame(processing_allele):
    """Two alleles scored, plus one peptide-level row credited somewhere."""
    rows = [
        dict(source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
             allele=allele, kind="pMHC_affinity", value=value, score=0.5,
             percentile_rank=1.0, prediction_method_name="netmhcpan",
             predictor_version="4.1")
        for allele, value in (("HLA-A*02:01", 50.0), ("HLA-B*07:02", 900.0))
    ]
    rows.append(dict(
        source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
        allele=processing_allele, kind="antigen_processing", value=0.8,
        score=0.8, percentile_rank=1.0,
        prediction_method_name="mhcflurry", predictor_version="2.1",
    ))
    return pd.DataFrame(rows)


def _scores(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return evaluate_scores(
            frame, parse("antigen_processing['mhcflurry'].score"),
        )


def test_narrowing_attribution_changes_the_answer():
    """The operation a policy needs, end to end.

    Every capability this uses was already exported before #232, and the
    workflow still did not work — which is the whole reason this file
    exists. Asserting the *difference* is what "the policy has an effect"
    means; asserting the pieces exist is not.
    """
    whole_genotype = _scores(_attribution_frame(None))
    one_allele = _scores(_attribution_frame("HLA-A*02:01"))

    assert whole_genotype.notna().sum() > one_allele.notna().sum()


def test_peptide_view_composes_with_a_score_expression():
    """peptide_view inside arithmetic, which is how it is documented."""
    frame = _attribution_frame(None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scores = evaluate_scores(
            frame,
            peptide_view(parse("antigen_processing.score"))
            * parse("affinity.score"),
        )

    assert scores.notna().any()


# ---------------------------------------------------------------------------
# Every source to a fragment, and back out through a consumer
# ---------------------------------------------------------------------------


class _ProteinSequence:
    amino_acids = "MKTVRQERLKSIVRILE"
    mutation_start_idx = 4
    mutation_end_idx = 6
    gene_name = "BRAF"
    transcript_ids = ["ENST1"]
    transcript_names = ["BRAF-204"]
    num_supporting_fragments = 27
    num_supporting_reads = 52


class _IsovarResult:
    top_protein_sequence = _ProteinSequence()
    variant = "chr7 g.140453136 A>T"
    num_total_fragments = 61
    num_total_reads = 118
    num_alt_fragments = 30
    num_alt_reads = 58
    num_ref_fragments = 31
    num_ref_reads = 60


class _Effect:
    mutant_protein_sequence = "MKTVRQERLK"
    original_protein_sequence = "MKTVAQERLK"
    aa_mutation_start_offset = 4
    aa_mutation_end_offset = 5
    gene_name = "BRAF"
    gene_id = "ENSG1"
    transcript_id = "ENST1"
    transcript_name = "BRAF-204"
    short_description = "p.A5R"
    variant = type("Variant", (), {"short_description": "chr7:1A>T"})()


def test_one_consumer_function_reads_every_source():
    """The multi-source premise, exercised rather than described."""
    def support(fragment):
        if not fragment.is_usable_as_biology("n_rna_alt_reads"):
            return None
        return fragment.is_approximate("n_rna_alt_reads")

    sources = {
        "isovar": fragment_from_isovar_result(_IsovarResult()),
        "varcode": fragment_from_effect(_Effect(), padding_around_mutation=2),
        "lens": fragments_from_dataframe(_long(read_lens, LENS))[0],
        "pvacseq": fragments_from_dataframe(_long(read_pvacseq, PVACSEQ))[0],
    }
    answers = {name: support(f) for name, f in sources.items()}

    assert answers["isovar"] is False        # counted
    assert answers["varcode"] is None        # no RNA evidence
    assert answers["pvacseq"] is True        # derived
    assert "lens" in answers                 # whatever LENS has, one call


def test_read_evidence_can_be_reported_without_walking_rows():
    """describe_read_evidence is for telling a user how numbers were got."""
    described = describe_read_evidence(_long(read_pvacseq, PVACSEQ))

    assert described
    assert all(isinstance(v, str) for v in described.values())


# ---------------------------------------------------------------------------
# A context, shared the way the docs say to share it
# ---------------------------------------------------------------------------


def test_a_shared_context_serves_several_operations_on_one_frame():
    df = _long(read_lens, LENS)
    context = EvalContext(df, default_methods=resolve_default_methods(df))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        first = evaluate_scores(df, parse("affinity.value"), context=context)
        second = evaluate_scores(df, parse("affinity.score"), context=context)
        ordered = apply_sort(df, [parse("affinity.value")], context=context)

    assert len(first) == len(second) == len(df)
    assert len(ordered) == len(df)


def test_a_context_from_another_frame_is_still_refused():
    """The guard that makes sharing safe, in the workflow it guards."""
    df = _long(read_lens, LENS)
    context = EvalContext(df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        smaller = apply_filter(
            df, parse("affinity['netmhcpan'].value <= 5000"),
        )

    with pytest.raises(ValueError, match="different DataFrame"):
        evaluate_scores(smaller, parse("affinity.score"), context=context)


# ---------------------------------------------------------------------------
# Real Isovar RNA assembly → fragments → predictions → ranking DSL (#279)
# ---------------------------------------------------------------------------


@pytest.fixture
def isovar_fragments_from_reads(monkeypatch):
    """Supply read/reference inputs without replacing the Isovar pipeline.

    Imports happen after the integration marker's availability check. No BAM
    decoding or downloaded Ensembl data is needed: only read collection,
    reference lookup, and effect annotation are supplied here. The installed
    run_isovar, assembly, translation, IsovarResult, and Topiary adapters run.

    Sequences are specified in transcript orientation, with reads converted
    to genomic orientation on the minus strand. Codon expectations follow
    NCBI's standard genetic code and CDS strand conventions:
    https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi#SG1
    https://www.ncbi.nlm.nih.gov/genbank/feature_table/
    """
    from varcode import Variant
    from isovar.allele_read import AlleleRead
    from isovar.dna import reverse_complement_dna
    from isovar.protein_sequence_creator import ProteinSequenceCreator
    from isovar.read_evidence import ReadEvidence
    from isovar.reference_context import ReferenceContext

    def assemble(
        strand, assembly, ref, alt, prefixes, suffix,
        protein_sequence_length=DEFAULT_PROTEIN_SEQUENCE_LENGTH,
    ):
        def genomic(sequence):
            return sequence if strand == "+" else reverse_complement_dna(sequence)

        variant = Variant("1", 100, genomic(ref), genomic(alt), "GRCh38")
        reads = []
        for index, prefix in enumerate(prefixes):
            left, right = (prefix, suffix) if strand == "+" else (suffix, prefix)
            reads.append(AlleleRead(
                genomic(left), genomic(alt), genomic(right), str(index),
                source_read_count=2,
            ))
        evidence = ReadEvidence.from_variant_and_allele_reads(variant, reads)
        context = ReferenceContext(
            strand=strand,
            sequence_before_variant_locus=min(prefixes, key=len),
            sequence_at_variant_locus=ref,
            sequence_after_variant_locus=suffix,
            offset_to_first_complete_codon=0,
            contains_start_codon=False,
            overlaps_start_codon=False,
            contains_five_prime_utr=False,
            amino_acids_before_variant="",
            variant=variant,
            transcripts=(),
        )

        class Collector:
            def read_evidence_generator(self, variants, alignment_file):
                assert list(variants) == [variant]
                yield variant, evidence

        with monkeypatch.context() as inputs:
            inputs.setattr(
                "isovar.protein_sequence_creator.reference_contexts_for_variant",
                lambda variant, **kwargs: [context],
            )
            inputs.setattr("isovar.main.top_varcode_effect", lambda variant, **kwargs: None)
            return fragments_from_variants(
                [variant], alignment_file=object(), read_collector=Collector(),
                protein_sequence_creator=ProteinSequenceCreator(
                    variant_sequence_assembly=assembly,
                    protein_sequence_length=protein_sequence_length,
                ),
                filter_thresholds={}, filter_flags=[],
            )

    return assemble


def _isovar_prediction_frame(fragment):
    """Exercise peptide selection and evidence handoff, not MHC accuracy."""
    from mhctools import RandomBindingPredictor

    model = RandomBindingPredictor(
        alleles=["HLA-A*02:01"], default_peptide_lengths=[9],
    )
    predictor = TopiaryPredictor(models=model, only_novel_epitopes=True)
    return predictor.predict_from_fragments([fragment])


@pytest.mark.isovar
@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("assembly", [False, True])
def test_isovar_multibase_mutation_keeps_second_changed_codon(
    isovar_fragments_from_reads, strand, assembly,
):
    # AT[GAA]A → AT[TCC]A changes ATG/AAA (MK) to ATT/CCA (IP).
    fragment, = isovar_fragments_from_reads(
        strand, assembly, "GAA", "TCC", ["AAA" * 4 + "AT"] * 2, "A" + "GGG" * 8,
    )
    assert fragment.sequence == "KKKKIP" + "G" * 8
    assert list(fragment.target_intervals) == [(4, 6)]

    frame = _isovar_prediction_frame(fragment)
    # The 9-mer beginning on the second mutant residue was lost with the
    # incorrect [4, 5) interval, despite a correctly translated sequence.
    assert "P" + "G" * 8 in set(frame.peptide)
    assert frame.contains_mutant_residues.all()


@pytest.mark.isovar
@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("assembly", [False, True])
@pytest.mark.parametrize("protein_sequence_length,sequence,interval", [
    (21, "TTT" + "K" * 15 + "GGG", (3, 18)),
    (49, "TTTT" + "K" * 15 + "G" * 10, (4, 19)),
])
def test_isovar_long_insertion_reaches_fragment_and_predictions(
    isovar_fragments_from_reads, strand, assembly,
    protein_sequence_length, sequence, interval,
):
    # Explicit windows cover Topiary's default and the longer context used by
    # Isovar 1.8.0. A dependency's default must not determine this fixture.
    fragment, = isovar_fragments_from_reads(
        strand, assembly, "", "A" * 45, ["ACG" * 4] * 3, "G" * 30,
        protein_sequence_length=protein_sequence_length,
    )
    assert fragment.sequence == sequence
    assert list(fragment.target_intervals) == [interval]
    assert fragment.n_rna_alt_reads_supporting_protein_sequence == 6
    assert fragment.n_rna_alt_fragments_supporting_protein_sequence == 3

    frame = _isovar_prediction_frame(fragment)
    assert "K" * 9 in set(frame.peptide)
    assert frame.contains_mutant_residues.all()


@pytest.mark.isovar
@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("assembly", [False, True])
def test_isovar_long_insertion_still_requires_real_flanking_context(
    isovar_fragments_from_reads, strand, assembly,
):
    # Nine transcript-prefix bases cannot satisfy Isovar's ten-base minimum.
    assert isovar_fragments_from_reads(
        strand, assembly, "", "A" * 45, ["ACG" * 3] * 3, "G" * 30,
    ) == []


@pytest.mark.isovar
@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("assembly", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_isovar_shared_support_survives_adapter_and_dsl(
    isovar_fragments_from_reads, strand, assembly, reverse,
):
    prefixes = [
        prefix + "A" * 12
        for prefix in ("", "GGG", "CCCGGG", "TTTCCCGGG", "GGGTTTCCCGGG")
    ]
    if reverse:
        prefixes.reverse()
    fragment, = isovar_fragments_from_reads(
        strand, assembly, "G", "C", prefixes, "A" * 30,
    )
    frame = _isovar_prediction_frame(fragment)
    assert not frame.empty

    # Five read pairs contribute ten raw reads, not five: neither unit may
    # disappear or be substituted for the other at either public handoff.
    for field, count in (
        ("n_rna_alt_reads_supporting_protein_sequence", 10),
        ("n_rna_alt_fragments_supporting_protein_sequence", 5),
    ):
        assert getattr(fragment, field) == count
        assert fragment.provenance_of(field) == "measured"
        assert evaluate_scores(frame, parse(field)).eq(count).all()
        assert not apply_filter(frame, parse(f"{field} >= {count}")).empty
        assert apply_filter(frame, parse(f"{field} > {count}")).empty


# ---------------------------------------------------------------------------
# Original osteosarc RNA → peptide-aware context → IO/prediction (#284)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def osteosarc_rna(tmp_path_factory):
    from tests.osteosarc_helpers import load_osteosarc

    return load_osteosarc(tmp_path_factory.mktemp("topiary_osteosarc"))


def osteosarc_fragments(data, sample, gene, **options):
    import pysam

    variants, bams, _ = data
    with pysam.AlignmentFile(bams[sample]) as bam:
        return fragments_from_variants(
            [variants[gene]], bam, filter_thresholds={}, filter_flags=[], **options,
        )


@pytest.mark.parametrize("value,expected", [
    (np.bool_(True), True),
    (np.bool_(False), False),
    (np.int32(-3), -3),
    (np.int64(2**60 + 1), 2**60 + 1),
    (np.uint64(2**64 - 1), 2**64 - 1),
    (np.float32(0.5), 0.5),
    (np.float64(0.85), 0.85),
    (np.str_("True"), "True"),
])
def test_fragment_numpy_scalars_survive_all_serialization_doors(value, expected, tmp_path):
    from tests.test_twin_conformance import (
        FRAGMENT_CONSTRUCTION_DOORS, FRAGMENT_SERIALIZATION_DOORS,
    )

    for _, construct in FRAGMENT_CONSTRUCTION_DOORS:
        annotations = {"nested": [{"value": value}]}
        fragment = construct(dict(
            target_intervals=[(np.int64(1), np.int32(2))],
            n_rna_alt_reads=np.int64(3), gene_expression=np.float32(0.5),
            annotations=annotations,
        ))
        assert type(fragment.annotations["nested"][0]["value"]) is type(expected)
        assert fragment.annotations["nested"][0]["value"] == expected
        assert type(annotations["nested"][0]["value"]) is type(value)
        # Mutable annotations added after construction must serialize too.
        fragment.annotations["later"] = {"value": value}
        for name, roundtrip in FRAGMENT_SERIALIZATION_DOORS:
            restored = roundtrip(fragment, tmp_path / f"{name}.tsv")
            assert restored.to_dict() == fragment.to_dict()
            assert type(restored.n_rna_alt_reads) is int
            assert type(restored.gene_expression) is float
            assert all(type(i) is int for pair in restored.target_intervals for i in pair)
            for scalar in (restored.annotations["nested"][0]["value"],
                           restored.annotations["later"]["value"]):
                assert type(scalar) is type(expected)
                assert scalar == expected


@pytest.mark.isovar
@pytest.mark.parametrize("assembly", [True, False])
def test_osteosarc_numpy_custom_creator_survives_serialization(osteosarc_rna, tmp_path, assembly):
    from isovar.protein_sequence_creator import ProteinSequenceCreator
    from tests.osteosarc_helpers import assert_expected_fragment
    from tests.test_twin_conformance import FRAGMENT_SERIALIZATION_DOORS

    fragments = []
    for bool_type in (bool, np.bool_):
        creator = ProteinSequenceCreator(
            variant_sequence_assembly=bool_type(assembly),
            count_mismatches_after_variant=bool_type(False),
            protein_context_peptide_length=np.int64(25),
            min_variant_sequence_coverage=np.int32(2),
            min_protein_sequence_support_fraction=np.float64(0.85),
        )
        fragment, = osteosarc_fragments(
            osteosarc_rna, "bulk_star_t0", "DYNC1H1", protein_sequence_creator=creator,
        )
        assert_expected_fragment(fragment, osteosarc_rna[2]["DYNC1H1"])
        for name, roundtrip in FRAGMENT_SERIALIZATION_DOORS:
            restored = roundtrip(fragment, tmp_path / f"{name}.tsv")
            assert restored.to_dict() == fragment.to_dict()
            assert restored.annotations["isovar_variant_sequence_assembly"] is assembly
            assert restored.annotations["isovar_count_mismatches_after_variant"] is False
        assert fragment.annotations["isovar_variant_sequence_assembly"] is assembly
        assert type(creator.variant_sequence_assembly) is bool_type
        frame = _isovar_prediction_frame(fragment)
        assert not frame.empty
        assert frame.isovar_variant_sequence_assembly.eq(assembly).all()
        fragments.append(fragment.to_dict())
    assert fragments[0] == fragments[1]


@pytest.mark.isovar
@pytest.mark.parametrize("sample", ["bulk_star_t0", "ont_t1"])
@pytest.mark.parametrize("peptide", [15, 25, 30])
@pytest.mark.parametrize("floor", [2, 5])
def test_osteosarc_peptide_size_and_floor_match_the_explicit_creator(
    osteosarc_rna, sample, peptide, floor,
):
    import pysam
    from tests.osteosarc_helpers import assert_expected_fragment
    from tests.test_twin_conformance import ISOVAR_RECONSTRUCTION_TWINS

    variants, bams, expected = osteosarc_rna
    twin = ISOVAR_RECONSTRUCTION_TWINS
    results = []
    for door in (twin.left, twin.right):
        with pysam.AlignmentFile(bams[sample]) as bam:
            fragment, = door(
                [variants["DYNC1H1"]], bam,
                protein_context_peptide_length=peptide,
                protein_sequence_preference="balanced",
                min_protein_sequence_support_fraction=0.85,
                min_variant_sequence_coverage=floor,
            )
        assert_expected_fragment(fragment, expected["DYNC1H1"])
        if sample == "bulk_star_t0":
            length = 2 * peptide - 1 if floor == 2 else 16
            counts = (9, 6)
        else:
            length, counts = 20, (11, 11)
        assert len(fragment.sequence) == length
        assert (fragment.n_rna_alt_reads_supporting_protein_sequence,
                fragment.n_rna_alt_fragments_supporting_protein_sequence) == counts
        assert fragment.annotations["isovar_protein_sequence_length"] == 2 * peptide - 1
        results.append(fragment.to_dict())
    assert results[0] == results[1]


@pytest.mark.isovar
@pytest.mark.parametrize("sample,gene,length", [
    ("bulk_star_t0", "EXOC4", 49), ("ont_t1", "EXOC4", 25),
    ("bulk_star_t0", "H1-2", None), ("ont_t1", "H1-2", 30),
    ("bulk_star_t0", "GTF3C5", 49), ("ont_t1", "GTF3C5", 29),
    ("bulk_star_t0", "PIP5K1A", None), ("ont_t1", "PIP5K1A", 46),
    ("bulk_star_t0", "MAP2", None), ("ont_t1", "MAP2", None),
])
def test_osteosarc_real_edits_survive_context_and_prediction(
    osteosarc_rna, sample, gene, length,
):
    from tests.osteosarc_helpers import assert_expected_fragment

    fragments = osteosarc_fragments(
        osteosarc_rna, sample, gene, protein_context_peptide_length=25,
    )
    if length is None:
        assert fragments == []
        return
    fragment, = fragments
    assert len(fragment.sequence) == length
    assert_expected_fragment(fragment, osteosarc_rna[2][gene])
    frame = _isovar_prediction_frame(fragment)
    assert not frame.empty
    assert frame.contains_mutant_residues.all()
    start, end = fragment.target_intervals[0]
    if gene == "GTF3C5":
        assert start == end  # zero-width novel adjacency, not a mutant residue
    for row in frame.itertuples():
        assert row.peptide == fragment.sequence[row.peptide_offset:row.peptide_offset + 9]
        assert row.peptide_offset < end and row.peptide_offset + 9 > start


@pytest.mark.isovar
def test_osteosarc_context_settings_survive_io_predictions_and_dsl(osteosarc_rna, tmp_path):
    short, = osteosarc_fragments(osteosarc_rna, "bulk_star_t0", "DYNC1H1")
    long, = osteosarc_fragments(
        osteosarc_rna, "bulk_star_t0", "DYNC1H1", protein_context_peptide_length=25,
    )
    assert (len(short.sequence), len(long.sequence)) == (21, 49)
    assert short.annotations["isovar_protein_context_peptide_length"] == 11
    path = tmp_path / "rna-fragments.tsv"
    write_fragments([short, long], path)
    reloaded = read_fragments(path)
    assert [f.to_dict() for f in reloaded] == [f.to_dict() for f in (short, long)]
    frames = []
    for fragment in reloaded:
        frame = _isovar_prediction_frame(fragment)
        for key, value in fragment.annotations.items():
            if key.startswith("isovar_"):
                assert frame[key].eq(value).all()
        assert frame.n_rna_alt_reads_supporting_protein_sequence.eq(9).all()
        assert frame.n_rna_alt_fragments_supporting_protein_sequence.eq(6).all()
        frames.append(frame)
    # Both contexts already contain every mutant 9mer. The larger RNA
    # objective changes available vaccine windows, not MHC prediction lengths.
    assert set(frames[0].peptide) == set(frames[1].peptide)
    window_counts = []
    for fragment in reloaded:
        start, end = fragment.target_intervals[0]
        window_counts.append(sum(i < end and i + 25 > start
                                 for i in range(len(fragment.sequence) - 25 + 1)))
    assert window_counts == [0, 25]
    combined = pd.concat(frames, ignore_index=True)
    selected = apply_filter(combined, parse("isovar_protein_context_peptide_length >= 25"))
    assert set(selected.fragment_id) == {long.fragment_id}


@pytest.mark.isovar
def test_osteosarc_relative_support_changes_context_without_changing_allele_counts(osteosarc_rna):
    from tests.osteosarc_helpers import assert_expected_fragment

    fragments = []
    for options in ({}, {"min_protein_sequence_support_fraction": 0.8},
                    {"protein_sequence_preference": "context"}):
        fragment, = osteosarc_fragments(
            osteosarc_rna, "ont_t1", "DYNC1H1",
            protein_context_peptide_length=25, **options,
        )
        assert_expected_fragment(fragment, osteosarc_rna[2]["DYNC1H1"])
        fragments.append(fragment)
    assert [len(f.sequence) for f in fragments] == [20, 37, 49]
    assert [f.n_rna_alt_fragments_supporting_protein_sequence for f in fragments] == [11, 9, 7]
    assert {f.n_rna_alt_fragments for f in fragments} == {16}
    assert {f.annotations["isovar_min_variant_sequence_coverage"] for f in fragments} == {2}
    # The default is allowed to be too short for a full 25mer. Neither an
    # implicit reference extension nor a relaxed budget manufactures one.
    assert len(fragments[0].sequence) < 25


@pytest.mark.isovar
def test_osteosarc_support_preference_and_explicit_length_are_respected(osteosarc_rna):
    from tests.osteosarc_helpers import assert_expected_fragment
    from isovar.protein_sequence_creator import ProteinSequenceCreator

    options = dict(protein_context_peptide_length=25, protein_sequence_length=20,
                   protein_sequence_preference="support")
    explicit, = osteosarc_fragments(osteosarc_rna, "ont_t1", "DYNC1H1", **options)
    custom, = osteosarc_fragments(
        osteosarc_rna, "ont_t1", "DYNC1H1",
        protein_sequence_creator=ProteinSequenceCreator(variant_sequence_assembly=True, **options),
    )
    assert explicit.to_dict() == custom.to_dict()
    assert_expected_fragment(explicit, osteosarc_rna[2]["DYNC1H1"])
    assert len(explicit.sequence) <= 20


@pytest.mark.isovar
@pytest.mark.parametrize("preference", ["balanced", "support", "context"])
def test_osteosarc_absolute_floor_is_not_relaxed_by_any_preference(osteosarc_rna, preference):
    fragment, = osteosarc_fragments(
        osteosarc_rna, "bulk_star_t0", "DYNC1H1",
        protein_context_peptide_length=25, protein_sequence_preference=preference,
        min_variant_sequence_coverage=5,
    )
    assert len(fragment.sequence) == 16
    assert osteosarc_fragments(
        osteosarc_rna, "bulk_star_t0", "DYNC1H1",
        protein_context_peptide_length=25, protein_sequence_preference=preference,
        min_variant_sequence_coverage=1000,
    ) == []


@pytest.mark.isovar
def test_osteosarc_no_alt_reference_fallback_is_explicit_and_separate(osteosarc_rna):
    assert osteosarc_fragments(
        osteosarc_rna, "bulk_star_t0", "MAP2", protein_context_peptide_length=30,
    ) == []
    reference, = osteosarc_fragments(
        osteosarc_rna, "bulk_star_t0", "MAP2", protein_context_peptide_length=30,
        allow_reference_fallback=True, padding_around_mutation=14,
    )
    assert reference.annotations["sequence_source"] == "varcode_translation"
    assert reference.n_rna_alt_reads is None
    assert reference.n_rna_alt_fragments is None
    assert not any(key.startswith("isovar_") for key in reference.annotations)


@pytest.mark.isovar
@pytest.mark.parametrize("option", [
    {"protein_context_peptide_length": 0},
    {"protein_context_peptide_length": 1.5},
    {"protein_sequence_preference": "typo"},
    {"min_protein_sequence_support_fraction": 1.1},
    {"min_protein_sequence_support_fraction": float("nan")},
    {"min_variant_sequence_coverage": -1},
    {"min_variant_sequence_coverage": 1.5},
])
def test_invalid_rna_settings_fail_before_reading_alignments(option):
    with pytest.raises(ValueError):
        fragments_from_variants([], alignment_file=object(), **option)
