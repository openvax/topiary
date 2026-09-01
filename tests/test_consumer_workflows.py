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

import pandas as pd
import pytest

from topiary import (
    EvalContext,
    apply_filter,
    apply_sort,
    describe_read_evidence,
    evaluate_scores,
    fragment_from_isovar_result,
    fragments_from_dataframe,
    peptide_view,
    read_lens,
    read_pvacseq,
    resolve_default_methods,
    resolve_default_versions,
)
from topiary.ranking import parse

LENS = "tests/data/lens/sample_v1_4.tsv"
PVACSEQ = "tests/data/pvacseq/mhc_i_all_epitopes.tsv"


def _long(reader, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = reader(path)
        return result.to_long().df if result.metadata.form == "wide" else result.df


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
    "vaf > 0.1",
    "n_alt_reads > 5",
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


class _IsovarResult:
    top_protein_sequence = _ProteinSequence()
    variant = "chr7 g.140453136 A>T"
    num_total_fragments = 61
    num_alt_fragments = 30
    num_ref_fragments = 31


def test_one_consumer_function_reads_every_source():
    """The multi-source premise, exercised rather than described."""
    def support(fragment):
        if not fragment.is_usable_as_biology("n_alt_reads"):
            return None
        return fragment.is_approximate("n_alt_reads")

    sources = {
        "isovar": fragment_from_isovar_result(_IsovarResult()),
        "lens": fragments_from_dataframe(_long(read_lens, LENS))[0],
        "pvacseq": fragments_from_dataframe(_long(read_pvacseq, PVACSEQ))[0],
    }
    answers = {name: support(f) for name, f in sources.items()}

    assert answers["isovar"] is False        # counted
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
