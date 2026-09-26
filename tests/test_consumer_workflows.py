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
from io import StringIO
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import (
    CachedPredictor,
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
    mhc_dependence,
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
from .pvacseq_corpus_helpers import REPORTS as PVACSEQ_CORPUS, ROOT as PVACSEQ_CORPUS_ROOT
from .test_twin_conformance import DSL_MEASUREMENT_TWINS


def test_sv_interest_api_and_cli_retain_and_rank_the_same_nominations(tmp_path):
    import json
    from .test_sv_interest import catalogue, export
    from .test_twin_conformance import SV_INTEREST_REPORT_TWINS
    build, cli = SV_INTEREST_REPORT_TWINS
    cat, orfs = catalogue(), [export(), export("INTRON", priority=4)]
    expected = build(cat, orfs)
    catalogue_path = tmp_path / "catalogue.json"
    catalogue_path.write_text(json.dumps(cat))
    args = ["--catalogue", str(catalogue_path), "--output-prefix", str(tmp_path / "report")]
    for i, record in enumerate(orfs):
        path = tmp_path / (str(i) + ".json")
        path.write_text(json.dumps(record))
        args.extend(["--orf-export", str(path)])
    assert cli(args) == 0
    assert json.loads((tmp_path / "report.json").read_text()) == json.loads(json.dumps(expected))
    assert expected["candidates"][0]["event_id"] == "UTR"
    # Changing actual transcript evidence changes ranking through both doors.
    orfs[1]["candidates"][0]["start_evidence_summary"]["priority"] = 1
    (tmp_path / "1.json").write_text(json.dumps(orfs[1]))
    changed = build(cat, orfs)
    assert changed["candidates"][0]["event_id"] == "INTRON"
    assert cli(args) == 0
    assert json.loads((tmp_path / "report.json").read_text()) == json.loads(json.dumps(changed))


def _repeated_measurements(values, **columns):
    row = dict(source_sequence_name="observation", peptide="SIINFEKL", peptide_offset=0,
               allele="HLA-A*02:01", kind="pMHC_affinity",
               prediction_method_name="fixture", predictor_version="1")
    frame = pd.DataFrame([dict(row, value=value) for value in values])
    for column, value in columns.items():
        frame[column] = value
    return frame


@pytest.mark.parametrize("door,run", DSL_MEASUREMENT_TWINS)
@pytest.mark.parametrize("expression,column", [
    ("affinity.value", "value"), ("affinity.score", "score"),
    ("affinity.rank", "percentile_rank"), ("wt.affinity.value", "wt_value"),
    ("affinity.best_value", "value"), ("peptide_view(affinity.value)", "value"),
    ("percentile_rank", "percentile_rank"), ("column(review_score)", "review_score"),
])
@pytest.mark.parametrize("version", ["1", None, "nan"])
def test_conflicting_measurements_reject_every_consumer_in_both_orders(door, run, expression, column, version):
    frame = _repeated_measurements([50., 50.], predictor_version=version)
    frame[column] = [50., 60.]
    original = frame.copy(deep=True)
    for ordered in (frame, frame.iloc[::-1]):
        with pytest.raises(ValueError, match="Conflicting prediction measurements"):
            run(ordered, expression)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("door,run", DSL_MEASUREMENT_TWINS)
@pytest.mark.parametrize("expression,column", [
    ("affinity.value", "value"), ("percentile_rank", "percentile_rank"),
    ("column(review_score)", "review_score"),
])
@pytest.mark.parametrize("values,expected", [([50., 50.], 50.), ([None, 50.], 50.),
                                              ([None, None], None), (["50", 50.], 50.),
                                              ([50. + 1e-10, 50.], 50.)])
def test_equal_and_missing_measurements_compose_without_order_dependence(door, run, expression, column, values, expected):
    frame = pd.concat([_repeated_measurements(values),
                       _repeated_measurements([55.], peptide="GILGFVFTL")], ignore_index=True)
    frame[column] = frame.value
    original = frame.copy(deep=True)
    for ordered in (frame, frame.iloc[::-1]):
        answer = run(ordered, expression)
        if door == "score":
            target = answer.loc[ordered.peptide.eq("SIINFEKL")]
            assert target.isna().all() if expected is None else target.eq(expected).all()
            assert answer.loc[ordered.peptide.eq("GILGFVFTL")].eq(55.).all()
        elif "filter" in door:
            assert answer.peptide.tolist() == ([] if expected is None else ["SIINFEKL"] * 2)
        else:
            assert sorted(answer.peptide) == sorted(frame.peptide)
            if expected is not None:
                order = ["SIINFEKL", "SIINFEKL", "GILGFVFTL"]
                # Result sorting infers direction: affinity values ascend,
                # arbitrary numeric columns descend unless transformed.
                if door == "result_sort" and column != "value":
                    order = order[::-1]
                assert answer.peptide.tolist() == order
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("door,run", DSL_MEASUREMENT_TWINS)
@pytest.mark.parametrize("identity", ["prediction_run_name", "source_label"])
def test_named_measurement_observations_remain_independently_scoreable(door, run, identity, tmp_path):
    frame = _repeated_measurements([50., 60.], **{identity: ["first", "second"]})
    path = tmp_path / "runs.tsv"
    TopiaryResult(frame).to_tsv(path)
    from topiary import read_tsv
    restored = read_tsv(path).df
    for ordered in (restored, restored.iloc[::-1]):
        answer = run(ordered, "affinity.value")
        if door == "score":
            assert answer.tolist() == ordered.value.tolist()
        elif "filter" in door:
            assert answer[identity].tolist() == ["first"]
        else:
            assert answer[identity].tolist() == ["first", "second"]
        with pytest.raises(ValueError, match="Conflicting prediction measurements"):
            run(ordered, "affinity.value", group_keys=["peptide", "allele"])


@pytest.mark.parametrize("door,run", DSL_MEASUREMENT_TWINS)
def test_explicit_model_version_selection_precedes_measurement_conflict_check(door, run):
    frame = pd.concat([
        _repeated_measurements([50., 60.]),
        _repeated_measurements([40.], predictor_version="2"),
        _repeated_measurements([30.], prediction_method_name="other"),
    ], ignore_index=True)
    for ordered in (frame, frame.iloc[::-1]):
        for expression, expected in [("affinity['fixture', '2'].value", 40.),
                                     ("affinity['other'].value", 30.)]:
            answer = run(ordered, expression)
            if door == "score":
                assert answer.eq(expected).all()
            else:
                assert len(answer) == len(frame)


def test_conflict_guard_keeps_legitimate_kind_and_allele_axes():
    frame = pd.concat([
        _repeated_measurements([50., 50.]),
        _repeated_measurements([30.], allele="HLA-B*07:02"),
        _repeated_measurements([100.], kind="pMHC_stability"),
    ], ignore_index=True)
    for ordered in (frame, frame.iloc[::-1]):
        scores = evaluate_scores(ordered, parse("affinity.value"))
        assert scores.loc[ordered.allele.eq("HLA-A*02:01")].eq(50.).all()
        assert scores.loc[ordered.allele.eq("HLA-B*07:02")].eq(30.).all()
        assert evaluate_scores(ordered, parse("affinity.best_value")).eq(30.).all()


def test_peptide_level_measurement_conflicts_are_rejected_before_projection():
    frame = _repeated_measurements([50., 60.], kind="antigen_processing", allele="")
    for expression in ("processing.value", "peptide_view(processing.value)"):
        for ordered in (frame, frame.iloc[::-1]):
            with pytest.raises(ValueError, match="Conflicting prediction measurements"):
                evaluate_scores(ordered, parse(expression), alleles=["HLA-A*02:01"])

LENS = "tests/data/lens/sample_v1_4.tsv"
PVACSEQ = "tests/data/pvacseq/mhc_i_all_epitopes.tsv"
PVACSEQ_PRESENTATION = (
    "tests/data/pvacseq/mhc_i_all_epitopes_presentation.tsv"
)


@pytest.mark.parametrize("wide", [False, True])
def test_source_tables_combine_rank_and_export_without_predicting(monkeypatch, tmp_path, wide):
    from topiary import (
        combine_sources, melt_pvacseq_algorithms, protein_evidence_view,
        rank_candidates, read_tsv, rescore_candidates,
    )
    from .test_candidate_tables import Model

    direct = TopiaryPredictor(models=RandomBindingPredictor(
        alleles=["HLA-A*02:01"], default_peptide_lengths=[9],
    )).predict_from_named_sequences({"ORF1": "MAAASIINFEKLGGGSYFPEITHII"})
    orfs = pd.DataFrame(dict(
        event_id=["event-1", "event-1"],
        protein_sequence=["MAAASIINFEKLGGGSYFPEITHII", "MQQQSIINFEKL"],
        transcript_expression=[11., 22.], expression_unit=["TPM", "TPM"],
    ))

    def forbidden(*args, **kwargs):
        raise AssertionError("Table-only ranking must not execute a predictor")

    monkeypatch.setattr(TopiaryPredictor, "__init__", forbidden)
    sources = {"lens": read_lens(LENS), "pvacseq": melt_pvacseq_algorithms(read_pvacseq(PVACSEQ)),
               "direct": direct, "exacto_normalized": orfs}
    combined = combine_sources(sources, sample_name="fixture-patient")
    expression = "affinity['netmhcpan'].value"
    pooled = rank_candidates(combined, expression, ascending=True, duplicates="best")
    strata = rank_candidates(combined, expression, ascending=True, duplicates="best",
                             strata=["source_label", "candidate_mhc_class"])
    assert set(strata.loc[strata.candidate_score.notna(), "source_label"]) == {"lens", "pvacseq"}
    assert strata.loc[strata.source_label.eq("direct"), "ranking_status"].eq("missing_score").all()
    assert set(pooled.candidate_id) == set(combined.df.candidate_id.dropna())
    assert len(protein_evidence_view(combined)) == 2
    selected = combined.filter_by("n_rna_alt > 5")
    assert 0 < len(selected) < len(combined)
    path = tmp_path / "combined.tsv"
    (combined.to_wide() if wide else combined).to_tsv(path)
    restored = read_tsv(path)
    reranked = rank_candidates(restored, expression, ascending=True, duplicates="best")
    assert reranked.candidate_id.tolist() == pooled.candidate_id.tolist()
    np.testing.assert_allclose(reranked.candidate_score, pooled.candidate_score, equal_nan=True)
    assert restored.extra["combined_sources"] == combined.extra["combined_sources"]
    restored_orfs = restored.long_df.loc[restored.long_df.source_label.eq("exacto_normalized")]
    pd.testing.assert_frame_equal(restored_orfs[orfs.columns].reset_index(drop=True), orfs,
                                  check_dtype=False)
    assert restored_orfs.candidate_id.isna().all()
    assert restored_orfs.prediction_method_name.isna().all()

    # A raw multi-model measurement is ambiguous even with a source-selection
    # policy. Select a model through the DSL, as above, before ranking sources.
    for result in (combined, restored):
        for policy in ("error", "best", "worst"):
            with pytest.raises(ValueError, match="Conflicting prediction measurements"):
                rank_candidates(result, "percentile_rank", ascending=True, duplicates=policy)

    model = Model()
    enriched = rescore_candidates(restored, model, prefix="fresh", select="source_label == 'direct'",
                                  use_flanks=False)
    assert len(model.calls) == len(direct)
    original_again = rank_candidates(enriched, expression, ascending=True, duplicates="best")
    assert original_again.candidate_id.tolist() == pooled.candidate_id.tolist()
    np.testing.assert_allclose(original_again.candidate_score, pooled.candidate_score, equal_nan=True)
    feature = "fresh__testmodel__pMHC_affinity__value"
    fresh = rank_candidates(enriched, feature, ascending=True, duplicates="best")
    assert set(fresh.loc[fresh.candidate_score.notna(), "source_label"]) == {"direct"}
    assert set(fresh.candidate_id) == set(pooled.candidate_id)
    assert enriched.df.loc[~enriched.df.source_label.eq("direct"), feature].isna().all()


@pytest.mark.parametrize("wide", [False, True])
def test_terminal_flanks_survive_save_reload_and_contextual_rescoring(tmp_path, wide):
    from topiary import combine_sources, rank_candidates, rescore_candidates
    from .test_candidate_tables import Model, source
    from .test_twin_conformance import DELIMITED_IO_TWINS

    combined = combine_sources({
        "terminal": source(n_flank=["", "AAA"], c_flank=["GGG", ""]),
        "unknown": source(n_flank=[None, "AAA"], c_flank=["GGG", None]),
    }, sample_name="p")
    baseline_model = Model()
    baseline = rescore_candidates(combined, baseline_model, prefix="fresh",
                                  select="source_label == 'terminal'")
    feature = "fresh__testmodel__pMHC_affinity__value"
    expected = rank_candidates(baseline, feature, ascending=True, duplicates="best")
    assert expected.candidate_score.tolist() == [13., 800.]

    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("terminal-flanks." + suffix)
        method(combined.to_wide() if wide else combined, path)
        restored = reader(path)
        model = Model()
        enriched = rescore_candidates(restored, model, prefix="fresh",
                                      select="source_label == 'terminal'")
        assert model.calls == baseline_model.calls
        actual = rank_candidates(enriched, feature, ascending=True, duplicates="best")
        # Readers may add file provenance and wide-form WT convenience fields;
        # every original field and the resulting ranking must still agree.
        original_fields = actual[expected.columns]
        pd.testing.assert_frame_equal(original_fields.where(original_fields.notna(), np.nan),
                                      expected.where(expected.notna(), np.nan), check_dtype=False)
        for result in (combined, restored):
            with pytest.raises(ValueError, match="requires both flanks"):
                rescore_candidates(result, Model(), prefix="unknown",
                                   select="source_label == 'unknown'")


def test_table_rescoring_changes_only_the_explicit_dsl_policy(tmp_path):
    from topiary import combine_sources, rank_candidates, read_tsv, rescore_candidates
    from .test_candidate_tables import Model, source

    tables = {"mutation": source(source_type="variant:snv"),
              "fusion": source(source_type="sv:fusion", values=(70., 700.))}
    combined = combine_sources(tables, sample_name="fixture-patient")
    original = rank_candidates(combined, "affinity.value", ascending=True, duplicates="best")
    enriched = rescore_candidates(combined, Model(), prefix="new", select="source_label == 'fusion'")
    path = tmp_path / "enriched.tsv"
    enriched.to_tsv(path)
    restored = read_tsv(path)
    reranked = rank_candidates(restored, "affinity.value", ascending=True, duplicates="best")
    assert original.peptide.tolist() == reranked.peptide.tolist()
    assert original.candidate_score.tolist() == reranked.candidate_score.tolist()
    feature = "new__testmodel__pMHC_affinity__value"
    using_new = rank_candidates(restored, feature, ascending=True, duplicates="best")
    assert using_new.peptide.tolist() == original.peptide.tolist()[::-1]
    filtered = restored.filter_by(feature + " < 100")
    assert set(filtered.df.source_label) == {"fusion"}
    assert set(filtered.df.peptide) == {"GILGFVFTL"}


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("export_name", [None, "protein-v2", "protein-v2-labelled"])
@pytest.mark.parametrize("repeated", [False, True])
def test_isovar_comparison_import_keeps_default_candidates_and_calls_unchanged(
    tmp_path, wide, export_name, repeated,
):
    from copy import deepcopy
    from topiary import combine_sources, protein_evidence_view, rank_candidates, read_isovar_hypotheses
    from topiary import fragments_from_dataframe, rescore_candidates
    from .test_candidate_tables import Model, source
    from .test_isovar_hypotheses import hypothesis_export
    from .test_twin_conformance import DELIMITED_IO_TWINS

    export = hypothesis_export(export_name)
    if repeated:
        translations = export["events"][0]["protein_hypotheses"][0]["translations"]
        translations.extend(deepcopy(translations))
    export["events"][0]["filters"] = {"values": {"min_support": False}, "passes_all_filters": False}
    imported = read_isovar_hypotheses(export)
    with pytest.raises(ValueError, match="No sequence column found"):
        fragments_from_dataframe(imported.df)
    # Keep the report's source identity across file I/O; an absent source would
    # correctly acquire the output filename when the file is first read.
    baseline = combine_sources({"reported": source(source="reported-input")}, sample_name="tumor-1")
    comparison = combine_sources({"reported": source(source="reported-input"), "hypotheses": imported},
                                 sample_name="tumor-1")
    baseline_model = Model()
    enriched = rescore_candidates(baseline, baseline_model, prefix="fresh")
    expected = rank_candidates(enriched, "fresh__testmodel__pMHC_affinity__value", ascending=True)
    keys = ["candidate_id", "peptide", "candidate_score", "candidate_rank"]

    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("comparisons." + suffix)
        method(comparison.to_wide() if wide else comparison, path)
        restored = reader(path)
        for result in (comparison, restored):
            assert [(f.fragment_id, f.sequence, f.target_intervals)
                    for f in fragments_from_dataframe(result.long_df)] == [
                        (f.fragment_id, f.sequence, f.target_intervals)
                        for f in fragments_from_dataframe(baseline.long_df)]
            model = Model()
            rescored = rescore_candidates(result, model, prefix="fresh")
            assert model.calls == baseline_model.calls
            actual = rank_candidates(rescored, "fresh__testmodel__pMHC_affinity__value", ascending=True)
            pd.testing.assert_frame_equal(actual[keys], expected[keys])
            # Importing alternatives also leaves the original scoring policy intact.
            pd.testing.assert_frame_equal(
                rank_candidates(result, "affinity.value")[keys],
                rank_candidates(baseline, "affinity.value")[keys])
            alternatives = result.filter_by("isovar_rank > 1").df
            assert alternatives.protein_hypothesis_sequence.tolist() == ["MAQD", "AQG"]
            assert alternatives.candidate_id.isna().all()
            assert alternatives.passes_all_filters.eq(False).all()
            assert len(protein_evidence_view(result)) == 2
            assert result.extra["combined_sources"]["hypotheses"]["extra"]["isovar_hypotheses"] == export


@pytest.mark.isovar
@pytest.mark.parametrize("export_name", [None, "protein-v2", "protein-v2-labelled"])
def test_imported_isovar_rna_union_counts_shared_reads_once_after_reload(tmp_path, export_name):
    from isovar import union_rna_support
    from topiary import combine_sources, read_isovar_hypotheses, normalize_isovar_rna_support
    from .test_isovar_hypotheses import hypothesis_export
    from .test_twin_conformance import DELIMITED_IO_TWINS

    export = hypothesis_export(export_name)
    combined = combine_sources({"hypotheses": read_isovar_hypotheses(export)})
    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("rna-comparison." + suffix)
        method(combined.to_wide(), path)
        for result in (combined, reader(path)):
            provenance = result.extra["combined_sources"]["hypotheses"]["extra"]["isovar_hypotheses"]
            selected = result.filter_by("isovar_rank <= 2").df
            sets = [normalize_isovar_rna_support(provenance["evidence_sets"][key])
                    for key in selected.protein_evidence_set_id]
            # Synonymous rows repeat a protein's support, and another protein
            # shares one of those reads. Neither repetition adds new evidence.
            support = union_rna_support(sets)
            assert (support["reads"], support["fragments"]) == (5, 4)
            other_scope = dict(sets[0], evidence_scope=["tumor-1", "reprocessed reads"])
            with pytest.raises(ValueError, match="different evidence scopes"):
                union_rna_support([sets[0], other_scope])
            counts_only = provenance["events"][0]["protein_hypotheses"][0]["rna_support"]
            with pytest.raises(ValueError, match="without read identities"):
                union_rna_support([counts_only])


@pytest.mark.isovar
def test_real_repeated_isovar_translations_survive_import_and_reload(tmp_path):
    from collections import Counter
    import json
    from pathlib import Path

    import isovar
    import pysam
    from varcode import Variant
    from scripts.osteosarc_rna_overlay import POLICY
    from scripts.osteosarc_variant_audit import reference_genome
    from topiary import combine_sources, normalize_isovar_rna_support
    from .sid_data import sid_data_root, sid_read
    from .test_isovar_hypotheses import read_both
    from .test_twin_conformance import DELIMITED_IO_TWINS

    manifest = json.loads((Path(__file__).parent / "data/isovar_repeats/manifest.json").read_text())
    reads = sid_read("isovar_repeats/reads.bam")
    with pysam.AlignmentFile(reads) as bam:
        names = [read.query_name for read in bam]
    assert (len(names), len(set(names))) == (manifest["records"], manifest["templates"])
    genome = reference_genome(sid_data_root("osteosarc_all_variants"), tmp_path / "reference")
    variant = Variant("MT", 12994, "G", "A", ensembl=genome)
    with pysam.AlignmentFile(reads) as bam:
        upstream, = isovar.run_isovar(
            [variant], bam, read_collector=isovar.ReadCollector(**POLICY),
            protein_sequence_creator=isovar.ProteinSequenceCreator(
                protein_context_peptide_length=25, variant_sequence_assembly=True,
                max_protein_sequences_per_variant=0))
        export = isovar.export_protein_hypotheses(
            [upstream], sample_id="Sid-T2-UCLA-2025-01-06",
            source="osteosarc-0.7.0-repeat-fixture", alignment_header=bam.header)
    proteins = export["events"][0]["protein_hypotheses"]
    counts = [Counter(t["translation_id"] for t in p["translations"]) for p in proteins]
    assert sorted(n for c in counts for n in c.values() if n > 1) == [3, 3, 3]
    expected_rows = sum(len(c) for c in counts)
    raw_rows = sum(sum(c.values()) for c in counts)
    repeated = next(p for p, c in zip(proteins, counts) if max(c.values()) > 1)

    for imported in read_both(export, tmp_path):
        assert len(imported.df) == expected_rows == raw_rows - 6
        assert imported.df.hypothesis_id.nunique() == len(proteins)
        assert imported.extra["isovar_hypotheses"] == export
    combined = combine_sources({"rna": imported})
    for wide in (False, True):
        for suffix, writer, method, reader in DELIMITED_IO_TWINS:
            path = tmp_path / (f"real-rna-{wide}." + suffix)
            method(combined.to_wide() if wide else combined, path)
            restored = reader(path)
            assert len(restored.long_df) == expected_rows
            assert restored.long_df.candidate_id.isna().all()
            saved = restored.extra["combined_sources"]["rna"]["extra"]["isovar_hypotheses"]
            assert saved == export
            selected = restored.filter_by(f"hypothesis_id == '{repeated['hypothesis_id']}'").long_df
            assert len(selected) == 2
            assert selected.translation_reads.tolist() == [685, 685]
            assert selected.translation_fragments.tolist() == [419, 419]
            support = isovar.union_rna_support([
                normalize_isovar_rna_support(saved["evidence_sets"][key])
                for key in selected.translation_evidence_set_id])
            assert (support["reads"], support["fragments"]) == (685, 419)


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("export_name", ["protein-v2", "protein-v2-labelled"])
def test_isovar_label_completeness_controls_filtering_after_reload(tmp_path, wide, export_name):
    from topiary import combine_sources, read_isovar_hypotheses
    from .test_isovar_hypotheses import hypothesis_export
    from .test_twin_conformance import DELIMITED_IO_TWINS

    export = hypothesis_export(export_name)
    result = combine_sources({"rna": read_isovar_hypotheses(export)})
    for suffix, _, write, read in DELIMITED_IO_TWINS:
        path = tmp_path / ("labels." + suffix)
        write(result.to_wide() if wide else result, path)
        restored = read(path)
        assert restored.extra["combined_sources"]["rna"]["extra"]["isovar_hypotheses"] == export
        for observed in (result, restored):
            assert observed.df.candidate_id.isna().all()
            if export_name.endswith("labelled"):
                # Lower bounds stay visible, but only the short window's UMI
                # count is complete. Cells can be complete despite a missing UMI.
                assert observed.filter_by("protein_umis_complete").df.isovar_rank.tolist() == [3]
                assert observed.filter_by("protein_cells_complete").df.isovar_rank.tolist() == [1, 1, 3]
                assert observed.filter_by("protein_umis >= 2").df.isovar_rank.tolist() == [1, 1]
                assert observed.filter_by("protein_umis_complete & protein_umis >= 2").df.empty
            else:
                assert observed.df.protein_umis.isna().all()
                assert observed.df.protein_umis_complete.isna().all()


@pytest.mark.isovar
def test_current_isovar_sv_producers_compose_with_report_api_and_cli(tmp_path):
    import json

    from isovar import export_sv_rna_orfs, compare_sv_rna_predictions
    from topiary import build_sv_interest_report, write_sv_interest_report
    from .test_twin_conformance import SV_INTEREST_REPORT_TWINS

    root = Path(__file__).parent / "data" / "isovar_exports"
    orfs = export_sv_rna_orfs(json.loads((root / "orfs-input.json").read_text()))
    comparison = compare_sv_rna_predictions(*json.loads((root / "comparison-input.json").read_text()))
    assert orfs == json.loads((root / "orfs-v4.json").read_text())
    assert comparison == json.loads((root / "comparison-v3.json").read_text())
    catalogue = dict(targets={orfs["event_id"]: {}, comparison["event_id"]: {}})
    api, cli = SV_INTEREST_REPORT_TWINS
    expected = api(catalogue, [orfs], comparisons=[comparison])
    inputs = {"catalogue": catalogue, "orf-export": orfs, "comparison": comparison}
    args = []
    for flag, data in inputs.items():
        path = tmp_path / (flag + ".json")
        path.write_text(json.dumps(data))
        args.extend(["--" + flag, str(path)])
    assert cli(args + ["--output-prefix", str(tmp_path / "cli")]) == 0
    assert json.loads((tmp_path / "cli.json").read_text()) == json.loads(json.dumps(expected))
    assert {p["kind"] for p in expected["protein_hypotheses"]} == {"exploratory_orf", "annotated_frame"}
    assert all(p["translation_observed"] is False for p in expected["protein_hypotheses"])


def test_isovar_v2_zero_and_unknown_labels_and_missing_identity_survive_reload(tmp_path):
    from topiary import combine_sources, read_isovar_hypotheses
    from .test_isovar_hypotheses import hypothesis_export
    from .test_twin_conformance import DELIMITED_IO_TWINS

    export = hypothesis_export("protein-v2")
    empty = dict(reads=0, fragments=0, evidence_set_id="empty",
                 evidence_scope=export["evidence_scope"], read_ids=[], fragment_ids=[])
    export["evidence_sets"]["empty"] = empty
    proteins = export["events"][0]["protein_hypotheses"]
    proteins[0]["rna_support"] = dict(reads=0, fragments=0, evidence_set_id="empty", umis=0, cells=0,
        umis_complete=False, cells_complete=False, unlabeled_reads=0, unknown_library_reads=0, label_statuses={})
    proteins[1]["rna_support"]["evidence_set_id"] = None
    combined = combine_sources({"rna": read_isovar_hypotheses(export)})
    for suffix, _, write, read in DELIMITED_IO_TWINS:
        path = tmp_path / ("zero-and-unknown." + suffix)
        write(combined.to_wide(), path)
        for result in (combined, read(path)):
            first = result.df.iloc[0]
            assert first.protein_reads == first.protein_umis == 0
            assert not pd.isna(first.protein_umis_complete) and not first.protein_umis_complete
            assert first.protein_evidence_set_id == "empty"
            unknown = result.df.iloc[2]
            assert unknown.protein_reads == 2
            assert pd.isna(unknown.protein_umis) and pd.isna(unknown.protein_umis_complete)
            assert pd.isna(unknown.protein_evidence_set_id)
            assert result.extra["combined_sources"]["rna"]["extra"]["isovar_hypotheses"] == export


def test_isovar_comparison_import_does_not_change_default_fragment_selection(monkeypatch):
    from copy import deepcopy
    from topiary import fragment_from_isovar_result, fragments_from_variants, read_isovar_hypotheses
    from .test_isovar_hypotheses import hypothesis_export
    from .test_isovar_run import _Result, _fake

    result = _Result()
    module = _fake(monkeypatch, [result])
    completed = fragment_from_isovar_result(result).to_dict()
    assembled = [f.to_dict() for f in fragments_from_variants(["v"], alignment_file=object())]
    extra = deepcopy(result.top_protein_sequence)
    extra.amino_acids = "MQQQQQQQQQQQQQQQQQQQQ"
    extra.num_supporting_reads = extra.num_supporting_fragments = 1
    result.sorted_protein_sequences = [result.top_protein_sequence, extra]
    assert len(read_isovar_hypotheses(hypothesis_export())) == 4
    assert fragment_from_isovar_result(result).to_dict() == completed
    assert [f.to_dict() for f in fragments_from_variants(["v"], alignment_file=object())] == assembled
    for call in module.calls:
        creator = call["protein_sequence_creator"]
        assert creator.protein_sequence_preference == "balanced"
        assert creator.min_protein_sequence_support_fraction == 0.85


@pytest.mark.parametrize("form", ["native", "long", "wide"])
@pytest.mark.parametrize("sample", ["001", "NA"])
def test_isovar_comparison_roundtrip_preserves_literal_identity_and_sequence(tmp_path, form, sample):
    from topiary import combine_sources, protein_evidence_view, read_isovar_hypotheses
    from .test_isovar_hypotheses import hypothesis_export
    from .test_twin_conformance import DELIMITED_IO_TWINS

    export = hypothesis_export()
    export["sample_id"] = sample
    export["evidence_scope"][0] = sample
    for evidence in export["evidence_sets"].values():
        evidence["evidence_scope"][0] = sample
    partial = export["events"][0]["protein_hypotheses"][-1]
    partial["amino_acids"] = "NA"
    partial["mutation_interval"] = [0, 1]
    partial["translations"][0].update(
        nucleotide_sequence="AATGCT", translated_interval=[0, 6], variant_cdna_interval=[0, 1])
    imported = read_isovar_hypotheses(export)
    baseline = combine_sources({"hypotheses": imported})
    result = imported if form == "native" else baseline.to_wide() if form == "wide" else baseline
    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("literal-hypotheses." + suffix)
        method(result, path)
        restored = reader(path)
        if form == "native":
            restored = combine_sources({"hypotheses": restored})
        frame = restored.long_df
        assert frame.sample_name.tolist() == [sample] * 4
        assert frame.candidate_sample.tolist() == [sample] * 4
        assert frame.protein_hypothesis_sequence.tolist() == ["MAQG", "MAQG", "MAQD", "NA"]
        assert frame.candidate_id.isna().all()
        assert restored.filter_by('protein_hypothesis_sequence == "NA"').df.translation_id.tolist() == [
            partial["translations"][0]["translation_id"]]
        view = protein_evidence_view(restored)
        assert view.candidate_sample.tolist() == [sample, sample]
        assert view.protein_sequence.tolist() == ["MAQG", "MAQD"]
        saved = restored.extra["combined_sources"]["hypotheses"]["extra"]["isovar_hypotheses"]
        assert saved == export


def test_orf_abundance_can_enrich_matching_candidates_without_blending_alternative_orfs():
    from topiary import combine_sources, join_annotations, protein_evidence_view, rank_candidates
    from .test_candidate_tables import source

    proteins = ["MAAASIINFEKL", "MAAAGILGFVFTL"]
    candidates = source(protein_sequence=proteins, event_id=["event-1", "event-2"])
    rna = pd.DataFrame(dict(protein_sequence=proteins + ["MQQQSIINFEKL"],
                            event_id=["event-1", "event-2", "event-1"],
                            transcript_expression=[1., 1000., 9999.]))
    combined = combine_sources({"lens_normalized": candidates, "exacto_normalized": rna}, sample_name="p")
    assert len(protein_evidence_view(combined)) == 3
    keys = ["candidate_sample", "event_id", "protein_sequence_id"]
    annotations = combined.df.loc[combined.df.source_label.eq("exacto_normalized"),
                                  [*keys, "transcript_expression"]]
    enriched = join_annotations(combined, annotations, on=keys, prefix="exacto",
                                provenance={"source": "exacto_normalized", "unit": "TPM",
                                            "subject": "full ORF", "policy": "exact event and sequence"})
    first = rank_candidates(combined, "1 / affinity.value")
    second = rank_candidates(enriched, "exacto_transcript_expression / affinity.value")
    assert first.peptide.tolist() == ["SIINFEKL", "GILGFVFTL"]
    assert second.peptide.tolist() == first.peptide.tolist()[::-1]
    assert second.candidate_score.tolist() == [2., .02]
    assert enriched.filter_by("exacto_transcript_expression > 10").df.candidate_id.dropna().nunique() == 1


def test_nested_dataset_provenance_survives_filter_sort_and_file_io(tmp_path):
    from .test_twin_conformance import DELIMITED_IO_TWINS

    # Synthetic review scores exercise metadata transport, not binding claims.
    frame = pd.DataFrame({
        "peptide": ["SIINFEKL", "ELAGIGILT", "GILGFVFTL"],
        "kind": ["review"] * 3, "prediction_method_name": ["fixture"] * 3,
        "predictor_version": ["1"] * 3, "review_score": [10, 20, 30],
    })
    provenance = {"source": "a" * 64, "form": "original reads",
                  "filter_by": {"minimum_reads": 5}, "sort_by": ["sample", "locus"],
                  "model:fixture": {"version": "upstream"}}
    extra = {"dataset": provenance, "notes": "review\r\n#source=not-a-new-input",
             "literal": 'json:{"source":"still text"}', "label": " padded label "}
    result = TopiaryResult(frame, sources=["original-input"], extra=extra)
    selected = result.filter_by("review_score <= 20", group_keys=["peptide"]).sort_by(
        "review_score", group_keys=["peptide"])
    assert selected.df.peptide.tolist() == ["ELAGIGILT", "SIINFEKL"]
    assert len(result.filter_by("review_score <= 10", group_keys=["peptide"])) == 1
    restored_frames = []
    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("selected." + suffix)
        method(selected, path)
        restored = reader(path, tag="selected")
        assert restored.extra == extra
        assert restored.sources == ["original-input", "selected"]
        assert restored.models == {"fixture": "1"}
        assert restored.filter_by_str == "review_score <= 20"
        assert restored.sort_by_str == "review_score"
        assert restored.df.peptide.tolist() == selected.df.peptide.tolist()
        assert len(restored.filter_by("review_score <= 10", group_keys=["peptide"])) == 1
        restored_frames.append(restored.df)
    pd.testing.assert_frame_equal(*restored_frames)


@pytest.mark.parametrize("report", PVACSEQ_CORPUS, ids=lambda r: r["file"])
def test_real_pvacseq_rna_overlay_preserves_history_and_changes_filtering(report, tmp_path):
    from topiary import melt_pvacseq_algorithms, read_tsv
    from .osteosarc_overlay_helpers import add_rna

    path = PVACSEQ_CORPUS_ROOT / report["file"]
    raw = pd.read_csv(path, sep="\t")
    original = melt_pvacseq_algorithms(read_pvacseq(path, tag=report["file"]))
    enriched = add_rna(original, raw)
    pd.testing.assert_frame_equal(enriched.df[original.columns], original.df)
    saved = tmp_path / "enriched.tsv"
    enriched.to_tsv(saved)
    restored = read_tsv(saved, tag=report["file"])
    pd.testing.assert_frame_equal(restored.df.isna(), enriched.df.isna())
    pd.testing.assert_frame_equal(
        restored.df.astype(object).where(restored.df.notna(), None),
        enriched.df.astype(object).where(enriched.df.notna(), None),
        check_dtype=False, rtol=1e-12, atol=1e-12)
    assert restored.extra == enriched.extra
    if not len(enriched):
        return  # the historical filtered files must remain empty
    selections = []
    for result in (enriched, restored):
        assert result.df.gene_expression.isna().all()
        assert result.df.pvacseq_tumor_rna_depth.isna().all()
        assert result.df.rna_t2_gene_tpm.notna().all()
        permissive = result.filter_by("rna_t2_alt_reads >= 1")
        stricter = result.filter_by("(rna_t2_alt_reads >= 3) & (rna_t2_gene_tpm >= 1)")
        low = set(permissive.df.rna_t2_allele_key)
        high = set(stricter.df.rna_t2_allele_key)
        assert high < low
        selections.append((low, high))
    assert selections[0] == selections[1]


@pytest.mark.parametrize("report", PVACSEQ_CORPUS, ids=lambda r: r["file"])
def test_real_pvacseq_selected_columns_survive_import_melt_save_reload(report, tmp_path):
    from topiary import melt_pvacseq_algorithms, read_tsv
    from .pvacseq_corpus_helpers import assert_selected_columns

    path = PVACSEQ_CORPUS_ROOT / report["file"]
    raw = pd.read_csv(path, sep="\t", na_values=["X"])
    imported = melt_pvacseq_algorithms(read_pvacseq(path, tag=report["file"]))
    saved = tmp_path / "topiary.tsv"
    imported.to_tsv(saved)
    restored = read_tsv(saved, tag=report["file"])
    assert imported.sources == restored.sources
    assert imported.extra == restored.extra
    assert len(imported) == len(restored)
    for result in (imported, restored):
        assert_selected_columns(raw, result.df, report["category"] == "aggregated")
        # A release was not recorded in these source reports. Do not invent it.
        assert result.df.predictor_version.isna().all()


def test_real_pvacseq_saved_cache_method_and_threshold_change_selection(tmp_path):
    from topiary import Affinity, melt_pvacseq_algorithms, read_tsv

    path = PVACSEQ_CORPUS_ROOT / "2025.04.27.mhc_class_i.all_epitopes.tsv"
    result = melt_pvacseq_algorithms(read_pvacseq(path))
    saved = tmp_path / "predictions.tsv"
    result.to_tsv(saved)
    restored = read_tsv(saved)

    def selected(frame, method, threshold):
        kept = apply_filter(frame, Affinity[method].value <= threshold)
        return set(kept[["peptide", "allele"]].itertuples(index=False, name=None))

    for frame in (result.df, restored.df):
        stringent = selected(frame, "mhcflurry", 100)
        permissive = selected(frame, "mhcflurry", 500)
        other_model = selected(frame, "netmhcpan", 500)
        assert stringent and stringent < permissive
        assert other_model and other_model != permissive
    for method in ("mhcflurry", "netmhcpan"):
        for threshold in (100, 500):
            assert selected(result.df, method, threshold) == selected(restored.df, method, threshold)


@pytest.mark.isovar
@pytest.mark.parametrize("assembly", [False, True])
@pytest.mark.parametrize("secondary", [False, True])
@pytest.mark.parametrize("gene,sample", [
    ("GLIS3", "T1-ONT-dedup"), ("GLIS3", "T2-ONT-dedup"),
    ("GLIS3", "T1-short"), ("GLIS3", "T2-short"),
    ("KTN1", "T1-ONT-dedup"), ("KTN1", "T2-ONT-dedup"),
    ("KTN1", "T1-short"), ("KTN1", "T2-short"),
])
def test_missing_indel_rna_to_prediction_is_source_grounded(
    additional_indel_rna, gene, sample, secondary, assembly, tmp_path,
):
    import pysam
    from isovar import ProteinSequenceCreator, ReadCollector
    from .osteosarc_helpers import assert_expected_fragment
    from .test_twin_conformance import ISOVAR_HANDOFF_TWINS

    variants, bams, expected = additional_indel_rna
    observed = []
    for door in ISOVAR_HANDOFF_TWINS:
        with pysam.AlignmentFile(bams[gene + "." + sample]) as bam:
            fragments = door(
                [variants[gene]], bam,
                read_collector=ReadCollector(use_secondary_alignments=secondary),
                protein_sequence_creator=ProteinSequenceCreator(
                    variant_sequence_assembly=assembly,
                    protein_context_peptide_length=25))
        observed.append([f.to_dict() for f in fragments])
        if (gene, sample) not in {("GLIS3", "T1-short"), ("KTN1", "T2-ONT-dedup"),
                                  ("KTN1", "T2-short")}:
            assert fragments == []
            continue
        fragment, = fragments
        assert len(fragment.sequence) == {
            ("GLIS3", "T1-short"): 46,
            ("KTN1", "T2-ONT-dedup"): 29,
            ("KTN1", "T2-short"): 18,
        }[gene, sample]
        assert_expected_fragment(fragment, expected[gene])
        path = tmp_path / "fragment.json"
        write_fragments(fragments, path)
        restored, = read_fragments(path)
        assert restored.to_dict() == fragment.to_dict()
        frame = _isovar_prediction_frame(restored)
        assert not frame.empty and frame.contains_mutant_residues.all()
        for row in frame.itertuples():
            assert row.peptide == fragment.sequence[row.peptide_offset:row.peptide_offset + 9]
            start, end = fragment.target_intervals[0]
            assert row.peptide_offset < end and row.peptide_offset + 9 > start
    # Only the convenience API knows the creator settings. A bare IsovarResult
    # cannot retrospectively supply them; compare all biology/evidence and
    # explicitly check the legitimate provenance-only difference.
    provenance = {
        "isovar_version", "isovar_creator", "isovar_protein_sequence_length",
        "isovar_protein_context_peptide_length", "isovar_protein_sequence_preference",
        "isovar_min_protein_sequence_support_fraction", "isovar_min_variant_sequence_coverage",
        "isovar_variant_sequence_assembly", "isovar_min_assembly_overlap_size",
        "isovar_min_transcript_prefix_length", "isovar_max_transcript_mismatches",
        "isovar_count_mismatches_after_variant", "isovar_max_protein_sequences_per_variant",
    }
    for direct, adapted in zip(observed[0], observed[1]):
        assert set(direct["annotations"]) - set(adapted["annotations"]) == provenance
        assert direct["annotations"]["isovar_variant_sequence_assembly"] == assembly
        assert direct["annotations"]["isovar_min_variant_sequence_coverage"] == 2
        direct["annotations"] = {k: v for k, v in direct["annotations"].items() if k not in provenance}
    assert observed[0] == observed[1]


@pytest.mark.isovar
@pytest.mark.parametrize("gene,sample,secondary", [
    ("GLIS3", "T1-short", True), ("GLIS3", "T1-short", False),
    ("KTN1", "T2-ONT-dedup", True), ("KTN1", "T2-ONT-dedup", False),
    ("KTN1", "T2-short", True), ("KTN1", "T2-short", False),
])
def test_missing_indel_default_filter_is_not_confused_with_reconstruction(
    additional_indel_rna, gene, sample, secondary,
):
    """Since Isovar 1.18.1 no reconstructed indel here fails a default filter.

    Rejection by default filters on real reads is covered by the nine
    ``filtered`` loci of the all-variant corpus, and synthetically by
    ``test_isovar_run``; this pins that these reconstructions pass them.
    """
    import pysam
    from isovar import ProteinSequenceCreator, ReadCollector, run_isovar

    variants, bams, _ = additional_indel_rna
    options = dict(read_collector=ReadCollector(use_secondary_alignments=secondary),
                   protein_sequence_creator=ProteinSequenceCreator(protein_context_peptide_length=25))
    with pysam.AlignmentFile(bams[gene + "." + sample]) as bam:
        upstream, = run_isovar([variants[gene]], bam, **options)
    assert upstream.has_mutant_protein_sequence_from_rna
    assert {name for name, passed in upstream.filter_values.items() if not passed} == set()
    with pysam.AlignmentFile(bams[gene + "." + sample]) as bam:
        fragments = fragments_from_variants([variants[gene]], bam, **options)
    assert len(fragments) == 1


@pytest.fixture
def cli_output_request(tmp_path, monkeypatch):
    """A real input/cache pair with two peptides and three kinds of evidence."""
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    # Subprocesses must run this checkout even if an older wheel is installed.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    peptides = tmp_path / "peptides.csv"
    peptides.write_text("name,peptide\nfirst,SIINFEKL\nsecond,GILGFVFTL\n")
    rows = []
    for peptide, affinity in (("SIINFEKL", 50.0), ("GILGFVFTL", 500.0)):
        for kind in ("pMHC_affinity", "pMHC_presentation", "antigen_processing"):
            rows.append(dict(
                peptide=peptide, peptide_length=len(peptide), kind=kind,
                allele="" if kind == "antigen_processing" else "HLA-A*02:01",
                value=affinity if kind == "pMHC_affinity" else None,
                value_unit="nM" if kind == "pMHC_affinity" else None,
                affinity=affinity if kind == "pMHC_affinity" else None,
                score=0.5, percentile_rank=1.0,
                prediction_method_name="synthetic", predictor_version="1",
            ))
    cache = tmp_path / "cache.csv"
    pd.DataFrame(rows).to_csv(cache, index=False)
    return [
        "--peptide-csv", str(peptides), "--mhc-cache-file", str(cache),
        "--mhc-cache-format", "topiary_output",
    ]


def test_cli_default_preview_shows_the_filtered_and_ranked_results(
    cli_output_request, capsys,
):
    from topiary.cli.script import main

    main(cli_output_request + ["--sort-by", "ba", "--sort-direction", "desc"])
    captured = capsys.readouterr()
    assert "SIINFEKL" in captured.out and "GILGFVFTL" in captured.out
    assert captured.out.index("GILGFVFTL") < captured.out.index("SIINFEKL")
    assert "antigen_processing" in captured.out
    assert "6 prediction rows (2 unique peptides, 1 named allele)" in captured.err

    main(cli_output_request + ["--filter-by", "ba <= 100"])
    captured = capsys.readouterr()
    assert "SIINFEKL" in captured.out and "GILGFVFTL" not in captured.out
    assert "3 prediction rows (1 unique peptide, 1 named allele)" in captured.err


@pytest.mark.parametrize("separator", [",", "\t"])
def test_cli_csv_stdout_and_file_carry_the_same_selected_results(
    cli_output_request, tmp_path, capsys, separator,
):
    from topiary.cli.script import main

    options = cli_output_request + [
        "--filter-by", "ba <= 100", "--output-csv-sep", separator,
        "--subset-output-columns", "peptide", "kind", "value",
        "--rename-output-column", "value", "measurement", "--print-columns",
    ]
    output = tmp_path / "results.csv"
    main(options + ["--output-csv", str(output)])
    capsys.readouterr()
    main(options + ["--output-csv", "-"])
    captured = capsys.readouterr()
    assert captured.out == output.read_text()
    frame = pd.read_csv(StringIO(captured.out), sep=separator, index_col="#")
    assert list(frame.columns) == ["peptide", "kind", "measurement"]
    assert list(frame.peptide) == ["SIINFEKL"] * 3
    assert "Columns:" in captured.err
    assert "3 prediction rows (1 unique peptide, 1 named allele)" in captured.err
    assert not (tmp_path / "-").exists()


def test_cli_distinguishes_empty_input_from_filtered_results(
    cli_output_request, capsys, caplog,
):
    from pathlib import Path
    from topiary.cli.script import main

    main(cli_output_request + ["--filter-by", "ba < 1"])
    captured = capsys.readouterr()
    assert "No prediction rows" in captured.err
    assert "No peptides found" not in caplog.text

    Path(cli_output_request[1]).write_text("name,peptide\n")
    assert main(cli_output_request) == 0
    assert "No peptides found in the input" in caplog.text
    assert "No prediction rows" in capsys.readouterr().err


def test_cli_csv_can_be_consumed_by_another_process(cli_output_request):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "from topiary.cli.script import main; main()",
         *cli_output_request, "--output-csv", "-"],
        capture_output=True, text=True, check=True,
    )
    frame = pd.read_csv(StringIO(result.stdout), index_col="#")
    assert len(frame) == 6
    assert set(frame.peptide) == {"SIINFEKL", "GILGFVFTL"}
    assert "6 prediction rows" in result.stderr


def test_predictor_progress_does_not_pollute_csv(cli_output_request, monkeypatch, capsys):
    from topiary.cli import script

    predict = script.predict_epitopes_from_args

    def noisy_predict(args):
        print("Predictor progress")
        return predict(args)

    monkeypatch.setattr(script, "predict_epitopes_from_args", noisy_predict)
    script.main(cli_output_request + ["--output-csv", "-"])
    captured = capsys.readouterr()
    assert len(pd.read_csv(StringIO(captured.out))) == 6
    assert "Predictor progress" in captured.err


def test_cli_csv_consumer_can_close_the_pipe_early(cli_output_request):
    from pathlib import Path
    import subprocess
    import sys

    Path(cli_output_request[1]).write_text(
        "name,peptide\n" + "".join(f"sample_{i},SIINFEKL\n" for i in range(1500))
    )
    html = Path(cli_output_request[1]).with_name("results.html")
    with subprocess.Popen(
        [sys.executable, "-c", "from topiary.cli.script import main; main()",
         *cli_output_request, "--output-csv", "-", "--output-html", str(html)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as process:
        assert "peptide" in process.stdout.readline()
        process.stdout.close()
        process.stdout = None
        _, stderr = process.communicate(timeout=30)
        assert process.returncode == 0
        assert "BrokenPipeError" not in stderr
        assert "Traceback" not in stderr
    # Closing one destination must not discard the other requested output.
    assert "sample_1499" in html.read_text()
    assert html.read_text().count("<tr>") == 4500


@pytest.fixture(params=["file", "directory"])
def cached_length_request(tmp_path, request):
    """Distinct stored measurements make selection errors visible (#329)."""
    proteins = tmp_path / "proteins.fasta"
    proteins.write_text(">protein\nSIINFEKLA\n")
    rows = []
    for peptide, value in (("SIINFEKL", 11.0), ("IINFEKLA", 22.0), ("SIINFEKLA", 33.0)):
        for kind, allele, measurement in (
            ("pMHC_affinity", "HLA-A*02:01", value),
            ("antigen_processing", "", value / 100),
        ):
            rows.append(dict(
                peptide=peptide, peptide_length=len(peptide), kind=kind,
                allele=allele, value=measurement, score=measurement / 100,
                affinity=value if allele else None, percentile_rank=value / 10,
                prediction_method_name="synthetic", predictor_version="1",
            ))
    frame = pd.DataFrame(rows)
    if request.param == "file":
        cache = tmp_path / "cache.csv"
        frame.to_csv(cache, index=False)
        cache_args = ["--mhc-cache-file", str(cache)]
    else:
        cache = tmp_path / "cache"
        cache.mkdir()
        for length, shard in frame.groupby("peptide_length"):
            shard.to_csv(cache / f"{length}.csv", index=False)
        cache_args = ["--mhc-cache-directory", str(cache)]
    return ["--fasta", str(proteins), *cache_args], frame


@pytest.mark.parametrize("length_args,peptides", [
    ([], {"SIINFEKL", "IINFEKLA", "SIINFEKLA"}),
    (["--mhc-peptide-lengths", "8"], {"SIINFEKL", "IINFEKLA"}),
    (["--mhc-peptide-lengths", "9"], {"SIINFEKLA"}),
    (["--mhc-epitope-lengths", "9"], {"SIINFEKLA"}),
    (["--mhc-peptide-lengths", "9", "--mhc-epitope-lengths", "8"], {"SIINFEKLA"}),
])
def test_cached_scan_selects_lengths_without_changing_stored_values(
    cached_length_request, length_args, peptides, tmp_path, capsys,
):
    from topiary.cli.script import main

    args, stored = cached_length_request
    files_before = {path: path.read_bytes() for path in tmp_path.rglob("*.csv")}
    assert main(args + length_args + ["--mhc-alleles", "HLA-A*02:01", "--output-csv", "-"]) == 0
    result = pd.read_csv(StringIO(capsys.readouterr().out), index_col="#")
    assert set(result.peptide) == peptides
    assert set(result.kind) == {"pMHC_affinity", "antigen_processing"}
    columns = ["peptide", "kind", "value", "affinity", "percentile_rank", "score"]
    expected = stored[stored.peptide.isin(peptides)]
    pd.testing.assert_frame_equal(
        result[columns].sort_values(["peptide", "kind"]).reset_index(drop=True),
        expected[columns].sort_values(["peptide", "kind"]).reset_index(drop=True),
    )
    assert all(path.read_bytes() == contents for path, contents in files_before.items())


def test_cached_scan_does_not_look_up_unrequested_windows(tmp_path):
    from topiary.cli.args import arg_parser, predict_epitopes_from_args

    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">protein\nSIINFEKLA\n")
    cache = tmp_path / "cache.csv"
    pd.DataFrame([
        dict(peptide=peptide, peptide_length=len(peptide), allele="HLA-A*02:01",
             kind="pMHC_affinity", value=value, affinity=value,
             prediction_method_name="synthetic", predictor_version="1")
        for peptide, value in (("AAAAAAAA", 11.0), ("SIINFEKLA", 33.0))
    ]).to_csv(cache, index=False)
    args = ["--fasta", str(fasta), "--mhc-cache-file", str(cache)]
    # The stored 8-mer advertises that length, but covers neither 8-mer in
    # this protein. Restriction must happen before lookup, not on its output.
    result = predict_epitopes_from_args(arg_parser.parse_args(
        args + ["--mhc-peptide-lengths", "9"],
    ))
    assert list(result.peptide) == ["SIINFEKLA"]
    assert list(result.value) == [33.0]


@pytest.mark.parametrize("flag", ["--mhc-peptide-lengths", "--mhc-epitope-lengths"])
def test_cached_scan_refuses_missing_requested_lengths(cached_length_request, flag):
    from topiary.cli.args import arg_parser, predict_epitopes_from_args

    args, _ = cached_length_request
    with pytest.raises(ValueError, match="lengths"):
        predict_epitopes_from_args(arg_parser.parse_args(args + [flag, "8,9,10"]))


def test_cached_peptides_and_filters_preserve_stored_measurements(cached_length_request, tmp_path):
    from topiary.cli.args import arg_parser, predict_epitopes_from_args

    args, _ = cached_length_request
    peptides = tmp_path / "peptides.csv"
    peptides.write_text("peptide\nSIINFEKL\nSIINFEKLA\n")
    args = ["--peptide-csv", str(peptides), *args[2:], "--mhc-peptide-lengths", "9"]
    before = predict_epitopes_from_args(arg_parser.parse_args(args))
    after = predict_epitopes_from_args(arg_parser.parse_args(args + ["--filter-by", "ba < 20"]))
    assert set(before.peptide) == {"SIINFEKL", "SIINFEKLA"}  # explicit inputs, not windows
    assert set(after.peptide) == {"SIINFEKL"}
    assert after.loc[after.kind == "pMHC_affinity", "value"].tolist() == [11.0]


@pytest.mark.parametrize("lengths", [[8], [9], [8, 9]])
def test_live_and_cached_predictors_agree_on_scan_vs_explicit_lengths(lengths):
    from mhctools import RandomBindingPredictor
    from tests.test_twin_conformance import CACHE_LENGTH_TWINS

    proteins = {"protein": "SIINFEKLA"}
    explicit = ["SIINFEKL", "SIINFEKLA"]
    live = RandomBindingPredictor(alleles=["HLA-A*02:01"], default_peptide_lengths=[8, 9])
    original = TopiaryPredictor(models=live).predict_from_named_sequences(proteins)
    original["predictor_version"] = "test"
    cache = CachedPredictor.from_dataframe(original)
    live.default_peptide_lengths = lengths
    cache.default_peptide_lengths = lengths
    for mode, live_call, cache_call in CACHE_LENGTH_TWINS:
        inputs = proteins if mode == "protein windows" else explicit
        outputs = [call(model, inputs) for call, model in ((live_call, live), (cache_call, cache))]
        assert sorted(outputs[0].peptide) == sorted(outputs[1].peptide)
        if mode == "protein windows":
            assert set(outputs[1].peptide.str.len()) == set(lengths)
        else:
            assert set(outputs[1].peptide) == set(explicit)


@pytest.fixture(params=["topiary_output", "directory", "tsv"])
def cache_loading_request(cli_output_request, tmp_path, request):
    directory = tmp_path / "archive"
    directory.mkdir()
    sep = "\t" if request.param == "tsv" else ","
    path = directory / ("predictions.tsv" if sep == "\t" else "predictions.csv")
    frame = pd.read_csv(cli_output_request[3])
    frame["predictor_version"] = "bundle-1"
    if request.param == "directory":
        args = ["--mhc-cache-directory", str(directory)]
    else:
        args = ["--mhc-cache-file", str(path), "--mhc-cache-format", request.param]
    return path, sep, frame, [*cli_output_request[:2], *args]


@pytest.mark.parametrize("missing", ["column", "null", "mixed", "stated"])
def test_cache_loading_fills_missing_provenance_without_changing_measurements(
    cache_loading_request, missing, capsys,
):
    from topiary.cli.script import main

    path, sep, stored, args = cache_loading_request
    frame = stored.copy()
    identity = ["prediction_method_name", "predictor_version"]
    if missing == "column":
        frame = frame.drop(columns=identity)
    elif missing == "null":
        frame[identity] = None
    elif missing == "mixed":
        frame.loc[0, identity] = " "
        frame.loc[1, identity] = None
    frame.to_csv(path, sep=sep, index=False)
    contents = path.read_bytes()
    assert main(args + [
        "--mhc-cache-predictor-name", "synthetic",
        "--mhc-cache-predictor-version", "bundle-1", "--output-csv", "-",
    ]) == 0
    result = pd.read_csv(StringIO(capsys.readouterr().out), index_col="#")
    assert set(result.prediction_method_name) == {"synthetic"}
    assert set(result.predictor_version) == {"bundle-1"}
    columns = ["peptide", "kind", "affinity", "score", "percentile_rank"]
    pd.testing.assert_frame_equal(
        result[columns].sort_values(["peptide", "kind"]).reset_index(drop=True),
        stored[columns].sort_values(["peptide", "kind"]).reset_index(drop=True),
    )
    assert path.read_bytes() == contents


@pytest.mark.parametrize("column,flag", [
    ("prediction_method_name", "--mhc-cache-predictor-name"),
    ("predictor_version", "--mhc-cache-predictor-version"),
])
def test_cache_loading_refuses_to_relabel_stated_provenance(
    cache_loading_request, column, flag, capsys,
):
    from topiary.cli.script import main

    path, sep, frame, args = cache_loading_request
    frame.to_csv(path, sep=sep, index=False)
    with pytest.raises(SystemExit) as error:
        main(args + [flag, "different"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert column in message and "conflict" in message
    assert "different" in message


def test_tsv_kind_mapping_and_missing_kind_guidance(tmp_path, capsys):
    from topiary.cli.script import main

    peptides = tmp_path / "peptides.csv"
    peptides.write_text("peptide\nSIINFEKL\n")
    path = tmp_path / "measurements.tsv"
    frame = pd.DataFrame([dict(
        peptide="SIINFEKL", allele="HLA-A*02:01", IC50=12.5, assay="pMHC_affinity",
    )])
    frame.to_csv(path, sep="\t", index=False)
    args = [
        "--peptide-csv", str(peptides), "--mhc-cache-file", str(path),
        "--mhc-cache-format", "tsv", "--mhc-cache-predictor-name", "laboratory",
        "--mhc-cache-predictor-version", "experiment-1",
        "--mhc-cache-tsv-column", "affinity=IC50", "--output-csv", "-",
    ]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    message = capsys.readouterr().err.split("topiary: error:", 1)[-1]
    assert str(path) in message
    assert "kind" in message and "--mhc-cache-tsv-column kind=" in message
    assert "predictor_name / predictor_version" not in message
    assert main(args + ["--mhc-cache-tsv-column", "kind=assay"]) == 0
    result = pd.read_csv(StringIO(capsys.readouterr().out), index_col="#")
    assert result.kind.tolist() == ["pMHC_affinity"]
    assert result.affinity.tolist() == [12.5]


@pytest.mark.parametrize("explicit_format", [False, True])
def test_cache_missing_file_reports_the_path_not_format_detection(
    cli_output_request, tmp_path, capsys, explicit_format,
):
    from topiary.cli.script import main

    missing = tmp_path / "absent-cache.csv"
    args = [*cli_output_request[:2], "--mhc-cache-file", str(missing)]
    if explicit_format:
        args += ["--mhc-cache-format", "topiary_output"]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    message = capsys.readouterr().err.split("topiary: error:", 1)[-1]
    assert str(missing) in message and "No such file" in message
    assert "auto-detect" not in message


@pytest.mark.parametrize("mode", ["rb", "r"])
def test_cache_unreadable_file_reports_permission_error(
    cli_output_request, monkeypatch, capsys, mode,
):
    import builtins
    import errno
    from topiary.cli.script import main

    original_open = builtins.open
    path = cli_output_request[3]

    def denied(file, open_mode="r", *args, **kwargs):
        if str(file) == path and open_mode == mode:
            raise PermissionError(errno.EACCES, "Permission denied", path)
        return original_open(file, open_mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", denied)
    with pytest.raises(SystemExit) as error:
        main(cli_output_request[:4])  # exercise automatic detection
    assert error.value.code == 2
    message = capsys.readouterr().err.split("topiary: error:", 1)[-1]
    assert path in message and "Permission denied" in message
    assert "auto-detect" not in message


def test_bad_cache_directory_shard_names_the_file_and_expected_format(
    cli_output_request, tmp_path, capsys,
):
    from topiary.cli.script import main

    directory = tmp_path / "bad-shards"
    directory.mkdir()
    bad = directory / "netmhc.out"
    bad.write_text("# NetMHCpan version 4.1\none,two,three\nfour,five,six,seven\n")
    with pytest.raises(SystemExit) as error:
        main([*cli_output_request[:2], "--mhc-cache-directory", str(directory)])
    assert error.value.code == 2
    message = capsys.readouterr().err.split("topiary: error:", 1)[-1]
    assert str(bad) in message and "topiary_output" in message


@pytest.mark.parametrize("predictor_name", ["mhcflurry", "mhcflurry-affinity"])
@pytest.mark.parametrize("input_kind", ["peptide", "protein"])
def test_live_mhcflurry_cli_output_replays_with_identical_provenance_and_values(
    predictor_name, input_kind, tmp_path, monkeypatch, capsys,
):
    """Only the numerical backend is stubbed; wrappers, CLI and files are real."""
    import sys
    import types
    from mhctools import mhcflurry as wrapper
    from topiary.cli.script import main

    def affinity(peptides, alleles, **kwargs):
        return pd.DataFrame(dict(
            peptide=peptides, allele=alleles,
            prediction=[11.0 if p == "SIINFEKLA" else 22.0 for p in peptides],
            prediction_percentile=[1.0] * len(peptides),
        ))

    def presentation(peptides, **kwargs):
        return pd.DataFrame(dict(
            peptide=peptides, peptide_num=range(len(peptides)),
            best_allele=["HLA-A*02:01"] * len(peptides),
            presentation_score=[0.7] * len(peptides),
            presentation_percentile=[2.0] * len(peptides),
            processing_score=[0.3] * len(peptides),
        ))

    aff = types.SimpleNamespace(
        supported_alleles=["HLA-A*02:01"], predict_to_dataframe=affinity,
    )
    backend = types.SimpleNamespace(
        supported_alleles=aff.supported_alleles, affinity_predictor=aff,
        predict=presentation,
    )
    package = types.ModuleType("mhcflurry")
    package.__version__ = "2.2.1"
    package.Class1PresentationPredictor = types.SimpleNamespace(load=lambda path: backend)
    package.Class1AffinityPredictor = types.SimpleNamespace(load=lambda path: aff)
    downloads = types.ModuleType("mhcflurry.downloads")
    downloads.get_current_release = lambda: "2.2.0"
    downloads.get_path = lambda *args, **kwargs: str(tmp_path / "official")
    downloads.get_default_class1_models_dir = lambda **kwargs: downloads.get_path()
    downloads.get_default_class1_presentation_models_dir = lambda **kwargs: downloads.get_path()
    package.downloads = downloads
    monkeypatch.setitem(sys.modules, "mhcflurry", package)
    monkeypatch.setitem(sys.modules, "mhcflurry.downloads", downloads)
    monkeypatch.setattr(wrapper, "_model_cache", {})

    source = tmp_path / "input.fasta"
    source.write_text(">first\nSIINFEKLA\n>second\nGILGFVFTL\n")
    inputs = ["--peptide-fasta" if input_kind == "peptide" else "--fasta", str(source)]
    live = tmp_path / "live.csv"
    assert main([
        *inputs, "--mhc-predictor", predictor_name, "--mhc-alleles", "HLA-A*02:01",
        "--mhc-peptide-lengths", "9", "--output-csv", str(live),
    ]) == 0
    capsys.readouterr()
    stored = pd.read_csv(live, index_col="#")
    assert set(stored.predictor_version) == {"2.2.1+release-2.2.0"}
    assert set(stored.loc[stored.kind == "pMHC_affinity", "value"]) == {11.0, 22.0}

    # Replay must not consult whichever MHCflurry happens to be installed now.
    monkeypatch.setitem(sys.modules, "mhcflurry", None)
    monkeypatch.setitem(sys.modules, "mhcflurry.downloads", None)
    assert main([*inputs, "--mhc-cache-file", str(live), "--output-csv", "-"]) == 0
    replayed = pd.read_csv(StringIO(capsys.readouterr().out), index_col="#")
    columns = [
        "peptide", "kind", "allele", "value", "score", "percentile_rank",
        "prediction_method_name", "predictor_version", "n_flank", "c_flank",
    ]
    if "allele_set" in stored:
        columns.append("allele_set")
    else:
        assert replayed.allele_set.isna().all()  # no genotype context on affinity-only rows
    order = ["peptide", "kind", "allele"]
    pd.testing.assert_frame_equal(
        stored[columns].sort_values(order).reset_index(drop=True),
        replayed[columns].sort_values(order).reset_index(drop=True),
    )


@pytest.mark.parametrize("scenario, allowed", [
    ("absent", True), ("published", False), ("timeout", False),
    ("http_error", False), ("bad_metadata", False), ("wrong_version", False),
])
def test_release_api_and_cli_enforce_the_same_publish_policy(monkeypatch, scenario, allowed):
    from urllib.error import HTTPError
    from topiary import release
    from tests.test_release import response
    from tests.test_twin_conformance import RELEASE_PREFLIGHT_DOORS

    def lookup(request, timeout):
        if scenario in {"absent", "http_error"}:
            status = 404 if scenario == "absent" else 503
            raise HTTPError(request.full_url, status, "simulated", {}, None)
        if scenario == "timeout":
            raise TimeoutError("lookup timed out")
        if scenario == "bad_metadata":
            return response({})
        version = "5.53.0" if scenario == "published" else "5.53.1"
        return response({"info": {"name": "topiary", "version": version}})

    monkeypatch.setattr(release, "urlopen", lookup)
    for door in RELEASE_PREFLIGHT_DOORS:
        assert door("topiary", "5.53.0") is allowed


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
    (np.str_("value\0"), "value\0"),
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


def test_string_enum_variant_predictions_survive_construction_and_serialization(tmp_path):
    from dataclasses import replace
    from tests.test_twin_conformance import (
        FRAGMENT_CONSTRUCTION_DOORS, FRAGMENT_SERIALIZATION_DOORS,
    )

    class Origin(str, Enum):
        SNV = "variant:snv"

    expected = {"SIINFEKLA", "IINFEKLAA"}
    for source_type in ("variant:snv", Origin.SNV):
        for _, construct in FRAGMENT_CONSTRUCTION_DOORS:
            fragment = construct(dict(source_type=source_type, target_intervals=[(4, 5)]))
            # The construction doors use a nine-residue sequence. Add one
            # residue to exercise two distinct mutation-overlapping windows.
            fragment = replace(fragment, sequence="SIINFEKLAA")
            candidates = [fragment] + [
                roundtrip(fragment, tmp_path / f"{name}.tsv")
                for name, roundtrip in FRAGMENT_SERIALIZATION_DOORS
            ]
            for restored in candidates:
                frame = _isovar_prediction_frame(restored)
                assert set(frame.peptide) == expected
                assert frame.contains_mutant_residues.all()
                assert restored.source_type == "variant:snv"
                assert type(restored.source_type) is str


@pytest.mark.parametrize("dtype", [np.datetime64, np.timedelta64])
@pytest.mark.parametrize("unit", ["s", "ns"])
def test_fragment_temporal_values_reach_custom_json_encoder_with_units(dtype, unit):
    from tests.test_twin_conformance import FRAGMENT_CONSTRUCTION_DOORS

    value = dtype(1, unit)
    for _, construct in FRAGMENT_CONSTRUCTION_DOORS:
        fragment = construct({"annotations": {"duration_or_date": value}})
        seen = []

        def encode_temporal(obj):
            assert type(obj) is dtype
            assert obj.dtype == value.dtype
            assert obj == value
            seen.append(obj)
            return {"dtype": obj.dtype.str, "ticks": int(obj.astype("int64"))}

        restored = ProteinFragment.from_json(fragment.to_json(default=encode_temporal))
        encoded = restored.annotations["duration_or_date"]
        decoded = np.array(encoded["ticks"], dtype=encoded["dtype"])[()]
        assert len(seen) == 1
        assert decoded.dtype == value.dtype
        assert decoded == value


def test_nested_dataclass_annotations_survive_all_serialization_doors(tmp_path):
    from dataclasses import dataclass
    from tests.test_twin_conformance import FRAGMENT_SERIALIZATION_DOORS

    @dataclass
    class Settings:
        enabled: object
        label: object

    settings = Settings(np.bool_(True), np.str_("label\0"))
    fragment = ProteinFragment(fragment_id="nested", annotations={"settings": [settings]})
    assert fragment.annotations["settings"][0] is settings
    for name, roundtrip in FRAGMENT_SERIALIZATION_DOORS:
        restored = roundtrip(fragment, tmp_path / f"{name}.tsv")
        assert restored.annotations == {"settings": [{"enabled": True, "label": "label\0"}]}
        assert restored.annotations["settings"][0]["enabled"] is True
    assert type(settings.enabled) is np.bool_
    assert settings.label == "label\0"


@pytest.fixture
def half_life_models(tmp_path, monkeypatch):
    """Exercise actual wrapper APIs without unsafe sidecars or model downloads.

    Values are synthetic transport fixtures, not independently validated
    biological predictions (mhctools #310/#311).
    """
    from mhctools import PeptiVerse, PlifePred2

    # Build each snapshot from the artifact lists mhctools declares,
    # rather than a hardcoded copy of them. Those lists grew three times
    # in a week -- the exact log-scale model directory, then a pinned
    # ESM2 snapshot, then a third Pfeature resource -- and each growth
    # broke this fixture with a FileNotFoundError naming one file at a
    # time. Reading them keeps the fixture right for whatever version is
    # installed; `or` fallbacks cover versions predating a given list.
    from mhctools import peptiverse as _peptiverse
    from mhctools import plifepred2 as _plifepred2

    def _touch_all(root, relative_paths):
        for relative in relative_paths:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

    peptiverse = tmp_path / "peptiverse"
    _touch_all(peptiverse, getattr(
        _peptiverse, "_PEPTIVERSE_ARTIFACTS", {"inference.py": None},
    ))
    # The model directory is checked for existence separately from its
    # contents, and is empty on versions that declare no artifacts.
    (peptiverse / getattr(
        _peptiverse, "_MODEL_DIRECTORY", "training_classifiers/half_life",
    )).mkdir(parents=True, exist_ok=True)
    esm = peptiverse / "esm2_t33_650M_UR50D"
    _touch_all(esm, getattr(_peptiverse, "_ESM2_ARTIFACTS", {}))
    _touch_all(esm, getattr(_peptiverse, "_ESM2_WEIGHT_ARTIFACTS", {}))
    esm.mkdir(parents=True, exist_ok=True)

    plifepred2 = tmp_path / "plifepred2"
    _touch_all(plifepred2, getattr(
        _plifepred2, "_PLIFEPRED2_ARTIFACTS",
        {"models/plifepred2_natural_model.sav": None},
    ))
    pfeature = tmp_path / "pfeature"
    _touch_all(pfeature, getattr(
        _plifepred2, "_PFEATURE_ARTIFACTS",
        {"pfeature_comp.py": None, "Data/Schneider-Wrede.csv": None,
         "Data/Grantham.csv": None},
    ))
    (pfeature / "Data").mkdir(parents=True, exist_ok=True)

    for key, path in (("PEPTIVERSE_HOME", peptiverse),
                      ("PEPTIVERSE_ESM_HOME", esm),
                      ("PLIFEPRED2_HOME", plifepred2),
                      ("PFEATURE_HOME", pfeature)):
        monkeypatch.setenv(key, str(path))

    # mhctools >=3.44.5 refuses to construct a backend whose artifacts
    # do not match its pinned checksums, which empty files never will.
    # Neutralized here rather than by passing allow_unverified_assets,
    # because topiary also builds these models from a class or a bare
    # name -- the doors
    # test_whole_peptide_model_construction_and_scanning_boundary
    # exercises -- and cannot pass per-model flags on those paths, nor
    # should it hardcode one. Under test is topiary's transport of
    # whole-peptide predictions, not mhctools' asset verification.
    try:
        from mhctools.optional_backend import BackendInventory
    except ImportError:
        pass
    else:
        monkeypatch.setattr(
            BackendInventory, "require_usable",
            lambda self, allow_unverified=False: self,
        )

    def synthetic_output(model, peptides):
        # Opposite preference between matrices proves that they stay distinct.
        serum = model._predictor_name() == "peptiverse"
        values = [2.0 if p.startswith("S") else 8.0 for p in peptides]
        if not serum:
            values = [12.0 if p.startswith("S") else 1.0 for p in peptides]
        # Native outputs are supplied too, for the reviewed PlifePred2 API
        # that withholds an unverified hours conversion by default.
        return pd.DataFrame({"hours": values, "log10_seconds": values})

    for cls in (PeptiVerse, PlifePred2):
        monkeypatch.setattr(cls, "_run_sidecar", synthetic_output)
    return PeptiVerse(), PlifePred2()


@pytest.mark.parametrize("model_index", [0, 1])
def test_whole_peptide_wrappers_survive_prediction_cache_io_and_ranking(
    half_life_models, model_index, tmp_path,
):
    from tests.test_twin_conformance import WHOLE_PEPTIDE_PREDICTION_DOORS
    from topiary import read_tsv, to_tsv

    model = half_life_models[model_index]
    peptides = ["SIINFEKLGGALQ", "KLGGALQAKKYKY", "SIINFEKLGGALQ"]
    kind, = model.supported_kinds
    columns = ["source_sequence_name", "peptide", "allele", "kind", "value", "score",
               "percentile_rank", "prediction_method_name", "predictor_version"]
    frames = [
        door(model, peptides).astype({"value": float, "score": float, "percentile_rank": float})
        .sort_values("source_sequence_name").reset_index(drop=True)
        for door in WHOLE_PEPTIDE_PREDICTION_DOORS
    ]
    pd.testing.assert_frame_equal(frames[0][columns], frames[1][columns], check_dtype=False)
    native = frames[0]
    assert len(native) == 3
    assert native.allele.eq("").all()
    assert native.affinity.isna().all()
    assert native.kind.eq(kind).all()
    assert mhc_dependence(kind) == "none"
    if model_index == 0:
        assert native.value.tolist() == [2.0, 8.0, 2.0]
    else:
        # The default PlifePred2 output has no independently verified unit.
        assert native.value.isna().all()

    path = tmp_path / "half-life.tsv"
    to_tsv(native, path)
    restored = read_tsv(path).df
    cache = CachedPredictor(restored)
    assert cache.kind_support()[kind]["mhc_dependence"] == "none"
    replayed = TopiaryPredictor(models=cache).predict_from_named_peptides(
        {str(i): p for i, p in enumerate(peptides)},
    ).sort_values("source_sequence_name").reset_index(drop=True)
    # TSV uses NaN where an in-memory wrapper may use None.
    replayed = replayed.astype({"value": float, "score": float, "percentile_rank": float})
    pd.testing.assert_frame_equal(native[columns], replayed[columns], check_dtype=False)

    # A whole-peptide observation stays one row, even alongside two alleles.
    # Projection is an explicit query operation, not duplicated measurements.
    from mhctools import RandomBindingPredictor
    affinity = TopiaryPredictor(models=RandomBindingPredictor(
        alleles=["HLA-A*02:01", "HLA-B*07:02"], default_peptide_lengths=[12],
    )).predict_from_named_peptides({"0": peptides[0], "1": peptides[1]})
    mixed = pd.concat([native[native.source_sequence_name != "2"], affinity], ignore_index=True)
    for frame in (mixed, TopiaryResult(mixed).to_wide().to_long().df):
        assert len(frame[frame.kind == kind]) == 2
        scores = evaluate_scores(frame, parse(f"peptide_view({kind}.score)"))
        assert scores.notna().all()
        threshold = (native.score.min() + native.score.max()) / 2
        selected = apply_filter(frame, parse(f"peptide_view({kind}.score) > {threshold}"))
        assert set(selected.peptide) == {peptides[1 if model_index == 0 else 0]}


@pytest.mark.parametrize("model_index", [0, 1])
@pytest.mark.parametrize("door", ["instance", "class", "name"])
def test_whole_peptide_model_construction_and_scanning_boundary(half_life_models, model_index, door):
    model = half_life_models[model_index]
    supplied = {"instance": model, "class": type(model), "name": model._predictor_name()}[door]
    predictor = TopiaryPredictor(models=supplied)
    assert predictor.padding_around_mutation == 0
    assert predictor.predict_from_named_peptides({}).empty
    frame = predictor.predict_from_named_peptides({"vaccine": "SIINFEKLGGALQ"})
    assert len(frame) == 1
    assert frame.peptide.tolist() == ["SIINFEKLGGALQ"]
    with pytest.raises(ValueError, match="predict_from_named_peptides"):
        predictor.predict_from_named_sequences({"protein": "SIINFEKLGGALQ"})
    with pytest.raises(ValueError, match="predict_from_named_peptides"):
        predictor.predict_from_fragments([
            ProteinFragment(fragment_id="protein", sequence="SIINFEKLGGALQ"),
        ])


def test_whole_peptide_model_rejects_mixed_scanning_before_running_any_model(half_life_models):
    from mhctools import RandomBindingPredictor

    class MustNotRun(RandomBindingPredictor):
        def predict_proteins_dataframe(self, sequences):
            pytest.fail("Protein scanning started before model compatibility was checked")

    predictor = TopiaryPredictor(models=[
        MustNotRun(alleles=["HLA-A*02:01"], default_peptide_lengths=[9]), half_life_models[0],
    ])
    assert predictor.padding_around_mutation == 8
    with pytest.raises(ValueError, match="predict_from_named_peptides"):
        predictor.predict_from_named_sequences({"protein": "SIINFEKLGGALQ"})


def test_whole_peptide_models_keep_domain_errors_and_unknown_units(half_life_models):
    from mhctools import Prediction
    from topiary import from_predictions

    with pytest.raises(ValueError, match="12-100 residues"):
        TopiaryPredictor(models=half_life_models[1]).predict_from_named_peptides({"short": "SIINFEKL"})
    for kind in ("serum_half_life", "blood_half_life", "pMHC_stability"):
        frame = from_predictions([Prediction(
            kind=kind, peptide="SIINFEKLGGALQ", score=2.5, value=None,
            predictor_name="unknown-unit", predictor_version="1",
        )])
        assert frame.value.isna().all()
        assert frame.score.eq(2.5).all()


def test_stability_stdout_retains_hours_through_cache_io_and_selection(tmp_path):
    from pathlib import Path
    from tests.test_twin_conformance import STABILITY_PREDICTION_DOORS
    from topiary import read_tsv, to_tsv

    path = Path(__file__).parent / "data" / "netmhc_fixtures" / "netmhcstabpan_SLLQHLIGL_A0201.out"
    columns = ["peptide", "allele", "kind", "value", "score", "affinity", "percentile_rank"]
    cached, direct = [door(path) for door in STABILITY_PREDICTION_DOORS]
    for frame in (cached, direct):
        assert frame.value.tolist() == [7.04]  # Explicit Thalf(h) in the original output.
        assert frame.affinity.isna().all()
        assert frame.percentile_rank.tolist() == [0.4]
    pd.testing.assert_frame_equal(cached[columns], direct[columns], check_dtype=False)

    to_tsv(cached, tmp_path / "stability.tsv")
    restored = read_tsv(tmp_path / "stability.tsv").df
    replayed = TopiaryPredictor(models=CachedPredictor(restored)).predict_from_named_peptides(
        {"candidate": "SLLQHLIGL"},
    )
    assert len(apply_filter(replayed, parse("stability.value > 7"))) == 1
    assert apply_filter(replayed, parse("stability.value > 8")).empty
    assert replayed.affinity.isna().all()


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
def test_osteosarc_h1_read_identity_preserves_default_floor_and_diagnostic(
    osteosarc_rna, tmp_path,
):
    """A real deletion is not proof of two independent supporting templates.

    Isovar 1.17 no longer treats competing primary/secondary placements as
    definitive deletion support. One unambiguous paired template remains.
    An explicitly labelled coverage-one diagnostic keeps the stricter
    independent translation checks alive without relaxing the default.
    """
    import pysam
    from tests.osteosarc_helpers import assert_expected_fragment
    from tests.test_twin_conformance import ISOVAR_RECONSTRUCTION_TWINS

    variants, bams, expected = osteosarc_rna
    observed = []
    for door in (ISOVAR_RECONSTRUCTION_TWINS.left, ISOVAR_RECONSTRUCTION_TWINS.right):
        options = dict(protein_context_peptide_length=25)
        with pysam.AlignmentFile(bams["bulk_star_t0"]) as bam:
            assert door([variants["H1-2"]], bam, **options) == []
        with pysam.AlignmentFile(bams["bulk_star_t0"]) as bam:
            fragment, = door(
                [variants["H1-2"]], bam,
                min_variant_sequence_coverage=1, **options,
            )
        assert_expected_fragment(fragment, expected["H1-2"])
        assert fragment.n_rna_alt_reads == 2
        assert fragment.n_rna_alt_fragments == 1
        assert fragment.n_rna_alt_reads_supporting_protein_sequence == 2
        assert fragment.n_rna_alt_fragments_supporting_protein_sequence == 1
        assert fragment.annotations["isovar_min_variant_sequence_coverage"] == 1
        path = tmp_path / "h1-diagnostic.tsv"
        write_fragments([fragment], path)
        restored, = read_fragments(path)
        assert restored.to_dict() == fragment.to_dict()
        frame = _isovar_prediction_frame(restored)
        assert not frame.empty
        assert frame.isovar_min_variant_sequence_coverage.eq(1).all()
        assert frame.n_rna_alt_fragments_supporting_protein_sequence.eq(1).all()
        observed.append(fragment.to_dict())
    assert observed[0] == observed[1]


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


def test_same_peptide_different_rna_observations_survive_prediction_and_filtering(tmp_path):
    """Identical peptide does not mean identical sample/policy evidence (#345)."""
    from mhctools import RandomBindingPredictor
    from topiary import (
        ProteinFragment, TopiaryPredictor, TopiaryResult, fragments_for_sample,
        make_fragment_id, read_tsv,
    )
    from topiary.evidence import RNA_ALIGNMENT

    sequence = "SIINFEKLL"
    fragments = [
        fragment
        for sample, count in (("T1", 9), ("T2", 2))
        for fragment in fragments_for_sample([ProteinFragment(
            fragment_id=make_fragment_id("same-allele", sequence, variant="same-allele"),
            sequence=sequence, n_rna_alt_fragments=count,
            annotations={"rna_evidence_method": RNA_ALIGNMENT})], sample)
    ]
    predictor = TopiaryPredictor(models=RandomBindingPredictor(
        alleles=["HLA-A*01:01"], default_peptide_lengths=[9]), only_novel_epitopes=False)
    frame = predictor.predict_from_fragments(fragments)
    assert len(frame) == 2 and set(frame.peptide) == {sequence}
    assert dict(zip(frame["sample_name"], frame.n_rna_alt_fragments)) == {"T1": 9, "T2": 2}
    path = tmp_path / "observations.tsv"
    TopiaryResult(frame).to_tsv(path)
    restored = read_tsv(path)
    assert set(restored.filter_by("n_rna_alt_fragments >= 2").df["sample_name"]) == {"T1", "T2"}
    assert set(restored.filter_by("n_rna_alt_fragments >= 3").df["sample_name"]) == {"T1"}
    # One candidate ID, two observations: the default keys pool them.
    pooled = aggregate_evidence_across_samples(restored.df)
    assert pooled[["n_samples", "n_rna_alt"]].values.tolist() == [[2, 11]]


def test_a_cache_never_lends_its_sample_to_an_unlabelled_fragment():
    """A cache built from labelled output carries that run's sample_name."""
    from mhctools import RandomBindingPredictor
    from topiary import CachedPredictor, ProteinFragment, TopiaryPredictor, fragments_for_sample

    fragment = ProteinFragment(fragment_id="candidate", sequence="SIINFEKLL")
    labelled = TopiaryPredictor(models=RandomBindingPredictor(
        alleles=["HLA-A*01:01"], default_peptide_lengths=[9]), only_novel_epitopes=False
    ).predict_from_fragments(fragments_for_sample([fragment], "T1"))
    cached = TopiaryPredictor(models=CachedPredictor.from_dataframe(
        labelled, predictor_version="random"), only_novel_epitopes=False)

    assert set(cached.predict_from_fragments([fragment])["sample_name"]) == {""}
    assert set(cached.predict_from_fragments(fragments_for_sample([fragment], "T2"))["sample_name"]) == {"T2"}


def test_a_pvacseq_peptide_on_two_transcripts_keeps_both_expressions(tmp_path):
    """Without an Index column the variant alone names the row; the transcript
    must still separate transcript-level expression."""
    raw = pd.read_csv(PVACSEQ, sep="\t", dtype=str, keep_default_na=False).iloc[[0]]
    other = raw.assign(**{"Transcript": "ENST00000999999.1", "Transcript Expression": "123.4"})
    path = tmp_path / "two-transcripts.all_epitopes.tsv"
    pd.concat([raw, other]).drop(columns=["Index"], errors="ignore").to_csv(path, sep="\t", index=False)

    fragments = fragments_from_dataframe(_long(read_pvacseq, str(path)))

    assert {f.transcript_id: f.transcript_expression for f in fragments} == {
        raw["Transcript"].iloc[0]: float(raw["Transcript Expression"].iloc[0]),
        "ENST00000999999.1": 123.4,
    }
    assert len({f.fragment_id for f in fragments}) == 2


DATA = Path(__file__).parent / "data"
READER_FRAMES = [
    *((read_lens, str(path)) for path in sorted(DATA.glob("lens/*.tsv"))),
    *((read_pvacseq, str(path)) for path in sorted(DATA.glob("pvacseq/*.tsv"))),
]
assert len(READER_FRAMES) >= 9, "reader fixtures not found"


@pytest.mark.parametrize("reader,path", READER_FRAMES, ids=lambda value: getattr(value, "__name__", value))
def test_every_reader_frame_reaches_predictions_with_its_own_evidence(reader, path):
    """The documented reader path, run whole, on every fixture.

    LENS reports RNA evidence per peptide, and several peptides share one
    context. Each reported peptide's evidence must reach its own prediction
    rows, not be merged with, or overwritten by, a neighbour's.
    """
    from mhctools import RandomBindingPredictor

    frame = _long(reader, path)
    fragments = fragments_from_dataframe(frame)
    # The shortest reported peptide is 8 aa; every fragment yields a window.
    predictor = TopiaryPredictor(models=RandomBindingPredictor(
        alleles=["HLA-A*02:01"], default_peptide_lengths=[8]), only_novel_epitopes=False)
    predictions = predictor.predict_from_fragments(fragments)

    assert not predictions.empty
    assert set(predictions.fragment_id) == {f.fragment_id for f in fragments}
    for fragment in fragments:
        reported = fragment.annotations.get("reported_peptide")
        if reported is None:
            continue
        rows = frame[(frame["peptide"] == reported) & (frame["pep_context"] == fragment.sequence)]
        stated = set(rows["n_rna_overlapping"].dropna())
        own = fragment.n_rna_overlapping_reads
        assert own in stated if stated else own is None
        attached = predictions.loc[predictions.fragment_id.eq(fragment.fragment_id)
                                   & predictions.sample_name.eq(fragment.sample_name or "")]
        if own is not None:
            assert set(attached["n_rna_overlapping_reads"]) == {own}


@pytest.mark.osteosarc
@pytest.mark.isovar
def test_shared_osteosarc_reads_reconstruct_and_rank_identically(tmp_path):
    """Original export and shared cache compose through real RNA reconstruction."""
    import json
    import shutil
    import isovar
    from osteosarc import Cache, Variant, Variants, digest, extract_reads
    from topiary import (
        CachedPredictor, CachedPredictorCoverageError, TopiaryPredictor,
        TopiaryResult, describe_isovar_result, fragment_from_isovar_result,
        read_fragments, read_tsv, write_fragments,
    )
    from scripts.osteosarc_variant_audit import check_mutation_windows, reference_genome
    from tests.osteosarc_all_helpers import reference_models, validate_rna_protein
    from tests.test_osteosarc_shared import MANIFEST, ROOT, SOURCE, exported_manifest
    from tests.test_twin_conformance import OSTEOSARC_SOURCE_TWINS

    translated = json.loads((ROOT / "translation-v1.json").read_text())
    prediction = json.loads((ROOT / "prediction-contract-v1.json").read_text())
    assert translated["parent_dataset_sha256"] == digest(SOURCE / "manifest.json")
    assert prediction["parent_translation_sha256"] == digest(ROOT / "translation-v1.json")
    reference = ROOT.parent / "osteosarc_all_variants"
    assert translated["reference_manifest_sha256"] == digest(reference / "reference/manifest.json")
    genome = reference_genome(reference, tmp_path / "reference-index")
    models = reference_models(reference / "reference")
    case = next(c for c in MANIFEST["cases"] if c["case_id"] == translated["case_id"])
    record = case["variant"]
    native_source = Variants([Variant(
        id=record["variant_id"], gene=record["gene"], assembly=record["assembly"],
        alleles=((record["chrom"], record["pos"], record["ref"], record["alt"]),), status="ready",
    )], source={"dataset_sha256": translated["parent_dataset_sha256"], "case": case})
    native = native_source.to_varcode(genome=genome, assembly="GRCh38")
    assert native.metadata[native[0]]["source"]["dataset_sha256"] == translated["parent_dataset_sha256"]
    manifest, files = exported_manifest()
    export = tmp_path / "export"
    export.mkdir()
    (export / "manifest.json").write_text(json.dumps(manifest))
    cache = Cache(tmp_path / "shared", offline=True)
    for asset in manifest["assets"]:
        shutil.copyfile(files[asset["filename"]], export / asset["filename"])
        cache.import_file(files[asset["filename"]], asset["url"],
                          sha256=asset["sha256"], size=asset["size_bytes"])
    answers = []
    for door in OSTEOSARC_SOURCE_TWINS:
        paths = door(manifest, export, cache)
        # Content-addressed BAM and BAI paths have different hashes, so always
        # pass the index explicitly; adjacency is not a shared-cache contract.
        subset = extract_reads(paths[case["bam"]], native_source.regions(padding=100),
                               index=paths[case["bam"] + ".bai"], cache=cache)
        with subset.open() as bam:
            upstream, = isovar.run_isovar(
                native, bam, read_collector=isovar.ReadCollector(**translated["read_collector"]),
                protein_sequence_creator=isovar.ProteinSequenceCreator(**translated["protein_sequence_creator"]))
        assert describe_isovar_result(upstream) == translated["description"]
        validate_rna_protein(upstream, models)
        fragment = fragment_from_isovar_result(upstream)
        assert fragment.to_dict() == translated["fragment"]
        # Original RNA resolves the adjacent substitution too: this is not
        # the protein obtained from applying only the nominated A>G to cDNA.
        assert fragment.sequence == "NKLSKQMVDVSENYQSTLPK"
        path = tmp_path / (door.__name__ + ".tsv")
        write_fragments([fragment], path)
        restored, = read_fragments(path)
        predictor = CachedPredictor(pd.DataFrame(prediction["rows"]))
        frame = TopiaryPredictor(models=predictor, only_novel_epitopes=True).predict_from_fragments([restored])
        check_mutation_windows(frame, [restored])
        assert sorted(frame.peptide_offset) == list(range(2, 11))
        assert set(frame.prediction_method_name) == {"synthetic_fixture_affinity"}
        assert set(frame.predictor_version) == {"fixture-v1"}
        output = tmp_path / "predictions.tsv"
        provenance = {"source": translated["parent_dataset_sha256"],
                      "translation": prediction["parent_translation_sha256"],
                      "prediction": digest(ROOT / "prediction-contract-v1.json")}
        TopiaryResult(frame, extra={"osteosarc_shared": provenance}).to_tsv(output)
        result = read_tsv(output)
        assert result.extra["osteosarc_shared"] == provenance
        assert len(result.filter_by("n_rna_alt_reads >= 9")) == 9
        assert len(result.filter_by("n_rna_alt_reads >= 10")) == 0
        ranked = result.sort_by("affinity.value")
        assert list(ranked.df.peptide_offset) == list(range(2, 11))
        assert len(result.filter_by("affinity.value <= 50")) == 3
        assert len(result.filter_by("affinity.value <= 80")) == 6
        answers.append(ranked.df.reset_index(drop=True))
        # A source hash does not grant coverage for another model version or
        # missing peptide/context. Cache misses must not turn into zero scores.
        incomplete = CachedPredictor(pd.DataFrame(prediction["rows"][1:]))
        with pytest.raises(CachedPredictorCoverageError):
            TopiaryPredictor(models=incomplete).predict_from_fragments([restored])
    pd.testing.assert_frame_equal(*answers)


@pytest.mark.isovar
def test_malformed_osteosarc_catalogue_row_survives_audit_and_report(tmp_path):
    """Native parsing diagnostics survive, and the next real RNA case runs."""
    import json
    import shutil
    from scripts.osteosarc_variant_audit import variant_inventory, audit, report
    from scripts.osteosarc_rna_overlay import digest, write_json
    from .sid_data import sid_data_root, sid_read
    from .test_osteosarc_audit_inventory import HEADER, INDEX

    data = sid_data_root("osteosarc_all_variants")
    good = next(v for v in json.loads((data / "checked-inventory.json").read_text())["variants"]
                if v["variant_id"] == "DYNC1H1-chr14-101980529")
    html = INDEX.replace("</table>", "") + INDEX.replace("<table>", "").replace(
        "example", good["variant_id"]).replace("GENE", good["gene"]).replace(
        "chr1:10", f"{good['chrom']}:{good['pos']}")
    rows = ("example\tGENE\tchr1\tbad\tA\tC\n"
            f"{good['variant_id']}\t{good['gene']}\t{good['chrom']}\t{good['pos']}\t{good['ref']}\t{good['alt']}\n")
    variants = variant_inventory(html, HEADER + rows)
    assert variants[0]["parse_errors"][0]["code"] == "invalid_position"
    assert variants[1]["input_status"] == "ready"
    write_json(tmp_path / "checked-inventory.json", dict(variants=variants))
    (tmp_path / "reference").symlink_to(data / "reference", target_is_directory=True)
    source = tmp_path / "source"
    source.mkdir()
    shared = sid_read("osteosarc_all_variants/source/t2-all-variant-regions.bam")
    for suffix in ("", ".bai"):
        shutil.copyfile(str(shared) + suffix, source / ("t2-all-variant-regions.bam" + suffix))
    bam = source / "t2-all-variant-regions.bam"
    write_json(source / "bam.receipt.json", dict(sha256=digest(bam), index_sha256=digest(str(bam) + ".bai")))
    audit(tmp_path)
    report(tmp_path)
    bad = json.loads((tmp_path / "outcomes/example.json").read_text())
    assert bad["status"] == "malformed_source_row"
    assert bad["rna"] is None
    assert bad["input"]["parse_errors"] == variants[0]["parse_errors"]
    result = json.loads((tmp_path / "outcomes" / (good["variant_id"] + ".json")).read_text())
    expected = next(r for r in json.loads((data / "expected.json").read_text())
                    if r["input"]["variant_id"] == good["variant_id"])
    assert result["status"] == expected["status"]
    assert result["rna"] == expected["rna"]
    assert "malformed_source_row | unavailable" in (tmp_path / "README.md").read_text()


@pytest.mark.parametrize("case", ["versioned", "unnamed", "unnamed_null", "simple", "simple_blank_version"])
@pytest.mark.parametrize("writer_style", ["result", "dataframe"])
def test_combined_tables_keep_measurements_and_rankings_through_wide_files(tmp_path, case, writer_style):
    from topiary import combine_sources, is_named_version, rank_candidates
    from .test_twin_conformance import DELIMITED_IO_TWINS

    def table(value, method, version):
        return pd.DataFrame(dict(peptide=["SIINFEKL"], allele=["HLA-A*02:01"],
                                 kind=["pMHC_affinity"], value=[value],
                                 prediction_method_name=[method], predictor_version=[version]))

    if case == "versioned":
        sources = {"old": table(50., "original", "4.1b"),
                   "new": table(75., "original", "4.2"),
                   "unversioned": table(120., "original", None),
                   # A real method name can itself look version-encoded.
                   "suffix": table(90., "original_4.1b", None)}
    elif case == "unnamed":
        sources = {"only": table(50., None, None).drop(columns=["prediction_method_name", "predictor_version"])}
    elif case == "unnamed_null":
        sources = {"only": table(50., pd.NA, pd.NA)}
    elif case == "simple_blank_version":
        sources = {"only": table(50., "original", "")}
    else:
        sources = {"only": table(50., "original", None).drop(columns="predictor_version")}
    combined = combine_sources(sources, sample_name="p")
    policy = dict(ascending=True, strata=["source_label"])
    before = rank_candidates(combined, "affinity.value", **policy)
    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        wide = combined.to_wide()
        path = tmp_path / f"combined.{suffix}"
        if writer_style == "result":
            method(wide, path)
        else:
            writer(wide.df, path)
        restored = reader(path).to_long()
        original_rows = combined.df.sort_values("source_label").reset_index(drop=True)
        restored_rows = restored.df.sort_values("source_label").reset_index(drop=True)
        for column in ("value", "prediction_method_name", "predictor_version", "source_observation_id"):
            left, right = original_rows[column], restored_rows[column]
            if column == "predictor_version":
                # Blank and null versions are both unstated; file IO must
                # preserve that fact instead of inventing a known version.
                left = left.where(left.map(is_named_version), np.nan)
                right = right.where(right.map(is_named_version), np.nan)
            pd.testing.assert_series_equal(left.where(left.notna(), np.nan), right.where(right.notna(), np.nan),
                                           check_dtype=False)
        after = rank_candidates(restored, "affinity.value", **policy)
        assert before.candidate_score.tolist() == after.candidate_score.tolist()
        assert before.source_label.tolist() == after.source_label.tolist()
        assert len(restored.filter_by("affinity.value < 60")) == 1


def test_nearest_self_predictions_keep_allele_aggregation_after_combination_and_reload(tmp_path):
    from topiary import combine_sources, evaluate_scores, parse, read_tsv
    from .test_self_nearest_population import _predict

    original = _predict(predict_self_nearest=True)
    combined = combine_sources({"only": original}, sample_name="p")
    path = tmp_path / "self.tsv"
    combined.to_wide().to_tsv(path)
    restored = read_tsv(path).to_long()
    for expression in ("affinity.best_value", "self_nearest.affinity.best_value"):
        native = evaluate_scores(original, parse(expression))
        assert native.nunique() == 1
        for frame in (combined.df, restored.df):
            np.testing.assert_allclose(evaluate_scores(frame, parse(expression)), native)


@pytest.mark.parametrize("versions", [("1", "2"), ("01", "1"), ("1.10", "1.1")])
@pytest.mark.parametrize("wide", [False, True])
def test_numeric_predictor_versions_are_opaque_through_files_and_ranking(tmp_path, versions, wide):
    from topiary import combine_sources, rank_candidates
    from .test_twin_conformance import DELIMITED_IO_TWINS

    def table(value, version):
        return pd.DataFrame(dict(peptide=["SIINFEKL"], allele=["HLA-A*02:01"],
                                 kind=["pMHC_affinity"], value=[value],
                                 prediction_method_name=["original"], predictor_version=[version],
                                 wt_value=[value + 1], wt_predictor_version=[version]))

    combined = combine_sources({"first": table(50., versions[0]),
                                "second": table(75., versions[1]),
                                "unknown": table(100., None)}, sample_name="p")
    policy = dict(strata=["source_label"])
    before = rank_candidates(combined, "affinity.value", **policy)
    assert before.candidate_score.tolist() == [100., 75., 50.]
    for suffix, _, writer, reader in DELIMITED_IO_TWINS:
        path = tmp_path / f"numeric.{suffix}"
        writer(combined.to_wide() if wide else combined, path)
        restored = reader(path).to_long()
        rows = restored.df.set_index("source_label")
        for column in ("predictor_version", "source_predictor_version", "wt_predictor_version"):
            assert rows.loc["first", column] == versions[0]
            assert rows.loc["second", column] == versions[1]
            assert pd.isna(rows.loc["unknown", column])
        after = rank_candidates(restored, "affinity.value", **policy)
        assert after.candidate_score.tolist() == before.candidate_score.tolist()
        assert after.source_label.tolist() == before.source_label.tolist()
        assert rows.loc[["first", "second", "unknown"], "wt_value"].tolist() == [51., 76., 101.]
        for version, value in zip(versions, (50., 75.)):
            selected = rank_candidates(restored, f"affinity['original', '{version}'].value", **policy)
            assert selected.candidate_score.dropna().tolist() == [value]
        assert len(restored.filter_by("affinity.value < 60")) == 1
@pytest.mark.parametrize("position,expected_offsets", [
    (2, {0, 1}), ("2,11", {0, 1, 3, 4}), ("2-3", {0, 1, 2}),
    (None, set()), ("?", set()), (99, set()),
])
def test_imported_pvacseq_geometry_changes_novel_only_windows(tmp_path, position, expected_offsets):
    from topiary import TopiaryPredictor, fragments_from_dataframe, read_pvacseq
    from tests.report_geometry_helpers import write_pvacseq_geometry_report
    from tests.test_twin_conformance import PVACSEQ_MUTATION_GEOMETRY_TWINS

    outputs = []
    for flavor in PVACSEQ_MUTATION_GEOMETRY_TWINS:
        path = write_pvacseq_geometry_report(tmp_path / f"{flavor}.tsv", flavor, position)
        source = read_pvacseq(path)
        # Imported measurements survive conversion-independent table IO.
        saved = tmp_path / f"{flavor}-saved.tsv"
        source.to_tsv(saved)
        from topiary import read_tsv
        frames = (source.long_df, read_tsv(saved).long_df)
        for frame in frames:
            fragments = fragments_from_dataframe(frame)
            model = RandomBindingPredictor(alleles=["HLA-A*02:01"], default_peptide_lengths=[8])
            all_rows = TopiaryPredictor(models=model).predict_from_fragments(fragments)
            novel = TopiaryPredictor(models=model, only_novel_epitopes=True).predict_from_fragments(fragments)
            assert set(all_rows.peptide_offset) == {0, 1, 2, 3, 4}
            assert set(novel.peptide_offset) == expected_offsets
            assert len(novel) < len(all_rows)
            assert novel.overlaps_target.all()
            assert all_rows.source_type.eq("variant:substitution").all()
            assert all_rows.source.eq(f"pvacseq-{flavor}:{path.name}").all()
            if position is None or position == "?":
                assert all_rows.overlaps_target.isna().all()
                assert all_rows.target_interval_status.eq("unknown").all()
            outputs.append(set(novel.peptide))
    assert all(peptides == outputs[0] for peptides in outputs)


def test_reported_wt_peptides_reach_optional_wt_predictions(tmp_path):
    from topiary import TopiaryPredictor, fragments_from_dataframe, read_pvacseq
    from tests.report_geometry_helpers import write_pvacseq_geometry_report
    from tests.test_twin_conformance import PVACSEQ_MUTATION_GEOMETRY_TWINS

    for flavor in PVACSEQ_MUTATION_GEOMETRY_TWINS:
        path = write_pvacseq_geometry_report(tmp_path / f"{flavor}.tsv", flavor)
        fragments = fragments_from_dataframe(read_pvacseq(path).long_df)
        model = RandomBindingPredictor(alleles=["HLA-A*02:01"], default_peptide_lengths=[8])
        rows = TopiaryPredictor(models=model, predict_wt=True, only_novel_epitopes=True).predict_from_fragments(fragments)
        assert set(rows.wt_peptide) == {"ACDEFGHI", "CDEFGHIK"}
        assert rows.wt_value.notna().all()


def test_cache_miss_reporting_preserves_fragment_targets_and_separates_wt_failures():
    from topiary import CachedPredictorCoverageError, PartialPredictionWarning
    from tests.test_cached_protein_scan_context import _covering_cache, PEPTIDE

    missing = "AAAAAAAAA"
    fragments = [
        ProteinFragment(fragment_id="complete", sequence=PEPTIDE, reference_sequence=PEPTIDE,
                        target_intervals=[(3, 4)], source_type="variant:substitution"),
        ProteinFragment(fragment_id="missing_wt", sequence=PEPTIDE, reference_sequence=missing,
                        target_intervals=[(3, 4)], source_type="variant:substitution"),
        ProteinFragment(fragment_id="missing_mutant", sequence=missing,
                        target_intervals=[(3, 4)], source_type="variant:substitution"),
    ]
    model = _covering_cache([PEPTIDE])
    strict = TopiaryPredictor(models=[model], predict_wt=True, only_novel_epitopes=True)
    with pytest.raises(CachedPredictorCoverageError):
        strict.predict_from_fragments(fragments)
    with pytest.raises(CachedPredictorCoverageError):
        strict.predict_from_fragments(fragments[:2])
    failures = []
    reporting = TopiaryPredictor(models=[model], predict_wt=True, only_novel_epitopes=True,
                                cache_miss_handler=failures.append, sort_by="affinity.value")
    with pytest.warns(PartialPredictionWarning):
        rows = reporting.predict_from_fragments(fragments).set_index("fragment_id")
    assert set(rows.index) == {"complete", "missing_wt"}
    assert rows.overlaps_target.all() and rows.value.notna().all()
    assert rows.loc["complete", "wt_value"] == rows.loc["complete", "value"]
    assert pd.isna(rows.loc["missing_wt", "wt_value"])
    assert {(r["source_sequence_name"], r["stage"]) for r in failures} == {
        ("missing_mutant", "protein"), ("missing_wt", "wildtype")}


def test_self_nearest_cache_miss_retains_primary_scores_with_an_explicit_report():
    from topiary import CachedPredictorCoverageError, PartialPredictionWarning, SelfProteome
    from tests.test_cached_protein_scan_context import _covering_cache, PEPTIDE

    model = _covering_cache([PEPTIDE])
    proteome = SelfProteome.from_peptides({"self": "SIINFEKLL"}, peptide_lengths=[9])
    options = dict(models=[model], self_proteome=proteome, predict_self_nearest=True)
    with pytest.raises(CachedPredictorCoverageError):
        TopiaryPredictor(**options).predict_from_named_peptides({"candidate": PEPTIDE})
    failures = []
    with pytest.warns(PartialPredictionWarning):
        rows = TopiaryPredictor(**options, cache_miss_handler=failures.append).predict_from_named_peptides(
            {"candidate": PEPTIDE})
    assert rows.peptide.tolist() == [PEPTIDE] and rows.value.notna().all()
    assert rows.self_nearest_value.isna().all()
    assert failures[0]["stage"] == "self_nearest"
    assert failures[0]["source_sequence_name"] == "SIINFEKLL"


def test_self_nearest_miss_cannot_borrow_a_score_from_another_instance_of_the_same_model():
    from topiary import PartialPredictionWarning, SelfProteome
    from tests.test_cached_protein_scan_context import _covering_cache, PEPTIDE

    # The same logical model/version appears in two separately configured
    # caches. Only the second cache can answer the comparator query.
    models = [_covering_cache([PEPTIDE]), _covering_cache([PEPTIDE, "SIINFEKLL"])]
    failures = []
    predictor = TopiaryPredictor(
        models=models, self_proteome=SelfProteome.from_peptides({"self": "SIINFEKLL"}, peptide_lengths=[9]),
        predict_self_nearest=True, cache_miss_handler=failures.append)
    with pytest.warns(PartialPredictionWarning):
        rows = predictor.predict_from_named_peptides({"candidate": PEPTIDE})
    assert len(rows) == 2 and rows.value.notna().all()
    assert rows.self_nearest_value.isna().sum() == 1
    assert rows.self_nearest_value.dropna().tolist() == [150.0]
    assert len(failures) == 1 and failures[0]["model_key"] == "netmhcpan__1"


def test_lens_unknown_geometry_stays_unknown_through_rescanning():
    from topiary import TopiaryPredictor, fragments_from_dataframe, read_lens

    source = read_lens("tests/data/lens/sample_v1_4.tsv").long_df
    fragments = fragments_from_dataframe(source)
    assert {f.source_type for f in fragments} >= {"variant:snv", "sv:fusion", "erv", "self"}
    assert all(f.target_intervals is None for f in fragments)
    model = RandomBindingPredictor(alleles=["HLA-A*02:01"], default_peptide_lengths=[8])
    rows = TopiaryPredictor(models=model).predict_from_fragments(fragments)
    assert not rows.empty and rows.overlaps_target.isna().all()
    assert rows.target_interval_status.eq("unknown").all()
    assert rows.antigen_source.notna().all() and rows.source.eq("lens-v1.4").all()


def test_known_junction_geometry_survives_novel_only_context_rescanning():
    from topiary import TopiaryPredictor, fragments_from_dataframe

    frame = pd.DataFrame([dict(peptide="SIINFEKL", pep_context="AASIINFEKLCCGGGG",
                              source_type="sv:fusion", mutation_start_in_peptide=3,
                              mutation_end_in_peptide=3)])
    fragments = fragments_from_dataframe(frame)
    model = RandomBindingPredictor(alleles=["HLA-A*02:01"], default_peptide_lengths=[8])
    all_rows = TopiaryPredictor(models=model).predict_from_fragments(fragments)
    novel = TopiaryPredictor(models=model, only_novel_epitopes=True).predict_from_fragments(fragments)
    assert set(novel.peptide_offset) == {0, 1, 2, 3, 4}
    assert len(novel) < len(all_rows)
    assert novel.contains_mutant_residues.isna().all()  # Junctions are not substituted residues.
    assert novel.overlaps_target.all()


@pytest.mark.parametrize("departure,priority", [("breakpoint_junction", 10),
                                               ("splice_ambiguous_event_junction", 75),
                                               (None, 75)])
def test_sv_annotated_product_requires_its_own_event_linkage_in_both_doors(tmp_path, departure, priority):
    import json
    from .test_sv_interest import catalogue
    from .test_twin_conformance import SV_INTEREST_REPORT_TWINS
    build, cli = SV_INTEREST_REPORT_TWINS
    translated = {} if departure is None else dict(departure_relations=[departure])
    comparison = dict(schema="isovar.sv_rna_prediction_comparison.v1", event_id="UTR", sample_id="T2",
                      rna_source="library", hypotheses=[dict(kind="annotated_frame", hypothesis_id="frame",
                      amino_acids="MKK", nucleotide_sequence="ATGAAAAAA", complete_candidate=False,
                      paths={"path": dict(linkage="breakpoint_junction", candidate=translated)})])
    cat = catalogue()
    expected = build(cat, comparisons=[comparison])
    assert expected["protein_hypotheses"][0]["evidence_priority"] == priority
    assert expected["protein_hypotheses"][0]["event_linkage_relations"] == ([] if departure is None else [departure])
    (tmp_path / "catalogue.json").write_text(json.dumps(cat))
    (tmp_path / "comparison.json").write_text(json.dumps(comparison))
    assert cli(["--catalogue", str(tmp_path / "catalogue.json"), "--comparison", str(tmp_path / "comparison.json"),
                "--output-prefix", str(tmp_path / "report")]) == 0
    assert json.loads((tmp_path / "report.json").read_text()) == json.loads(json.dumps(expected))
