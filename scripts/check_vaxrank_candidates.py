"""Run with pytest and Vaxrank installed; verifies the real consumer boundary.

Vaxrank is a downstream integration-test dependency, never a Topiary runtime
dependency. The CI job installs a pinned released wheel before running this.
"""

import pytest
import pandas as pd
from mhctools import Prediction

from topiary import (
    SelectionPolicy, combine_sources, evaluate_scores, join_annotations, parse,
    rank_with_policy, read_selection_policy, write_selection_policy, rescore_candidates,
)
from tests.test_candidate_tables import Model, source
from vaxrank.candidate_epitope import candidate_epitopes_from_rows
from vaxrank.epitope_config import EpitopeConfig
from vaxrank.epitope_dsl import attach_per_allele_scores
from vaxrank.vaccine_antigen import (
    AminoAcidInterval, TargetableMask, TumorSpecificityAttestation, VaccineAntigen,
)
from vaxrank.vaccine_peptide import VaccinePeptide


@pytest.mark.parametrize("antigen_kind", ["mutation", "fusion", "splice", "CTA", "ERV", "viral"])
@pytest.mark.parametrize("policy", ["original", "rescored", "rna_overlay"])
def test_candidate_features_reach_vaxrank_scoring_and_vaccine_construction(antigen_kind, policy, tmp_path):
    proteins = ["MAAASIINFEKL", "MAAAGILGFVFTL"]
    identity = dict(protein_sequence=proteins, event_id=["event-1", "event-2"])
    combined = combine_sources({
        "pipeline_one": source(antigen_source=antigen_kind, **identity),
        "pipeline_two": source(values=(60., 600.), antigen_source=antigen_kind, **identity),
        "rna_only": pd.DataFrame(dict(**identity, transcript_expression=[1., 1000.])),
    }, sample_name="synthetic-patient")
    expression = "1 / affinity.value"
    if policy == "rescored":
        combined = rescore_candidates(combined, Model(), prefix="new")
        expression = "1 / new__testmodel__pMHC_affinity__value"
    elif policy == "rna_overlay":
        keys = ["candidate_sample", "event_id", "protein_sequence_id"]
        annotations = combined.df.loc[combined.df.source_label.eq("rna_only"), [*keys, "transcript_expression"]]
        combined = join_annotations(combined, annotations, on=keys, prefix="rna",
                                    provenance={"source": "rna_only", "unit": "TPM"})
        expression = "rna_transcript_expression / affinity.value"
    saved_policy = SelectionPolicy(
        name="example-" + policy, score_by=expression,
        filter_by="n_rna_alt >= 5", duplicates="best",
    )
    policy_path = tmp_path / "selection.json"
    write_selection_policy(saved_policy, policy_path)
    restored_policy = read_selection_policy(policy_path)
    assert restored_policy.sha256 == saved_policy.sha256
    ranked = rank_with_policy(combined, restored_policy)
    selected_ids = set(ranked.df.source_observation_id)
    frame = combined.long_df[combined.long_df.source_observation_id.isin(selected_ids)].copy()
    # Vaxrank's current public scoring interface names its provenance key
    # prediction_id. Retain originals in the combined result; this is a
    # separate consumer view. Generalized CLI ingestion is vaxrank#497.
    frame["prediction_id"] = frame.source_observation_id
    rows = []
    for row in frame.itertuples():
        rows.append(dict(
            peptide=row.peptide, source=row.peptide, source_name=row.source_label,
            offset=0, prediction_id=row.prediction_id,
            mutant=Prediction(kind=row.kind, peptide=row.peptide,
                              allele=row.allele, value=row.value, score=row.score,
                              predictor_name=row.prediction_method_name,
                              predictor_version=row.predictor_version),
            source_class="mutation" if antigen_kind in {"mutation", "fusion", "splice"} else "self",
            overlaps_targetable=True, patient_alleles=[row.allele],
        ))
    epitopes = candidate_epitopes_from_rows(rows)
    cfg = EpitopeConfig(score_expr=restored_policy.score_by,
                        filter_expr=restored_policy.filter_by, min_epitope_score=0.)
    scored = attach_per_allele_scores(epitopes, cfg, topiary_df=frame)
    expected = dict(zip(frame.source_observation_id, evaluate_scores(frame, parse(expression))))
    vaccines = []
    for epitope in scored:
        assert epitope.per_allele_scores["HLA-A*02:01"] == pytest.approx(expected[epitope.prediction_id])
        sequence = epitope.sequence
        antigen = VaccineAntigen(
            kind=antigen_kind, amino_acids=sequence,
            targetable_mask=TargetableMask((AminoAcidInterval(0, len(sequence)),)),
            # This admission belongs to the synthetic test. Importing and
            # ranking a source table does not produce biological admission.
            tumor_specificity=TumorSpecificityAttestation(
                status="admitted", evidence_kind="synthetic_fixture",
                evidence_source="check_vaxrank_candidates", patient_specific=True,
                rationale_code="test_only",
            ),
            source_identifier=epitope.prediction_id,
        )
        vaccine = VaccinePeptide(
            antigen=antigen, epitopes=[epitope],
            combined_score_expr="target_epitope_score",
            ranking_rules=("target_epitope_score",),
        )
        assert vaccine.target_epitope_score == pytest.approx(expected[epitope.prediction_id])
        vaccines.append(vaccine)
    vaccines.sort(key=lambda v: v.target_epitope_score, reverse=True)
    assert vaccines[0].antigen.amino_acids == ("SIINFEKL" if policy == "original" else "GILGFVFTL")
    assert len(vaccines) == 2  # two pipelines did not become four vaccine targets


def test_occurrence_policy_matches_vaxrank_sources_windows_constructs_and_native_reload(tmp_path):
    """Real Vaxrank consumer operations over retained heterogeneous evidence."""
    from dataclasses import replace
    import json
    from pathlib import Path
    from varcode import Variant
    from topiary import (
        CachedPredictor, ProteinFragment, TopiaryPredictor, TopiaryResult,
        SelectionCriterion, evaluate_selection_policy, replay_selection_policy,
        read_lens, read_pvacseq,
    )
    from vaxrank.epitope_dataset import EpitopeDataset
    from vaxrank.epitope_dsl import (
        default_score_expr, genotype_lookup, score_predictions,
        prediction_group_columns, resolve_default_methods, resolve_default_versions,
        epitopes_for_ranking,
    )
    from vaxrank.core_logic import vaccine_peptides_from_epitopes
    from vaxrank.mutant_protein_fragment import MutantProteinFragment
    from vaxrank.peptide import assemble_peptide_constructs, PeptideConstructConfig
    from vaxrank.mrna import assemble_mrna_constructs, RNAConstructConfig
    from vaxrank.native_serialization import to_native_json

    root = Path(__file__).resolve().parents[1] / "tests/data"
    translation = json.loads((root / "osteosarc_shared/translation-v1.json").read_text())
    prediction = json.loads((root / "osteosarc_shared/prediction-contract-v1.json").read_text())
    fragment = ProteinFragment.from_dict(translation["fragment"])
    # This versioned fixture is reconstructed from VCF/BAM in the full suite.
    # Its numeric binding predictions are deliberately synthetic and cached.
    direct = TopiaryPredictor(models=CachedPredictor(pd.DataFrame(prediction["rows"])),
                             only_novel_epitopes=True).predict_from_fragments([fragment])
    # Carry the original translated context alongside the measurement table.
    direct["source_sequence"] = fragment.sequence
    combined = combine_sources({
        "direct": TopiaryResult(direct),
        "normalized": source(),
        "lens": read_lens(root / "lens/sample_v1_4.tsv"),
        "pvacseq": read_pvacseq(root / "pvacseq/mhc_i_all_epitopes.tsv"),
    }, sample_name="fixture-patient")
    dataset = EpitopeDataset.from_topiary(combined)
    frame = dataset.scoring_frame()
    keys = prediction_group_columns(frame)
    cfg = EpitopeConfig()
    policy = SelectionPolicy("frozen-vaxrank", default_score_expr(cfg), score_fill=0.,
                             min_score=cfg.min_epitope_score, duplicates="best")
    contexts, expected = {}, []
    for label, part in frame.groupby("source_label", sort=False):
        contexts[label] = dict(
            default_methods=resolve_default_methods(cfg, part),
            default_versions=resolve_default_versions(cfg, part),
            alleles=genotype_lookup(dataset.epitopes, keys))
        expected.append(score_predictions(dataset.epitopes, cfg, topiary_df=part))
    evaluation = evaluate_selection_policy(TopiaryResult(frame, metadata=combined.metadata), policy,
                                           group_keys=keys, source_contexts=contexts)
    original_scores = pd.concat(expected).sort_index()
    actual_scores = evaluation.occurrences.set_index(keys).score.sort_index()
    pd.testing.assert_series_equal(actual_scores, original_scores, check_names=False, check_exact=True)

    def transfer_scores(result):
        # Match score_predictions: pre-filtered groups have no score entry.
        retained = result.occurrences.loc[result.occurrences.filter_retained]
        records = retained.set_index(keys).score.to_dict()
        return [replace(epitope, per_allele_scores={
            allele: score for (*identity, allele), score in records.items()
            if tuple(identity) == epitope.prediction_group_key}) for epitope in dataset.epitopes]

    source_ids = set(frame.loc[frame.source_label.eq("direct"), "prediction_id"])
    variant = Variant("12", 5494381, "A", "G")
    native_fragment = MutantProteinFragment(
        variant=variant, gene_name=fragment.gene, amino_acids=fragment.sequence,
        mutant_amino_acid_start_offset=10, mutant_amino_acid_end_offset=11,
        supporting_reference_transcripts=[], n_overlapping_reads=14, n_alt_reads=9,
        n_ref_reads=5, n_alt_reads_supporting_protein_sequence=9)

    def construct(epitopes):
        selected = [epitope for epitope in epitopes_for_ranking(epitopes, cfg)
                    if epitope.prediction_id in source_ids]
        windows = vaccine_peptides_from_epitopes(variant, native_fragment, selected, vaccine_peptide_length=11)
        assert windows
        ranked = [(variant, windows)]
        peptide = assemble_peptide_constructs(ranked, PeptideConstructConfig(
            min_antigen_length_aa=9, max_antigen_length_aa=11))
        mrna = assemble_mrna_constructs(ranked, RNAConstructConfig(
            signal_peptide="", include_mitd=False, poly_a_length=0, optimize_linkers=False,
            min_antigen_length_aa=9, max_antigen_length_aa=11))
        assert peptide and mrna
        return [window.amino_acids for window in windows], peptide, to_native_json(mrna)

    old_epitopes = []
    for label, part in frame.groupby("source_label", sort=False):
        ids = set(part.prediction_id)
        old_epitopes.extend(attach_per_allele_scores(
            [e for e in dataset.epitopes if e.prediction_id in ids], cfg, topiary_df=part))
    original = construct(old_epitopes)
    assert construct(transfer_scores(evaluation)) == original

    criterion = SelectionCriterion("late_occurrence", "peptide_offset >= 10", "eligibility")
    changed_policy = replace(policy, name="late-context", criteria=(criterion,),
                             filter_by='criterion("late_occurrence")')
    changed = evaluate_selection_policy(TopiaryResult(frame, metadata=combined.metadata), changed_policy,
                                       group_keys=keys, source_contexts=contexts)
    changed_epitopes = transfer_scores(changed)
    assert construct(changed_epitopes)[0] != original[0]
    excluded = changed.audit.loc[changed.audit.status.eq("fail")]
    assert not excluded.empty
    assert excluded.criterion.eq("late_occurrence").all()
    assert excluded.reason.eq("predicate_false").all()
    saved = EpitopeDataset(result=changed.evidence, epitopes=tuple(changed_epitopes), config=cfg,
                           selection={"audit": changed.audit.to_json(orient="records")})
    path = tmp_path / "native-vaxrank.tsv"
    saved.save(path)
    restored = EpitopeDataset.load(path)
    replay = replay_selection_policy(restored.result)
    assert restored.selection == saved.selection
    pd.testing.assert_frame_equal(replay.occurrences, changed.occurrences, check_exact=True)
    assert construct(restored.epitopes) == construct(changed_epitopes)
