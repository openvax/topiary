"""Run with pytest and Vaxrank installed; verifies the real consumer boundary.

Vaxrank is a downstream integration-test dependency, never a Topiary runtime
dependency. The CI job installs a pinned released wheel before running this.
"""

import pytest
import pandas as pd
from mhctools import Prediction

from topiary import combine_sources, evaluate_scores, join_annotations, parse, rank_candidates, rescore_candidates
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
def test_candidate_features_reach_vaxrank_scoring_and_vaccine_construction(antigen_kind, policy):
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
    ranked = rank_candidates(combined, expression, duplicates="best")
    selected_ids = set(ranked.source_observation_id)
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
    cfg = EpitopeConfig(score_expr=expression, filter_expr="n_rna_alt >= 5", min_epitope_score=0.)
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
