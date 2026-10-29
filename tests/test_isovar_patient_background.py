"""Patient-background exports remain typed comparisons, not candidates."""

from copy import deepcopy
import gzip
import json

import pandas as pd
import pytest

from topiary import read_isovar_patient_background_pairs, read_tsv
from .test_isovar_hypotheses import hypothesis_export
from .test_twin_conformance import ISOVAR_PATIENT_BACKGROUND_INPUT_TWINS


def _support(evidence_sets, identity, reads, fragments):
    support = dict(
        reads=reads, fragments=fragments, umis=None, cells=None,
        unlabeled_reads=None, unknown_library_reads=None,
        umis_complete=None, cells_complete=None, evidence_set_id=identity,
    )
    evidence_sets[identity] = dict(
        support,
        evidence_scope=["patient", "RNA"],
        read_ids=[f"{identity}-read-{index}" for index in range(reads)],
        fragment_ids=[f"{identity}-fragment-{index}" for index in range(fragments)],
    )
    return support


def _coding_support(evidence_sets, identity, interval, depth):
    span = _support(evidence_sets, identity + "-span", int(bool(min(depth))),
                    int(bool(min(depth))))
    single = _support(evidence_sets, identity + "-single", int(bool(min(depth))),
                      int(bool(min(depth))))
    return dict(
        scope="exact_observed_coding_interval",
        coding_interval=interval,
        fragment_depth=depth,
        minimum_fragment_depth=min(depth) if depth else 0,
        fully_tiled=bool(depth) and all(depth),
        allele_observation_span_support=span,
        single_read_span_support=single,
    )


def patient_background_export(*, unavailable=False):
    """A pair sharing exact identities with the checked-in hypothesis export."""
    hypotheses = hypothesis_export("protein-v2")
    source_event = hypotheses["events"][0]
    source_protein = source_event["protein_hypotheses"][0]
    source_translation = source_protein["translations"][0]
    evidence_sets = {}
    mutant_sequence = source_translation["nucleotide_sequence"]
    mutant_interval = source_translation["translated_interval"]
    mutant_support = _support(evidence_sets, "mutant-assembly", 2, 1)
    mutant_coding = _coding_support(
        evidence_sets, "mutant", mutant_interval,
        [1] * (mutant_interval[1] - mutant_interval[0]),
    )
    baseline_sequence = mutant_sequence[:8] + "A" + mutant_sequence[9:]
    baseline_support = _coding_support(
        evidence_sets, "baseline", mutant_interval,
        [0] * (mutant_interval[1] - mutant_interval[0]),
    )
    pair = dict(
        pair_id="pair-1",
        hypothesis_id=source_protein["hypothesis_id"],
        translation_id=source_translation["translation_id"],
        transcript_id=source_translation["reference_context"]["transcript_ids"][0],
        strand="+", translation_table=1,
        state="unavailable" if unavailable else "paired",
        reason="transcript_edits_do_not_reproduce_observed_rna" if unavailable else None,
        baseline_policy="focal",
        mutant=dict(
            nucleotide_sequence=mutant_sequence,
            nucleotide_sequence_id=source_translation["nucleotide_sequence_id"],
            coding_interval=mutant_interval,
            protein_sequence=source_protein["amino_acids"],
            ends_with_stop_codon=True,
            rna_support=mutant_support,
            coding_support=mutant_coding,
        ),
        phase=dict(
            state="single_original_read_span", scope="the_mutant_coding_window",
            normal_tissue_observed=False,
        ),
        baseline=None if unavailable else dict(
            construction="transcript_cdna_somatic_reversion",
            observation="computed_only",
            nucleotide_sequence=baseline_sequence,
            nucleotide_sequence_id="baseline-nt-1",
            coding_interval=mutant_interval,
            available_coding_interval=mutant_interval,
            protein_sequence="MAKG",
            ends_with_stop_codon=True,
            trailing_incomplete_codon_bases=0,
            reference_allele_coding_support=baseline_support,
            reference_allele_phase="incomplete_rna_coverage",
            normal_tissue_observed=False,
            translation_observed=False,
        ),
        reference=None if unavailable else dict(
            transcript_id=source_translation["reference_context"]["transcript_ids"][0],
            cdna_interval=[100, 100 + len(mutant_sequence)],
            nucleotide_sequence=baseline_sequence,
            sequence_id="reference-nt-1",
        ),
        removed_edits=[] if unavailable else [dict(
            origin="nominated_variant", cdna_interval=[108, 109],
            observed_interval=[8, 9], alt_bases="C", reference_bases="A",
        )],
        retained_edits=[],
        translation_observed=False,
    )
    return dict(
        schema="isovar.patient_background_pairs.v1",
        isovar_version="1.64.0",
        sample_id="patient", source="RNA", evidence_scope=["patient", "RNA"],
        interval_convention="zero_based_half_open", baseline_policy="focal",
        supplied_variants={"focal": {"origin": "somatic", "variant": source_event["variant"]}},
        events=[dict(
            event_id=source_event["event_id"], variant=source_event["variant"],
            pairs=[pair], state="retained", protein_hypotheses_complete=True,
            protein_sequence_limit=None,
            phased_variants_in_supporting_reads=[source_event["event_id"]],
        )],
        evidence_sets=evidence_sets,
    )


def read_all(export, tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(export))
    gzip_path = tmp_path / "pairs.json.gz"
    with gzip.open(gzip_path, "wt", encoding="utf-8") as handle:
        json.dump(export, handle)
    return [reader(export, path, gzip_path)
            for _, reader in ISOVAR_PATIENT_BACKGROUND_INPUT_TWINS]


@pytest.mark.parametrize("limit,complete", [(0, True), (1, False), (2, True), (None, None)])
def test_reader_twins_preserve_unlimited_capped_and_unknown_settings(tmp_path, limit, complete):
    export = patient_background_export()
    event = export["events"][0]
    event.update(protein_sequence_limit=limit, protein_hypotheses_complete=complete)
    for result in read_all(export, tmp_path):
        assert result.extra["isovar_patient_background_pairs"] == export
        row = result.df.iloc[0]
        assert pd.isna(row.protein_sequence_limit) if limit is None else row.protein_sequence_limit == limit
        assert pd.isna(row.protein_hypotheses_complete) if complete is None else row.protein_hypotheses_complete == complete


@pytest.mark.parametrize("limit", [-1, False, True, 0.0, "0"])
def test_reader_twins_reject_malformed_limits(tmp_path, limit):
    export = patient_background_export()
    export["events"][0]["protein_sequence_limit"] = limit
    path, compressed = tmp_path / "pairs.json", tmp_path / "pairs.json.gz"
    path.write_text(json.dumps(export))
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        json.dump(export, handle)
    for _, reader in ISOVAR_PATIENT_BACKGROUND_INPUT_TWINS:
        with pytest.raises(ValueError, match="protein_sequence_limit"):
            reader(export, path, compressed)


def test_reader_twins_coalesce_exact_pairs_without_summing_support(tmp_path):
    export = patient_background_export()
    event = export['events'][0]
    event['pairs'] *= 3
    original = deepcopy(export)
    for result in read_all(export, tmp_path):
        assert len(result) == 1
        assert result.extra['isovar_patient_background_pairs'] == original
        row = result.df.iloc[0]
        expected = event['pairs'][0]['mutant']['rna_support']
        assert {key: row.mutant_rna_support[key] for key in expected} == expected
    assert export == original
    other = deepcopy(event)
    other['event_id'] = 'different-event'
    export['events'].append(other)
    with pytest.raises(ValueError, match='Conflicting Isovar pair_id'):
        read_all(export, tmp_path)


@pytest.mark.parametrize("unavailable", [False, True])
def test_reader_twins_preserve_pair_identity_state_and_provenance(tmp_path, unavailable):
    export = patient_background_export(unavailable=unavailable)
    original = deepcopy(export)
    outputs = read_all(export, tmp_path)
    for result in outputs:
        row = result.df.iloc[0]
        assert row.pair_state == ("unavailable" if unavailable else "paired")
        assert row.patient_background_protein_comparison == (
            "unavailable" if unavailable else "amino_acid_divergent")
        assert row.baseline_policy == "focal"
        assert row.mutant_phase == "single_original_read_span"
        assert row.mutant_fragments == 1
        assert row.mutant_rna_support["fragment_ids"] == ["mutant-assembly-fragment-0"]
        assert row.mutant_coding_support["fully_tiled"]
        assert not row.mutant_normal_tissue_observed
        assert not row.mutant_translation_observed
        if unavailable:
            assert row.pair_reason == "transcript_edits_do_not_reproduce_observed_rna"
            assert pd.isna(row.patient_background_sequence)
            assert pd.isna(row.reference_nucleotide_sequence)
        else:
            assert row.patient_background_sequence == "MAKG"
            assert row.patient_background_observation == "computed_only"
            assert row.patient_background_reference_allele_phase == "incomplete_rna_coverage"
            assert row.removed_edits[0]["origin"] == "nominated_variant"
        assert "protein_sequence" not in result.df
        assert result.extra["isovar_patient_background_pairs"] == original
    for result in outputs[1:]:
        pd.testing.assert_frame_equal(outputs[0].df, result.df)
    outputs[0].df.at[0, "removed_edits"].append({"origin": "changed"})
    assert outputs[0].extra["isovar_patient_background_pairs"] == original
    assert export == original


def test_empty_export_retains_events_and_schema(tmp_path):
    export = patient_background_export()
    event = export["events"][0]
    event.update(pairs=[], state="no_protein_hypotheses")
    for result in read_all(export, tmp_path):
        assert result.df.empty
        assert result.df.columns.tolist()
        assert result.extra["isovar_patient_background_pairs"] == export


@pytest.mark.parametrize("ends_with_stop_codon,coding_length", [(True, 3), (False, 0)])
def test_zero_length_baseline_is_an_explicit_available_outcome(
        tmp_path, ends_with_stop_codon, coding_length):
    export = patient_background_export()
    pair = export["events"][0]["pairs"][0]
    baseline = pair["baseline"]
    start, available_end = baseline["available_coding_interval"]
    pair["state"] = "zero_length_baseline"
    baseline.update(
        coding_interval=[start, start + coding_length],
        protein_sequence="",
        ends_with_stop_codon=ends_with_stop_codon,
    )
    support = baseline["reference_allele_coding_support"]
    support.update(
        coding_interval=[start, start + coding_length],
        fragment_depth=[0] * coding_length,
        minimum_fragment_depth=0,
        fully_tiled=False,
    )
    assert available_end >= start + coding_length

    for result in read_all(export, tmp_path):
        row = result.df.iloc[0]
        assert row.pair_state == "zero_length_baseline"
        assert row.patient_background_protein_comparison == "amino_acid_divergent"
        assert row.patient_background_sequence == ""
        assert row.patient_background_coding_interval == [start, start + coding_length]
        assert row.patient_background_available_coding_interval == [start, available_end]
        path = tmp_path / f"zero-length-{coding_length}.tsv"
        result.to_tsv(path)
        assert read_tsv(path).df.iloc[0].patient_background_sequence == ""


def test_unavailable_after_replay_retains_reference_and_edit_provenance(tmp_path):
    export = patient_background_export()
    pair = export["events"][0]["pairs"][0]
    pair.update(
        state="unavailable",
        reason="focal_allele_outside_paired_coding_window",
        baseline=None,
    )

    for result in read_all(export, tmp_path):
        row = result.df.iloc[0]
        assert row.pair_state == "unavailable"
        assert row.patient_background_protein_comparison == "unavailable"
        assert row.reference_nucleotide_sequence == pair["reference"]["nucleotide_sequence"]
        assert row.removed_edits[0]["origin"] == "nominated_variant"
        assert pd.isna(row.patient_background_sequence)


def test_supplied_somatic_haplotype_policy_is_preserved(tmp_path):
    export = patient_background_export()
    export["baseline_policy"] = "supplied_somatic_haplotype"
    export["events"][0]["pairs"][0]["baseline_policy"] = "supplied_somatic_haplotype"

    for result in read_all(export, tmp_path):
        assert result.df.iloc[0].baseline_policy == "supplied_somatic_haplotype"


def test_synonymous_reversion_has_identical_bounded_proteins(tmp_path):
    export = patient_background_export()
    pair = export["events"][0]["pairs"][0]
    mutant_sequence = pair["mutant"]["nucleotide_sequence"]
    baseline_sequence = mutant_sequence[:10] + "G" + mutant_sequence[11:]
    pair["baseline"].update(
        nucleotide_sequence=baseline_sequence,
        protein_sequence=pair["mutant"]["protein_sequence"],
    )
    pair["reference"].update(nucleotide_sequence=baseline_sequence)
    pair["removed_edits"] = [dict(
        origin="nominated_variant", cdna_interval=[110, 111],
        observed_interval=[10, 11], alt_bases="A", reference_bases="G",
    )]

    for result in read_all(export, tmp_path):
        row = result.df.iloc[0]
        assert row.patient_background_protein_comparison == "identical"
        assert row.patient_background_sequence == row.protein_hypothesis_sequence


def test_structured_pair_fields_round_trip_through_typed_tsv(tmp_path):
    result = read_isovar_patient_background_pairs(patient_background_export())
    path = tmp_path / "pairs.tsv"
    result.to_tsv(path)
    restored = read_tsv(path)
    assert (restored.df.at[0, "patient_background_protein_comparison"]
            == "amino_acid_divergent")
    for field in (
        "mutant_rna_support", "mutant_coding_support",
        "patient_background_reference_allele_coding_support",
        "isovar_variant", "removed_edits", "retained_edits",
    ):
        assert restored.df.at[0, field] == result.df.at[0, field]


@pytest.mark.parametrize("fault,match", [
    ("schema", "Expected isovar.patient_background_pairs.v1"),
    ("scope", "evidence_scope"),
    ("policy", "baseline_policy"),
    ("event", "Duplicate Isovar event_id"),
    ("pair", "Conflicting Isovar pair_id"),
    ("state", "pair state"),
    ("translation", "translation length"),
    ("coding", "fragment_depth"),
    ("evidence", "Unknown Isovar evidence_set_id"),
    ("observation", "evidence state"),
    ("normal", "evidence state"),
    ("reference", "reference transcript_id"),
    ("edit", "removed_edits observed_interval"),
    ("edit_length", "edit sequence lengths"),
    ("replay", "do not reproduce mutant nucleotide sequence"),
    ("baseline", "do not reproduce patient-background sequence"),
    ("limit", "protein_sequence_limit"),
])
def test_reader_twins_reject_inconsistent_pair_exports(tmp_path, fault, match):
    export = patient_background_export()
    event = export["events"][0]
    pair = event["pairs"][0]
    if fault == "schema":
        export["schema"] = "isovar.patient_background_pairs.v2"
    elif fault == "scope":
        export["evidence_scope"] = ["other", "RNA"]
    elif fault == "policy":
        pair["baseline_policy"] = "supplied_somatic_haplotype"
    elif fault == "event":
        export["events"].append(deepcopy(event))
    elif fault == "pair":
        event["pairs"].append(deepcopy(pair))
        event["pairs"][-1]["hypothesis_id"] = "contradictory-hypothesis"
    elif fault == "state":
        pair["state"] = "fallback"
    elif fault == "translation":
        pair["mutant"]["coding_interval"][1] -= 1
    elif fault == "coding":
        pair["mutant"]["coding_support"]["fragment_depth"].pop()
    elif fault == "evidence":
        export["evidence_sets"].pop("mutant-assembly")
    elif fault == "observation":
        pair["baseline"]["observation"] = "reference_allele_rna_span"
    elif fault == "normal":
        pair["baseline"]["normal_tissue_observed"] = True
    elif fault == "reference":
        pair["reference"]["transcript_id"] = "different"
    elif fault == "edit":
        pair["removed_edits"][0]["observed_interval"] = [-1, 0]
    elif fault == "edit_length":
        pair["removed_edits"][0]["reference_bases"] = "AA"
    elif fault == "replay":
        pair["reference"]["nucleotide_sequence"] = (
            "T" + pair["reference"]["nucleotide_sequence"][1:])
    elif fault == "baseline":
        pair["baseline"]["nucleotide_sequence"] = (
            "T" + pair["baseline"]["nucleotide_sequence"][1:])
    else:
        event["protein_sequence_limit"] = -1
    for name, reader in ISOVAR_PATIENT_BACKGROUND_INPUT_TWINS:
        with pytest.raises(ValueError, match=match):
            if name == "mapping":
                reader(export, None, None)
            else:
                path = tmp_path / (name + ".json")
                if name == "gzip":
                    path = path.with_suffix(".json.gz")
                    with gzip.open(path, "wt", encoding="utf-8") as handle:
                        json.dump(export, handle)
                    reader(export, None, path)
                else:
                    path.write_text(json.dumps(export))
                    reader(export, path, None)


def test_reader_does_not_import_isovar(monkeypatch):
    import builtins

    original = builtins.__import__

    def guard(name, *args, **kwargs):
        assert name.split(".")[0] != "isovar", "comparison import invoked Isovar"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    assert len(read_isovar_patient_background_pairs(patient_background_export())) == 1
