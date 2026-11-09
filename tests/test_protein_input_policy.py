"""Whole-protein X exclusions must not abort unrelated predictions (#573)."""

import warnings

import pandas as pd
import pytest

from topiary import ProteinFragment, TopiaryPredictor
from .test_protein_fallback import ProteinModel, filled_cache

CLEAN = "MASIINFEKLGGG"
UNKNOWN = "XASIINFEKLGGG"


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("unknown", [UNKNOWN, "MASIINXEKLGG", "MASIINFEKLx"])
def test_default_excludes_whole_x_protein_before_model_or_cache(mixed, cached, unknown):
    from topiary import ProteinInputWarning

    inputs = {"bad": unknown}
    if mixed:
        inputs["good"] = CLEAN
    model = ProteinModel()
    if cached:
        producer, model = filled_cache(model, {"good": CLEAN})
        producer.fallback = None  # An excluded X input must not even request coverage.
    else:
        original = model.predict_proteins_dataframe

        def reject_x(sequences):
            assert all("X" not in value.upper() for value in sequences.values())
            assert sequences  # All-excluded input must not call the producer.
            return original(sequences)

        model.predict_proteins_dataframe = reject_x
        producer = model
    expected = TopiaryPredictor(models=producer).predict_from_named_sequences(
        {"good": CLEAN} if mixed else {})
    model.scans.clear()
    with pytest.warns(ProteinInputWarning, match="Skipped 1 protein.*'bad'") as caught:
        result = TopiaryPredictor(models=producer).predict_from_named_sequences(inputs)
    assert len(caught) == 1
    assert "ArrowStringArray" not in str(caught[0].message)
    assert set(result.source_sequence_name) == ({"good"} if mixed else set())
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    assert model.scans == ([{"good": CLEAN}] if mixed and not cached else [])


@pytest.mark.parametrize("allow_x", [False, True])
@pytest.mark.parametrize("unknown_in", ["target", "baseline"])
def test_fragment_and_wildtype_share_protein_policy(allow_x, unknown_in):
    from topiary import ProteinInputWarning

    target, baseline = (UNKNOWN, CLEAN) if unknown_in == "target" else (CLEAN, UNKNOWN)
    fragments = [
        ProteinFragment(fragment_id="test", sequence=target,
                        reference_sequence=baseline, source_type="variant:snv"),
        ProteinFragment(fragment_id="control", sequence=CLEAN,
                        reference_sequence=CLEAN, source_type="cta"),
    ]
    predictor = TopiaryPredictor(models=ProteinModel(), predict_wt=True,
                                allow_x_in_proteins=allow_x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ProteinInputWarning)
        result = predictor.predict_from_fragments(fragments)
    assert len(caught) == (0 if allow_x else 1)
    control = result[result.fragment_id == "control"]
    assert len(control) == len(CLEAN) - 8
    assert control.wt_value.notna().all()
    test = result[result.fragment_id == "test"]
    if unknown_in == "target" and not allow_x:
        assert test.empty
    else:
        assert len(test) == len(target) - 8
        assert test.wt_value.notna().all() if allow_x else test.wt_value.isna().all()


def test_public_filter_preserves_other_residues_names_and_input_mapping():
    from topiary import ProteinInputWarning, filter_protein_sequences

    inputs = {"u": "MASIUNFEKL", "x": UNKNOWN}
    with pytest.warns(ProteinInputWarning, match="'x'"):
        assert filter_protein_sequences(inputs) == {"u": "MASIUNFEKL"}
    assert filter_protein_sequences(inputs, allow_x=True) == inputs
    assert inputs == {"u": "MASIUNFEKL", "x": UNKNOWN}
    assert filter_protein_sequences({}) == {}


@pytest.mark.parametrize("mixed", [False, True])
def test_cli_outputs_parseable_csv_and_succeeds_when_x_inputs_are_excluded(tmp_path, capsys, monkeypatch, mixed):
    from io import StringIO
    from topiary import ProteinInputWarning
    from topiary.cli import args as cli
    from topiary.cli.script import main

    fasta = tmp_path / "input.fasta"
    fasta.write_text(f">bad\n{UNKNOWN}\n" + (f">good\n{CLEAN}\n" if mixed else ""))
    monkeypatch.setattr(cli, "predictors_from_args", lambda args: [ProteinModel()])
    with pytest.warns(ProteinInputWarning):
        assert main(["--fasta", str(fasta), "--mhc-predictor", "random",
                     "--mhc-alleles", "HLA-A*02:01", "--output-csv", "-"]) == 0
    output = capsys.readouterr()
    frame = pd.read_csv(StringIO(output.out))
    assert set(frame.source_sequence_name) == ({"good"} if mixed else set())
    assert "prediction row" in output.err


def test_multiple_models_receive_only_clean_inputs_with_one_warning():
    from topiary import ProteinInputWarning

    models = [ProteinModel(), ProteinModel()]
    models[1].method = "second"
    with pytest.warns(ProteinInputWarning) as caught:
        result = TopiaryPredictor(models=models).predict_from_named_sequences({"bad": UNKNOWN, "good": CLEAN})
    assert len(caught) == 1
    assert len(result) == 2 * (len(CLEAN) - 8)
    assert all(model.scans == [{"good": CLEAN}] for model in models)


class RejectingProteinModel(ProteinModel):
    """Exercise the legacy validators' two public error shapes."""

    def __init__(self, validator="mhctools"):
        super().__init__()
        self.validator = validator

    def predict_with_flanks(self, peptides, n_flanks, c_flanks):
        if self.validator == "mhctools":
            self._check_peptide_inputs(peptides)
        else:
            # MHCflurry reports this count/prefix plus pandas' version-dependent
            # array repr. Only that validation signature is recognized.
            bad = pd.Series(peptides)[lambda s: ~s.str.match("^[ACDEFGHIKLMNPQRSTVWY]+$")]
            if len(bad):
                raise ValueError(f"{len(bad)} peptides have nonstandard amino acids: {bad.unique()}")
        return super().predict_with_flanks(peptides, n_flanks, c_flanks)


@pytest.mark.parametrize("validator", ["mhctools", "mhcflurry"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_override_skips_recognized_rejections_and_preserves_good_inputs(validator, mixed, cached):
    from topiary import CachedPredictor, ProteinInputWarning

    inputs = {"bad": UNKNOWN}
    if mixed:
        inputs["good"] = CLEAN
    model = RejectingProteinModel(validator)
    # Demonstrate the actual producer aborts this batch without isolation.
    with pytest.raises(ValueError):
        model.predict_proteins_dataframe(inputs)
    model.scans.clear()
    producer = CachedPredictor(fallback=model) if cached else model
    with pytest.warns(ProteinInputWarning, match="producer residue rejection.*'bad'") as caught:
        result = TopiaryPredictor(models=producer, allow_x_in_proteins=True).predict_from_named_sequences(inputs)
    assert len(caught) == 1 and "ArrowStringArray" not in str(caught[0].message)
    assert set(result.source_sequence_name) == ({"good"} if mixed else set())
    assert len(result) == (len(CLEAN) - 8 if mixed else 0)
    if mixed:
        expected = TopiaryPredictor(models=producer, allow_x_in_proteins=True).predict_from_named_sequences({"good": CLEAN})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    if cached:
        assert not producer.to_dataframe().peptide.str.contains("X").any()


def test_override_preserves_accepting_models_when_another_rejects():
    from topiary import ProteinInputWarning

    accepting, rejecting = ProteinModel(), RejectingProteinModel()
    rejecting.method = "rejecting"
    with pytest.warns(ProteinInputWarning) as caught:
        result = TopiaryPredictor(models=[accepting, rejecting], allow_x_in_proteins=True).predict_from_named_sequences(
            {"bad": UNKNOWN, "good": CLEAN})
    assert len(caught) == 1
    assert set(zip(result.prediction_method_name, result.source_sequence_name)) == {
        ("protein-test", "bad"), ("protein-test", "good"), ("rejecting", "good")}
    assert accepting.scans == [{"bad": UNKNOWN, "good": CLEAN}]


@pytest.mark.parametrize("error", [ValueError("malformed model output"), RuntimeError("backend failed"),
                                   ValueError("Invalid character 'X' in peptide 'XASIINFEK'")])
def test_recovery_never_masks_unrecognized_or_clean_input_errors(error):
    from topiary import predict_proteins_with_x_recovery

    def fail(inputs):
        raise error

    inputs = {"good": CLEAN} if "Invalid character" in str(error) else {"bad": UNKNOWN}
    with pytest.raises(type(error)) as caught:
        predict_proteins_with_x_recovery(fail, inputs)
    assert caught.value is error


@pytest.mark.parametrize("cache_first", [False, True])
def test_cache_gap_reporting_and_x_rejection_compose_without_duplicate_warnings(cache_first):
    from topiary import (CachedPredictorCoverageError, PartialPredictionWarning,
                         ProteinInputWarning, predict_proteins_with_x_recovery)

    inputs = {"x": UNKNOWN, "missing": "AAAAAAAAAAAA", "good": CLEAN}
    model = RejectingProteinModel()

    def producer(batch):
        if not cache_first and "x" in batch:
            return model.predict_proteins_dataframe(batch)
        if "missing" in batch:
            raise CachedPredictorCoverageError("synthetic missing coverage")
        return model.predict_proteins_dataframe(batch)

    with pytest.raises(CachedPredictorCoverageError):
        predict_proteins_with_x_recovery(producer, inputs)
    failures = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = predict_proteins_with_x_recovery(producer, inputs, on_cache_miss=failures.append)
    assert set(result.source_sequence_name) == {"good"}
    assert [f["source_sequence_name"] for f in failures] == ["missing"]
    assert [w.category for w in caught].count(PartialPredictionWarning) == 1
    assert [w.category for w in caught].count(ProteinInputWarning) == 1


@pytest.mark.parametrize("unknown_in", ["target", "baseline"])
def test_override_recovery_also_covers_fragment_and_wildtype_scans(unknown_in):
    from topiary import ProteinInputWarning

    target, baseline = (UNKNOWN, CLEAN) if unknown_in == "target" else (CLEAN, UNKNOWN)
    fragments = [ProteinFragment(fragment_id="test", sequence=target, reference_sequence=baseline),
                 ProteinFragment(fragment_id="control", sequence=CLEAN, reference_sequence=CLEAN)]
    with pytest.warns(ProteinInputWarning, match="producer residue rejection") as caught:
        result = TopiaryPredictor(models=RejectingProteinModel(), allow_x_in_proteins=True,
                                  predict_wt=True).predict_from_fragments(fragments)
    assert len(caught) == 1
    assert result[result.fragment_id == "control"].wt_value.notna().all()
    if unknown_in == "target":
        assert set(result.fragment_id) == {"control"}
    else:
        test = result[result.fragment_id == "test"]
        assert len(test) == len(CLEAN) - 8 and test.wt_value.isna().all()


def test_sparse_rejections_keep_successful_partitions_batched():
    from topiary import ProteinInputWarning

    model = RejectingProteinModel()
    inputs = {str(index): UNKNOWN if index == 17 else CLEAN for index in range(64)}
    with pytest.warns(ProteinInputWarning) as caught:
        result = TopiaryPredictor(models=model, allow_x_in_proteins=True).predict_from_named_sequences(inputs)
    assert len(caught) == 1
    assert len(model.scans) == 13
    assert result.source_sequence_name.drop_duplicates().tolist() == [key for key in inputs if key != "17"]
    assert len(result) == 63 * (len(CLEAN) - 8)


@pytest.mark.parametrize("allow_x", [False, True])
def test_exact_peptide_inputs_keep_their_existing_producer_contract(allow_x):
    result = TopiaryPredictor(models=ProteinModel(), allow_x_in_proteins=allow_x).predict_from_named_peptides(
        {"peptide": "XASIINFEK"})
    assert result.peptide.tolist() == ["XASIINFEK"]
