"""#304: synthetic cache rows are constructed beside the regression tests."""

from io import StringIO
import json

import pandas as pd
import pytest

from topiary import (
    CachedPredictor, CachedPredictorCoverageError, PartialPredictionWarning,
    TopiaryPredictor, predict_with_cache_miss_report,
)
from topiary.cli.script import main
from tests.test_cached_protein_scan_context import _covering_cache, _row, PEPTIDE
from tests.test_twin_conformance import CACHE_MISS_PREDICTION_TWINS

MISSING = "AAAAAAAAA"


@pytest.mark.parametrize("door", CACHE_MISS_PREDICTION_TWINS)
def test_strict_and_reporting_policies_drive_both_prediction_doors(door):
    inputs = {"covered": PEPTIDE, "missing": MISSING, "also_covered": PEPTIDE}
    model = _covering_cache([PEPTIDE])
    strict = TopiaryPredictor(models=[model], raise_on_error=False)
    with pytest.raises(CachedPredictorCoverageError):
        door(strict, inputs)
    failures = []
    reporting = TopiaryPredictor(models=[model], cache_miss_handler=failures.append)
    with pytest.warns(PartialPredictionWarning, match="skipped 1"):
        actual = door(reporting, inputs)
    expected = door(strict, {key: value for key, value in inputs.items() if key != "missing"})
    pd.testing.assert_frame_equal(actual, expected)
    failure, = failures
    assert failure["source_sequence_name"] == "missing" and failure["scope"] == "model_input"
    assert failure["model_key"] == failure["prediction_method_name"] == "netmhcpan"
    assert failure["predictor_version"] == "4.2" and failure["error_type"] == "CachedPredictorCoverageError"
    assert MISSING in failure["message"]


@pytest.mark.parametrize("door", CACHE_MISS_PREDICTION_TWINS)
def test_other_models_still_score_the_missing_input_and_reports_do_not_leak_between_calls(door):
    first = _covering_cache([PEPTIDE])
    second = _covering_cache([PEPTIDE, MISSING], prediction_method_name="other", predictor_version="1")
    failures = []
    predictor = TopiaryPredictor(models=[first, second], cache_miss_handler=failures.append)
    with pytest.warns(PartialPredictionWarning):
        result = door(predictor, {"covered": PEPTIDE, "missing": MISSING})
    assert set(zip(result["prediction_method_name"], result["source_sequence_name"])) == {
        ("netmhcpan", "covered"), ("other", "covered"), ("other", "missing")}
    assert len(failures) == 1
    failures.clear()
    assert len(door(predictor, {"covered": PEPTIDE})) == 2 and not failures


def test_reporting_preserves_protein_context_and_skips_whole_affected_sequence():
    from tests.test_cached_protein_scan_context import _flanked_cache

    covered, mismatched = "MA" + PEPTIDE + "GG", "WW" + PEPTIDE + "CC"
    model = _flanked_cache(covered, mismatched, n_flank="MA", c_flank="GG")
    failures = []
    predictor = TopiaryPredictor(models=[model], cache_miss_handler=failures.append)
    with pytest.warns(PartialPredictionWarning):
        result = predictor.predict_from_named_sequences({"covered": covered, "wrong_flanks": mismatched})
    expected = TopiaryPredictor(models=[model]).predict_from_named_sequences({"covered": covered})
    pd.testing.assert_frame_equal(result, expected)
    assert "different flanking context" in failures[0]["message"]
    assert failures[0]["source_sequence_name"] == "wrong_flanks"


@pytest.mark.parametrize("dimension", ["kind", "genotype"])
def test_one_covered_prediction_cannot_hide_a_missing_kind_or_genotype(dimension):
    protein = "MA" + PEPTIDE + "GG"
    rows = [_row(protein[q:q + 9]) for q in range(len(protein) - 8) if protein[q:q + 9] != PEPTIDE]
    rows += [_row(MISSING)]  # An independent covered sequence survives.
    if dimension == "kind":
        rows += [_row(PEPTIDE), _row(PEPTIDE, kind="pMHC_presentation", n_flank="WW", c_flank="CC")]
    else:
        rows += [_row(PEPTIDE, kind="pMHC_presentation", allele_set=genotype, n_flank=n, c_flank=c)
                 for genotype, n, c in [("HLA-A*02:01,HLA-B*07:02", "MA", "GG"),
                                       ("HLA-A*02:01,HLA-C*07:01", "WW", "CC")]]
    cache = CachedPredictor.from_dataframe(pd.DataFrame(rows))
    failures = []
    predictor = TopiaryPredictor(models=[cache], cache_miss_handler=failures.append)
    with pytest.warns(PartialPredictionWarning):
        result = predictor.predict_from_named_sequences({"partly_covered": protein, "complete": MISSING})
    assert result.source_sequence_name.tolist() == ["complete"]
    assert failures[0]["source_sequence_name"] == "partly_covered"
    assert "pMHC_presentation" in failures[0]["message"]
    if dimension == "genotype":
        assert "HLA-C*07:01" in failures[0]["message"]


@pytest.mark.parametrize("door", CACHE_MISS_PREDICTION_TWINS)
def test_all_missing_returns_no_fabricated_predictions_and_reports_every_input(door):
    failures = []
    predictor = TopiaryPredictor(models=[_covering_cache([PEPTIDE])], cache_miss_handler=failures.append)
    with pytest.warns(PartialPredictionWarning, match="skipped 2"):
        result = door(predictor, {"one": MISSING, "two": MISSING})
    assert result.empty and [r["source_sequence_name"] for r in failures] == ["one", "two"]
    assert door(predictor, {}).empty


@pytest.mark.parametrize("error", [KeyError("bug"), ValueError("invalid"), RuntimeError("setup")])
def test_unrelated_errors_never_become_cache_misses(error):
    failures = []

    def broken(inputs):
        raise error

    with pytest.raises(type(error), match=str(error).strip("'")):
        predict_with_cache_miss_report(broken, {"one": PEPTIDE}, on_miss=failures.append)
    assert failures == []


def test_handler_failure_aborts_instead_of_returning_unreported_partial_results():
    def cannot_save(failure):
        raise OSError("report disk full")

    model = _covering_cache([PEPTIDE])
    with pytest.raises(OSError, match="report disk full"):
        predict_with_cache_miss_report(model.predict_proteins_dataframe, {"one": MISSING}, on_miss=cannot_save)
    with pytest.raises(TypeError, match="callable"):
        TopiaryPredictor(models=[model], cache_miss_handler=True)


def test_sparse_failures_keep_successful_partitions_batched():
    calls, failures = [], []
    inputs = {str(i): str(i) for i in range(64)}

    def predict(batch):
        calls.append(tuple(batch))
        if "17" in batch:
            raise CachedPredictorCoverageError("one missing prediction")
        return pd.DataFrame({"name": list(batch)})

    with pytest.warns(PartialPredictionWarning):
        result = predict_with_cache_miss_report(predict, inputs, on_miss=failures.append)
    assert result["name"].tolist() == [name for name in inputs if name != "17"]
    assert len(calls) == 13 and len(failures) == 1


def cli_inputs(tmp_path, peptides):
    cache = tmp_path / "cache.csv"
    pd.DataFrame([_row(PEPTIDE)]).to_csv(cache, index=False)
    fasta = tmp_path / "inputs.fasta"
    fasta.write_text("".join(f">input{i}\n{peptide}\n" for i, peptide in enumerate(peptides)))
    args = ["--mhc-cache-file", str(cache), "--mhc-cache-format", "topiary_output",
            "--fasta", str(fasta), "--output-csv", "-"]
    return args, cache, fasta


@pytest.mark.parametrize("peptides,exit_code", [([PEPTIDE, MISSING], 3), ([MISSING], 3), ([PEPTIDE], 0)])
@pytest.mark.parametrize("input_flag", ["--fasta", "--peptide-fasta"])
def test_cli_persists_explicit_completeness_and_returns_partial_exit_status(
        tmp_path, capsys, peptides, exit_code, input_flag):
    args, _, _ = cli_inputs(tmp_path, peptides)
    args[args.index("--fasta")] = input_flag
    report = tmp_path / "misses.json"
    if exit_code:
        with pytest.warns(PartialPredictionWarning):
            assert main(args + ["--cache-miss-report", str(report)]) == exit_code
    else:
        assert main(args + ["--cache-miss-report", str(report)]) == exit_code
    captured = capsys.readouterr()
    data = json.loads(report.read_text())
    assert data["schema"] == "topiary.cache_miss_report.v1" and data["complete"] == (exit_code == 0)
    assert len(data["failures"]) == peptides.count(MISSING)
    assert data["prediction_rows"] == peptides.count(PEPTIDE)
    frame = pd.read_csv(StringIO(captured.out))
    assert len(frame) == peptides.count(PEPTIDE)
    if PEPTIDE in peptides:
        assert frame["peptide"].tolist() == [PEPTIDE]
    if exit_code:
        assert "Partial result:" in captured.err and "Exit status 3" in captured.err


def test_cli_default_is_still_strict(tmp_path, capsys):
    args, _, _ = cli_inputs(tmp_path, [PEPTIDE, MISSING])
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2 and capsys.readouterr().out == ""


def test_cli_report_write_failure_prevents_prediction_output(tmp_path, capsys):
    args, _, _ = cli_inputs(tmp_path, [PEPTIDE, MISSING])
    with pytest.warns(PartialPredictionWarning), pytest.raises(SystemExit) as error:
        main(args + ["--cache-miss-report", str(tmp_path / "absent" / "report.json")])
    assert error.value.code == 2 and capsys.readouterr().out == ""


@pytest.mark.parametrize("collision", ["cache", "input", "output", "stdout", "hardlink"])
def test_cli_rejects_report_collisions_before_touching_inputs(tmp_path, collision):
    args, cache, fasta = cli_inputs(tmp_path, [PEPTIDE])
    before = cache.read_bytes(), fasta.read_bytes()
    path = {"cache": cache, "input": fasta, "output": tmp_path / "new.csv", "stdout": "-"}.get(collision)
    if collision == "hardlink":
        path = tmp_path / "alias.json"
        path.hardlink_to(cache)
    if collision == "output":
        args[-1] = str(path)
    with pytest.raises(SystemExit) as error:
        main(args + ["--cache-miss-report", str(path)])
    assert error.value.code == 2
    assert (cache.read_bytes(), fasta.read_bytes()) == before
