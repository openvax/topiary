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
from enum import Enum

import numpy as np
import pandas as pd
import pytest

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

LENS = "tests/data/lens/sample_v1_4.tsv"
PVACSEQ = "tests/data/pvacseq/mhc_i_all_epitopes.tsv"
PVACSEQ_PRESENTATION = (
    "tests/data/pvacseq/mhc_i_all_epitopes_presentation.tsv"
)


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
    ("bulk_star_t0", "H1-2", 24), ("ont_t1", "H1-2", 30),
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
