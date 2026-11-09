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
        {"good": CLEAN} if mixed else {}
    )
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
        ProteinFragment(
            fragment_id="test",
            sequence=target,
            reference_sequence=baseline,
            source_type="variant:snv",
        ),
        ProteinFragment(
            fragment_id="control",
            sequence=CLEAN,
            reference_sequence=CLEAN,
            source_type="cta",
        ),
    ]
    predictor = TopiaryPredictor(
        models=ProteinModel(),
        predict_wt=True,
        protein_x_policy="allow" if allow_x else "skip",
    )
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
def test_cli_outputs_parseable_csv_and_succeeds_when_x_inputs_are_excluded(
    tmp_path, capsys, monkeypatch, mixed
):
    from io import StringIO
    from topiary import ProteinInputWarning
    from topiary.cli import args as cli
    from topiary.cli.script import main

    fasta = tmp_path / "input.fasta"
    fasta.write_text(f">bad\n{UNKNOWN}\n" + (f">good\n{CLEAN}\n" if mixed else ""))
    monkeypatch.setattr(cli, "predictors_from_args", lambda args: [ProteinModel()])
    with pytest.warns(ProteinInputWarning):
        assert (
            main(
                [
                    "--fasta",
                    str(fasta),
                    "--mhc-predictor",
                    "random",
                    "--mhc-alleles",
                    "HLA-A*02:01",
                    "--output-csv",
                    "-",
                ]
            )
            == 0
        )
    output = capsys.readouterr()
    frame = pd.read_csv(StringIO(output.out))
    assert set(frame.source_sequence_name) == ({"good"} if mixed else set())
    assert "prediction row" in output.err


def test_multiple_models_receive_only_clean_inputs_with_one_warning():
    from topiary import ProteinInputWarning

    models = [ProteinModel(), ProteinModel()]
    models[1].method = "second"
    with pytest.warns(ProteinInputWarning) as caught:
        result = TopiaryPredictor(models=models).predict_from_named_sequences(
            {"bad": UNKNOWN, "good": CLEAN}
        )
    assert len(caught) == 1
    assert len(result) == 2 * (len(CLEAN) - 8)
    assert all(model.scans == [{"good": CLEAN}] for model in models)


class RejectingProteinModel(ProteinModel):
    """Reject X in peptides AND supplied context; count native inference batches."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batches = []

    def predict_with_flanks(self, peptides, n_flanks, c_flanks):
        self.batches.append(list(zip(peptides, n_flanks, c_flanks)))
        assert all("X" not in (p + n + c).upper() for p, n, c in self.batches[-1])
        return super().predict_with_flanks(peptides, n_flanks, c_flanks)


@pytest.mark.parametrize(
    "sequence, offsets",
    [
        ("XASIINFEKLGGG", [3, 4]),
        ("MASIINFEKLGGx", [0, 1]),
        ("MASIINFEKLGGGXXMASIINFEKLGGG", [0, 1, 2, 17, 18, 19]),
        ("XXXXXXXXXXXXX", []),
        ("AXA", []),
    ],
)
@pytest.mark.parametrize("cached", [False, True])
def test_windows_override_preserves_original_context_and_offline_replay(
    tmp_path, sequence, offsets, cached
):
    from topiary import CachedPredictor

    model = RejectingProteinModel()
    producer = CachedPredictor(fallback=model) if cached else model
    inputs = {"gene|transcript": sequence, "control": CLEAN}
    result = TopiaryPredictor(
        models=producer, protein_x_policy="windows"
    ).predict_from_named_sequences(inputs)
    selected = result[result.source_sequence_name == "gene|transcript"]
    assert sorted(selected.peptide_offset) == offsets
    assert len(result[result.source_sequence_name == "control"]) == len(CLEAN) - 8
    for row in selected.itertuples():
        assert row.peptide == sequence[row.peptide_offset : row.peptide_offset + 9]
        assert (
            row.n_flank == sequence[max(0, row.peptide_offset - 2) : row.peptide_offset]
        )
        assert row.c_flank == sequence[row.peptide_offset + 9 : row.peptide_offset + 11]
        assert row.value == 100 + sum(map(ord, row.n_flank + row.c_flank))
    assert len(model.batches) == (2 if offsets and not cached else 1)
    if cached:
        path = tmp_path / "selected.tsv"
        producer.save(path)
        restored = CachedPredictor.from_topiary_output(path)
        replay = TopiaryPredictor(
            models=restored, protein_x_policy="windows"
        ).predict_from_named_sequences(inputs)
        columns = [
            "source_sequence_name",
            "peptide_offset",
            "peptide",
            "n_flank",
            "c_flank",
            "value",
            "score",
        ]
        sort = ["source_sequence_name", "peptide_offset"]
        pd.testing.assert_frame_equal(
            result.sort_values(sort)[columns].reset_index(drop=True),
            replay.sort_values(sort)[columns].reset_index(drop=True),
            check_dtype=False,
        )


def test_asymmetric_context_lengths_repeats_and_batching():
    model = RejectingProteinModel(lengths=(8, 9))
    model.n_flank_length, model.c_flank_length = 3, 1
    inputs = {str(i): "X" + CLEAN + "X" + CLEAN for i in range(64)}
    result = TopiaryPredictor(
        models=model, protein_x_policy="windows"
    ).predict_from_named_sequences(inputs)
    assert len(model.batches) == 1
    assert len(model.batches[0]) < len(result)
    target = result[result.source_sequence_name == "0"]
    assert set(zip(target.peptide_offset, target.peptide_length)) == {
        (4, 8),
        (5, 8),
        (4, 9),
        (18, 8),
        (19, 8),
        (20, 8),
        (18, 9),
        (19, 9),
    }
    assert target.n_flank.str.len().eq(3).all()
    assert target.c_flank.str.len().max() == 1


@pytest.mark.parametrize("cached", [False, True])
def test_unknown_context_skips_x_inputs_without_inventing_termini(cached):
    from topiary import CachedPredictor, ProteinInputWarning

    model = ProteinModel()
    if cached:
        cache, _ = filled_cache(model, {"p": CLEAN})
        frame = cache.to_dataframe().drop(
            columns=["prediction_n_flank_limit", "prediction_c_flank_limit"]
        )
        producer = CachedPredictor(frame)
    else:
        model.protein_flank_lengths = None
        producer = model
    with pytest.warns(ProteinInputWarning, match="unknown protein flank limits"):
        result = TopiaryPredictor(
            models=producer, protein_x_policy="windows"
        ).predict_from_named_sequences({"p": UNKNOWN})
    assert result.empty


@pytest.mark.parametrize("unknown_in", ["target", "baseline"])
def test_windows_override_also_covers_fragment_and_wildtype_scans(unknown_in):
    target, baseline = (UNKNOWN, CLEAN) if unknown_in == "target" else (CLEAN, UNKNOWN)
    fragments = [
        ProteinFragment(
            fragment_id="test", sequence=target, reference_sequence=baseline
        ),
        ProteinFragment(
            fragment_id="control", sequence=CLEAN, reference_sequence=CLEAN
        ),
    ]
    result = TopiaryPredictor(
        models=RejectingProteinModel(), protein_x_policy="windows", predict_wt=True
    ).predict_from_fragments(fragments)
    assert result[result.fragment_id == "control"].wt_value.notna().all()
    test = result[result.fragment_id == "test"]
    assert test.peptide_offset.tolist() == (
        [3, 4] if unknown_in == "target" else [0, 1, 2, 3, 4]
    )
    assert test.loc[test.peptide_offset.isin([3, 4]), "wt_value"].notna().all()
    assert test.loc[~test.peptide_offset.isin([3, 4]), "wt_value"].isna().all()


def test_windows_cache_gaps_remain_strict_or_report_original_input():
    from topiary import (
        CachedPredictor,
        CachedPredictorCoverageError,
        PartialPredictionWarning,
    )

    live = TopiaryPredictor(models=ProteinModel(), protein_x_policy="windows")
    stored = live.predict_from_named_sequences({"x": UNKNOWN, "good": CLEAN})
    inputs = {"missing": "XAAAAAAAAAAAA", "x": UNKNOWN, "good": CLEAN}
    cache = CachedPredictor(stored)
    with pytest.raises(CachedPredictorCoverageError):
        TopiaryPredictor(
            models=cache, protein_x_policy="windows"
        ).predict_from_named_sequences(inputs)
    misses = []
    with pytest.warns(PartialPredictionWarning):
        result = TopiaryPredictor(
            models=cache, protein_x_policy="windows", cache_miss_handler=misses.append
        ).predict_from_named_sequences(inputs)
    assert set(result.source_sequence_name) == {"x", "good"}
    assert [m["source_sequence_name"] for m in misses] == ["missing"]


@pytest.mark.parametrize(
    "error", [ValueError("model configuration invalid"), RuntimeError("backend failed")]
)
def test_windows_override_does_not_hide_other_model_errors(error):
    model = ProteinModel()

    def fail(*args, **kwargs):
        raise error

    model.predict_with_flanks = fail
    with pytest.raises(type(error)) as caught:
        TopiaryPredictor(
            models=model, protein_x_policy="windows"
        ).predict_from_named_sequences({"p": UNKNOWN})
    assert caught.value is error


@pytest.mark.parametrize("policy", ["skip", "windows", "allow"])
def test_exact_peptide_inputs_keep_their_existing_producer_contract(policy):
    result = TopiaryPredictor(
        models=ProteinModel(), protein_x_policy=policy
    ).predict_from_named_peptides({"peptide": "XASIINFEK"})
    assert result.peptide.tolist() == ["XASIINFEK"]


def test_bad_policy_is_rejected_before_prediction():
    with pytest.raises(ValueError, match="protein_x_policy"):
        TopiaryPredictor(models=ProteinModel(), protein_x_policy="guess")


def test_selected_cache_fallback_fills_only_requested_windows_and_replays(tmp_path):
    from topiary import CachedPredictor

    model = RejectingProteinModel()
    cache = CachedPredictor(fallback=model)
    inputs = {"unknown": UNKNOWN}
    result = TopiaryPredictor(
        models=cache, protein_x_policy="windows"
    ).predict_from_named_sequences(inputs)
    assert result.peptide_offset.tolist() == [3, 4]
    assert len(model.batches) == 1
    assert model.scans == []
    assert len(cache.to_dataframe()) == 2
    path = tmp_path / "selected.parquet"
    cache.save(path)
    replay = TopiaryPredictor(
        models=CachedPredictor.from_topiary_output(path), protein_x_policy="windows"
    )
    pd.testing.assert_frame_equal(
        replay.predict_from_named_sequences(inputs), result, check_dtype=False
    )


def test_peptide_only_model_keeps_clean_windows_next_to_x():
    model = RejectingProteinModel()
    model.uses_flanking_sequences = False
    result = TopiaryPredictor(
        models=model, protein_x_policy="windows"
    ).predict_from_named_sequences({"p": UNKNOWN})
    assert result.peptide_offset.tolist() == [1, 2, 3, 4]
    assert result.n_flank.eq("").all() and result.c_flank.eq("").all()


def test_dataframe_only_producer_uses_original_selected_context():
    from types import SimpleNamespace

    native = RejectingProteinModel()
    model = SimpleNamespace(
        default_peptide_lengths=native.default_peptide_lengths,
        uses_flanking_sequences=True,
        protein_flank_lengths=native.protein_flank_lengths,
        kind_support=native.kind_support,
        alleles=native.alleles,
        predict_dataframe=native.predict_dataframe,
        predict_proteins_dataframe=native.predict_proteins_dataframe,
    )
    result = TopiaryPredictor(
        models=model, protein_x_policy="windows"
    ).predict_from_named_sequences({"p": UNKNOWN})
    assert result.peptide_offset.tolist() == [3, 4]
    assert result.n_flank.tolist() == ["AS", "SI"]


def test_multiple_model_contexts_select_independent_windows():
    short, long = RejectingProteinModel(), RejectingProteinModel()
    short.uses_flanking_sequences = False
    short.method = "peptide-only"
    result = TopiaryPredictor(
        models=[short, long], protein_x_policy="windows"
    ).predict_from_named_sequences({"p": UNKNOWN})
    assert result.groupby("prediction_method_name").peptide_offset.apply(
        list
    ).to_dict() == {"peptide-only": [1, 2, 3, 4], "protein-test": [3, 4]}


def test_cached_sparse_kinds_and_distinct_declared_contexts_are_preserved():
    from topiary import (
        CachedPredictor,
        annotate_protein_prediction_context,
        from_predictions,
    )

    inputs = {"p": "XMNPQRSTVWYACDEFGHIK"}
    frame = []
    for length, kind in [(1, "pMHC_affinity"), (3, "pMHC_presentation")]:
        model = ProteinModel(modes={kind: "single_allele"})
        model.flank_length = length
        raw = annotate_protein_prediction_context(
            model.predict_proteins_dataframe(inputs), model
        )
        frame.append(
            from_predictions(raw.iloc[::2] if kind == "pMHC_presentation" else raw)
        )
    cache = CachedPredictor(pd.concat(frame, ignore_index=True))
    assert cache.protein_window_flank_limits() == (3, 3)
    from topiary import describe_prediction_models

    (inventory,) = describe_prediction_models(TopiaryPredictor(models=cache))
    assert inventory["protein_flank_lengths"] is None
    result = TopiaryPredictor(
        models=cache, protein_x_policy="windows"
    ).predict_from_named_sequences(inputs)
    assert result.peptide_offset.min() == 4
    assert set(result.kind) == {"pMHC_affinity", "pMHC_presentation"}
    assert len(result[result.kind == "pMHC_affinity"]) > len(
        result[result.kind == "pMHC_presentation"]
    )
    assert result.groupby("kind").prediction_n_flank_limit.unique().map(
        list
    ).to_dict() == {"pMHC_affinity": [1], "pMHC_presentation": [3]}


@pytest.mark.parametrize(
    "windows",
    [[("p", -1, 9)], [("missing", 0, 9)], [("p", 1, 99)], [("p", 0, 9), ("p", 0, 9)]],
)
def test_selected_window_geometry_is_validated(windows):
    from topiary import CachedPredictor, predict_protein_windows

    for model in [ProteinModel(), CachedPredictor(fallback=ProteinModel())]:
        with pytest.raises(ValueError, match="Selected protein window"):
            predict_protein_windows(model, {"p": CLEAN}, windows)
