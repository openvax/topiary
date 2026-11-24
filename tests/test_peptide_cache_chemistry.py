"""Chemistry-aware peptide-level cache identity and round-trip tests (#569)."""

import numpy as np
import pandas as pd
import pytest
from mhctools import CavacoHalfLife, Kind, MeasurementContext, PeptideContext, PeptideInput, PeptideResult, Prediction

from topiary import (
    CachedPredictor,
    CachedPredictorCoverageError,
    PREDICTION_CONTEXT_COLUMNS,
    PREDICTION_KEY_COLUMNS,
    TopiaryPredictor,
    TopiaryResult,
    apply_sort,
    coerce_peptide_input,
    conflicting_predictions,
    from_predictions,
    from_wide,
    peptide_chemical_identity,
    peptide_inference_identity,
    peptide_record_identity,
    predict_peptide_occurrences,
    prediction_matches_protein_context,
    prediction_peptide_input_matches,
    read_csv,
    read_tsv,
    to_csv,
    to_tsv,
    to_wide,
)
from topiary.ranking import Field, peptide_view


class _ChemistryHalfLifePredictor:
    """Deterministic whole-peptide predictor sensitive to terminal chemistry and context."""

    predictor_name = "chem_half_life"
    prediction_method_name = "chem_half_life"
    predictor_version = "1.0"
    alleles = ()
    default_peptide_lengths = (8, 9)

    def __init__(self):
        self.calls = []

    def kind_support(self):
        return {
            Kind.peptide_half_life: {
                "mhc_dependence": "none",
                "mhc_class": "none",
            }
        }

    def predict(self, peptides):
        items = [
            p if isinstance(p, PeptideInput) else PeptideInput(str(p).strip().upper())
            for p in peptides
        ]
        self.calls.append(items)
        results = []
        for item in items:
            if item.n_term == "unknown":
                ctx = MeasurementContext(
                    estimate_type="ml_predicted",
                    analyte="parent peptide",
                    status="unsupported",
                    detail="unknown N-terminal chemistry cannot be scored",
                )
                pred = Prediction(
                    kind=Kind.peptide_half_life,
                    peptide=item.sequence,
                    score=None,
                    value=None,
                    predictor_name=self.predictor_name,
                    predictor_version=self.predictor_version,
                    measurement_context=ctx,
                    peptide_input=item,
                )
            else:
                hours = 1.0
                if item.n_term == "acetylated":
                    hours += 4.0
                if item.c_term == "amidated":
                    hours += 2.0
                if item.attachments:
                    hours += 10.0 * len(item.attachments)
                if item.context is not None and item.context.matrix == "plasma":
                    hours += 0.5
                ctx = MeasurementContext(
                    estimate_type="ml_predicted",
                    analyte="parent peptide",
                    status="available",
                    unit="hours",
                    transform="linear",
                    score_semantics="synthetic half-life hours",
                )
                pred = Prediction(
                    kind=Kind.peptide_half_life,
                    peptide=item.sequence,
                    score=hours,
                    value=hours,
                    predictor_name=self.predictor_name,
                    predictor_version=self.predictor_version,
                    measurement_context=ctx,
                    peptide_input=item,
                )
            results.append(PeptideResult(preds=(pred,)))
        return results

    def predict_dataframe(self, peptides, **_kwargs):
        rows = [
            pred.to_row()
            for result in self.predict(peptides)
            for pred in result.preds
        ]
        return pd.DataFrame(rows)


def test_public_identity_helpers_match_mhctools_contract():
    free_occ1 = PeptideInput(
        "SIINFEKL", occurrence_id="occ1", source_sequence_name="protA", source_start=10,
    )
    free_occ2 = PeptideInput(
        "SIINFEKL", occurrence_id="occ2", source_sequence_name="protA", source_start=50,
    )
    acetylated = PeptideInput(
        "SIINFEKL", n_term="acetylated", occurrence_id="occ1", source_start=10,
    )
    amidated = PeptideInput("SIINFEKL", c_term="amidated")
    attached = PeptideInput("SIINFEKL", attachments=(("K7", "PEG2"),))
    plasma = PeptideInput("SIINFEKL", context=PeptideContext(matrix="plasma"))

    # Missing context stays None, never coerced to an identity or score.
    for missing in (None, np.nan, pd.NA, "", "   "):
        assert coerce_peptide_input(missing) is None
        assert peptide_chemical_identity(missing) is None
        assert peptide_inference_identity(missing) is None
        assert peptide_record_identity(missing) is None

    # Exact parity with mhctools.PeptideInput properties.
    for item in (free_occ1, free_occ2, acetylated, amidated, attached, plasma):
        assert coerce_peptide_input(item) == item
        assert coerce_peptide_input(item.to_dict()) == item
        assert peptide_chemical_identity(item) == item.chemical_identity_sha256
        assert peptide_inference_identity(item) == item.inference_identity_sha256
        assert peptide_record_identity(item) == item.record_identity_sha256

    # Repeated occurrences share chemical/inference identity but keep distinct record identity.
    assert peptide_chemical_identity(free_occ1) == peptide_chemical_identity(free_occ2)
    assert peptide_inference_identity(free_occ1) == peptide_inference_identity(free_occ2)
    assert peptide_record_identity(free_occ1) != peptide_record_identity(free_occ2)

    # Distinct terminal chemistry, attachments, and assay context produce distinct inference identities.
    inference_ids = {
        peptide_inference_identity(x)
        for x in (free_occ1, acetylated, amidated, attached, plasma)
    }
    assert len(inference_ids) == 5
    # Plasma context shares chemical identity with free_occ1, but differs in inference identity.
    assert peptide_chemical_identity(free_occ1) == peptide_chemical_identity(plasma)
    assert peptide_inference_identity(free_occ1) != peptide_inference_identity(plasma)

    # Public schema constants expose the chemistry key and context columns.
    assert "peptide_inference_identity" in PREDICTION_KEY_COLUMNS
    assert "peptide_input" in PREDICTION_CONTEXT_COLUMNS


def test_cache_distinguishes_terminal_chemistry_attachments_and_context():
    model = _ChemistryHalfLifePredictor()
    free = PeptideInput("SIINFEKL", occurrence_id="occ1", source_start=10)
    acetylated = PeptideInput("SIINFEKL", n_term="acetylated", occurrence_id="occ1", source_start=10)
    attached = PeptideInput("SIINFEKL", attachments=(("K7", "biotin"),))
    plasma = PeptideInput("SIINFEKL", context=PeptideContext(matrix="plasma"))
    unknown_n = PeptideInput("SIINFEKL", n_term="unknown")
    amidated = PeptideInput("SIINFEKL", c_term="amidated")

    df = from_predictions(model.predict_dataframe([free, acetylated, attached, plasma, unknown_n]))
    cache = CachedPredictor(df)

    # Each chemical/context form returns its exact measurement without collision.
    assert cache.predict_peptides_dataframe([free]).value.tolist() == [1.0]
    assert cache.predict_peptides_dataframe([acetylated]).value.tolist() == [5.0]
    assert cache.predict_peptides_dataframe([attached]).value.tolist() == [11.0]
    assert cache.predict_peptides_dataframe([plasma]).value.tolist() == [1.5]

    # Unsupported/missing context remains NaN, never coerced to an unfavourable score.
    unknown_out = cache.predict_peptides_dataframe([unknown_n])
    assert len(unknown_out) == 1
    assert pd.isna(unknown_out.score.iloc[0])
    assert pd.isna(unknown_out.value.iloc[0])
    assert unknown_out.measurement_context.iloc[0]["status"] == "unsupported"

    # Uncached terminal chemistry misses rather than borrowing free-terminal score.
    with pytest.raises(CachedPredictorCoverageError):
        cache.predict_peptides_dataframe([amidated])

    # Plain string shorthand matches natural unmodified free/free (context=None) only.
    plain_out = cache.predict_peptides_dataframe(["SIINFEKL"])
    assert plain_out.value.tolist() == [1.0]

    # Cache containing ONLY modified chemistry does NOT answer a plain string query.
    ac_only = CachedPredictor(from_predictions(model.predict_dataframe([acetylated])))
    with pytest.raises(CachedPredictorCoverageError):
        ac_only.predict_peptides_dataframe(["SIINFEKL"])


def test_fallback_resolves_missing_chemistry_and_reuses_across_occurrences():
    model = _ChemistryHalfLifePredictor()
    free_occ1 = PeptideInput(
        "SIINFEKL", occurrence_id="occ1", source_sequence_name="protA", source_start=10,
    )
    free_occ2 = PeptideInput(
        "SIINFEKL", occurrence_id="occ2", source_sequence_name="protA", source_start=50,
    )
    amidated = PeptideInput(
        "SIINFEKL", c_term="amidated", occurrence_id="occ1", source_sequence_name="protA", source_start=10,
    )

    seed = from_predictions(model.predict_dataframe([free_occ1]))
    model.calls.clear()
    cache = CachedPredictor(seed, fallback=model)

    # Repeated occurrence at a different coordinate reuses cached inference while
    # preserving each occurrence's own source_start and record_identity_sha256.
    repeated = cache.predict_peptides_dataframe([free_occ1, free_occ2])
    assert model.calls == []
    assert repeated.value.tolist() == [1.0, 1.0]
    assert repeated.peptide_offset.tolist() == [10, 50]
    assert [peptide_record_identity(x) for x in repeated.peptide_input] == [
        free_occ1.record_identity_sha256,
        free_occ2.record_identity_sha256,
    ]

    # New terminal chemistry triggers fallback with the exact PeptideInput and caches it.
    res = cache.predict_peptides_dataframe([amidated])
    assert len(model.calls) == 1
    assert model.calls[0] == [amidated]
    assert res.value.tolist() == [3.0]

    # Subsequent query hits cache without calling fallback again.
    res_again = cache.predict_peptides_dataframe([amidated])
    assert len(model.calls) == 1
    assert res_again.value.tolist() == [3.0]


def test_tumor_source_context_and_administered_chemistry_never_substitute():
    # Tumor-source processing row (with protein flanks, no administered PeptideInput)
    tumor_row = dict(
        peptide="SIINFEKL", allele="", peptide_length=8,
        kind="antigen_processing", score=0.75, affinity=np.nan,
        percentile_rank=np.nan, value=0.75,
        prediction_method_name="proc_model", predictor_version="1.0",
        n_flank="AAA", c_flank="GGG", prediction_flanks_supplied=True,
        prediction_mhc_dependence="none",
        prediction_n_flank_limit=3, prediction_c_flank_limit=3,
        peptide_input=None,
    )
    # Administered acetylated row for the same sequence (no tumor flanks)
    ac_input = PeptideInput("SIINFEKL", n_term="acetylated")
    admin_row = dict(
        peptide="SIINFEKL", allele="", peptide_length=8,
        kind="antigen_processing", score=0.25, affinity=np.nan,
        percentile_rank=np.nan, value=0.25,
        prediction_method_name="proc_model", predictor_version="1.0",
        n_flank="", c_flank="", prediction_flanks_supplied=False,
        prediction_mhc_dependence="none",
        prediction_n_flank_limit=pd.NA, prediction_c_flank_limit=pd.NA,
        peptide_input=ac_input.to_dict(),
    )

    cache = CachedPredictor(pd.DataFrame([tumor_row, admin_row]))

    # Tumor-source protein scan matches ONLY the tumor-source row, never administered chemistry.
    assert prediction_matches_protein_context(cache._df.iloc[0], "AAASIINFEKLGGG", 3)
    assert not prediction_matches_protein_context(cache._df.iloc[1], "AAASIINFEKLGGG", 3)
    scan = cache.predict_proteins_dataframe(
        {"tumor_prot": "AAASIINFEKLGGG"},
        windows=[("tumor_prot", 3, 8)],
    )
    assert len(scan) == 1
    assert scan.score.tolist() == [0.75]
    assert pd.isna(scan.peptide_input.iloc[0])

    # Explicit administered query matches ONLY the administered row, never tumor-source row.
    admin_out = cache.predict_contextual_peptides_dataframe(
        ["SIINFEKL"], peptide_inputs=[ac_input],
    )
    assert len(admin_out) == 1
    assert admin_out.score.tolist() == [0.25]
    assert coerce_peptide_input(admin_out.peptide_input.iloc[0]) == ac_input

    # Tumor-source contextual query matches ONLY the tumor-source row.
    tumor_out = cache.predict_contextual_peptides_dataframe(
        ["SIINFEKL"], n_flanks=["AAA"], c_flanks=["GGG"],
    )
    assert len(tumor_out) == 1
    assert tumor_out.score.tolist() == [0.75]

    # Combining tumor flanks with acetylated chemistry misses rather than borrowing either.
    with pytest.raises(CachedPredictorCoverageError):
        cache.predict_contextual_peptides_dataframe(
            ["SIINFEKL"], n_flanks=["AAA"], c_flanks=["GGG"], peptide_inputs=[ac_input],
        )

    # Legacy cache without peptide_input still answers plain string queries,
    # never explicit PeptideInput queries.
    legacy = CachedPredictor(pd.DataFrame([tumor_row]))
    assert prediction_peptide_input_matches(legacy._df.iloc[0], "SIINFEKL")
    assert not prediction_peptide_input_matches(legacy._df.iloc[0], PeptideInput("SIINFEKL"))
    assert not prediction_peptide_input_matches(legacy._df.iloc[0], ac_input)


def test_conflicting_and_agreeing_peptide_inputs_across_cache_doors():
    occ1 = PeptideInput("SIINFEKL", occurrence_id="occ1", source_start=10)
    occ2 = PeptideInput("SIINFEKL", occurrence_id="occ2", source_start=50)
    ac = PeptideInput("SIINFEKL", n_term="acetylated", occurrence_id="occ1", source_start=10)

    def _row(item, score):
        return dict(
            peptide="SIINFEKL", allele="", peptide_length=8,
            kind="peptide_half_life", score=score, value=score,
            affinity=np.nan, percentile_rank=np.nan,
            prediction_method_name="chem", predictor_version="1.0",
            peptide_input=item.to_dict(),
        )

    # Same inference identity, different occurrence provenance, agreeing score -> no conflict.
    agreeing = pd.DataFrame([_row(occ1, 2.0), _row(occ2, 2.0)])
    assert conflicting_predictions(CachedPredictor(agreeing)._df).empty
    merged = CachedPredictor.concat([
        CachedPredictor(pd.DataFrame([_row(occ1, 2.0)])),
        CachedPredictor(pd.DataFrame([_row(occ2, 2.0)])),
    ])
    assert len(merged._df) == 2

    # Same inference identity, disagreeing score -> conflict in both constructor and concat.
    with pytest.raises(ValueError, match="peptide_inference_identity"):
        CachedPredictor(pd.DataFrame([_row(occ1, 2.0), _row(occ2, 9.0)]))
    with pytest.raises(ValueError, match="peptide_inference_identity"):
        CachedPredictor.concat([
            CachedPredictor(pd.DataFrame([_row(occ1, 2.0)])),
            CachedPredictor(pd.DataFrame([_row(occ2, 9.0)])),
        ])

    # Different chemistry, different scores -> NOT a conflict.
    distinct = CachedPredictor.concat([
        CachedPredictor(pd.DataFrame([_row(occ1, 2.0)])),
        CachedPredictor(pd.DataFrame([_row(ac, 9.0)])),
    ])
    assert len(distinct._df) == 2

    # Mismatched sequence or peptide_inference_identity is rejected.
    bad_seq = _row(occ1, 2.0)
    bad_seq["peptide"] = "AAAAAAAA"
    with pytest.raises(ValueError, match="disagrees with peptide"):
        CachedPredictor(pd.DataFrame([bad_seq]))

    bad_id = _row(occ1, 2.0)
    bad_id["peptide_inference_identity"] = "0" * 64
    with pytest.raises(ValueError, match="disagrees with peptide_input"):
        CachedPredictor(pd.DataFrame([bad_id]))


@pytest.mark.parametrize("suffix", ["tsv", "tsv.gz", "csv", "csv.gz", "parquet"])
def test_roundtrip_serialization_and_wide_tables_preserve_chemistry_and_occurrences(
    suffix, tmp_path,
):
    model = _ChemistryHalfLifePredictor()
    occ1 = PeptideInput(
        "SIINFEKL", occurrence_id="occ1", source_sequence_name="protA", source_start=10,
    )
    occ2 = PeptideInput(
        "SIINFEKL", occurrence_id="occ2", source_sequence_name="protA", source_start=50,
    )
    ac = PeptideInput(
        "SIINFEKL", n_term="acetylated", attachments=(("K7", "PEG"),),
        occurrence_id="occ1", source_sequence_name="protA", source_start=10,
        context=PeptideContext(matrix="plasma", conditions=(("pH", "7.4"),)),
    )
    unknown_n = PeptideInput(
        "SIINFEKL", n_term="unknown", occurrence_id="occ3", source_start=80,
    )

    # End-to-end through TopiaryPredictor.predict_from_named_peptides
    predictor = TopiaryPredictor(models=[model])
    predicted = predictor.predict_from_named_peptides({
        "first": occ1,
        "second": occ2,
        "modified": ac,
        "unknown": unknown_n,
    })
    assert len(predicted) == 4
    assert predicted.peptide_offset.tolist() == [10, 50, 10, 80]
    assert predicted.value.iloc[:3].tolist() == [1.0, 1.0, 15.5]
    assert pd.isna(predicted.value.iloc[3])

    # DSL ranking keeps chemically distinct forms of the same sequence as distinct groups.
    ranked = apply_sort(predicted.iloc[:3], peptide_view(Field(Kind.peptide_half_life, "value")))
    assert ranked.source_sequence_name.iloc[0] == "modified"
    assert len(ranked) == 3

    # Wide/long round-trip keeps all 4 distinct occurrences & chemistries separate.
    wide = to_wide(predicted)
    assert len(wide) == 4
    back_long = from_wide(wide)
    assert len(back_long) == 4
    assert [peptide_record_identity(x) for x in back_long.peptide_input] == [
        occ1.record_identity_sha256,
        occ2.record_identity_sha256,
        ac.record_identity_sha256,
        unknown_n.record_identity_sha256,
    ]
    assert pd.isna(back_long.loc[back_long.source_sequence_name == "unknown", "value"].iloc[0])

    # Cache save/reload round-trip across TSV, TSV.gz, CSV, CSV.gz, and Parquet.
    cache = CachedPredictor(predicted)
    path = tmp_path / f"chem_cache.{suffix}"
    cache.save(path)
    restored = CachedPredictor.from_topiary_output(path)

    replayed = TopiaryPredictor(models=restored).predict_from_named_peptides({
        "first": occ1,
        "second": occ2,
        "modified": ac,
        "unknown": unknown_n,
    })
    assert replayed.value.iloc[:3].tolist() == [1.0, 1.0, 15.5]
    assert pd.isna(replayed.value.iloc[3])
    assert replayed.peptide_offset.tolist() == [10, 50, 10, 80]
    assert [coerce_peptide_input(x) for x in replayed.peptide_input] == [
        occ1, occ2, ac, unknown_n,
    ]
    assert [peptide_inference_identity(x) for x in replayed.peptide_input] == [
        occ1.inference_identity_sha256,
        occ2.inference_identity_sha256,
        ac.inference_identity_sha256,
        unknown_n.inference_identity_sha256,
    ]

    # Topiary TSV/CSV direct table round-trip also preserves peptide_input & missing values.
    if suffix in ("tsv", "csv"):
        table_path = tmp_path / f"table.{suffix}"
        (to_tsv if suffix == "tsv" else to_csv)(TopiaryResult(predicted), table_path)
        table_back = (read_tsv if suffix == "tsv" else read_csv)(table_path).df
        assert [coerce_peptide_input(x) for x in table_back.peptide_input] == [
            occ1, occ2, ac, unknown_n,
        ]
        assert pd.isna(table_back.loc[table_back.source_sequence_name == "unknown", "value"].iloc[0])


def test_predict_peptide_occurrences_and_real_cavaco_cache_roundtrip(tmp_path):
    # Exercise real mhctools.CavacoHalfLife (offline, zero downloads) alongside
    # predict_peptide_occurrences and cache replay.
    cavaco = CavacoHalfLife()
    free_occ1 = PeptideInput("SIINFEKL", occurrence_id="occ1", source_start=10)
    free_occ2 = PeptideInput("SIINFEKL", occurrence_id="occ2", source_start=50)
    ac = PeptideInput("SIINFEKL", n_term="acetylated", occurrence_id="occ3", source_start=10)

    native = from_predictions(
        pd.concat([
            r.to_dataframe()
            for r in cavaco.predict([free_occ1, ac], on_unsupported="record")
        ], ignore_index=True)
    )
    cache = CachedPredictor(native)
    path = tmp_path / "cavaco.parquet"
    cache.save(path)
    reloaded = CachedPredictor.from_topiary_output(path)

    occ_table = pd.DataFrame([
        dict(prediction_id="p1", peptide="SIINFEKL", peptide_offset=10, peptide_input=free_occ1),
        dict(prediction_id="p2", peptide="SIINFEKL", peptide_offset=50, peptide_input=free_occ2),
        dict(prediction_id="p3", peptide="SIINFEKL", peptide_offset=10, peptide_input=ac),
    ])
    out = predict_peptide_occurrences(occ_table, reloaded, use_flanks=False)
    assert len(out) == 3
    assert out.loc[out.prediction_id == "p1", "value"].iloc[0] == pytest.approx(
        out.loc[out.prediction_id == "p2", "value"].iloc[0]
    )
    assert pd.isna(out.loc[out.prediction_id == "p3", "value"].iloc[0])
    assert out.loc[out.prediction_id == "p3", "measurement_context"].iloc[0]["status"] == "unsupported"
    assert [peptide_record_identity(x) for x in out.peptide_input] == [
        free_occ1.record_identity_sha256,
        free_occ2.record_identity_sha256,
        ac.record_identity_sha256,
    ]
