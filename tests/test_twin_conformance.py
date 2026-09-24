"""Functions that answer the same question must answer it the same way.

The defect class this exists for, three instances of it in one release
cycle:

- ``attach_rna_evidence`` let pandas align a misaligned Series, producing
  an all-null column; ``attach_dna_evidence`` assigned positionally,
  producing a misaligned one. Both silent, in opposite directions.
- pVACseq's aggregated and all-epitopes branches attached read evidence
  under different vocabularies.
- ``CachedPredictor.concat`` raises on any duplicate key while the
  constructor accepts contradictory rows silently (topiary#231).

None of these were caught by tests, because every test exercised one
door. A test that exercises one door can never see a divergence; only a
test that drives both through the same battery can.

**Adding a twin here is the point.** When you write a function that
answers a question an existing function already answers -- an assay
variant, a format branch, a second constructor -- add the pair to
``TWINS`` rather than trusting review to notice the next divergence.
"""

import dataclasses
from dataclasses import dataclass, field
from importlib.metadata import requires
from types import SimpleNamespace
from typing import Callable, Dict, Tuple

import pandas as pd
from osteosarc import parse_variants
from scripts.osteosarc_variant_audit import variant_inventory
import pytest
from packaging.requirements import Requirement
from mhctools import RandomBindingPredictor

from topiary.evidence import (
    RENAMED_COLUMNS,
    attach_dna_evidence,
    attach_rna_evidence,
)
from topiary import (
    APPROXIMATED, MEASURED, CachedPredictor, ProteinFragment, TopiaryPredictor, from_predictions, fragments_from_variants,
    read_fragments, read_pvacseq, write_fragments, unique_fragments,
    describe_isovar_result, fragment_from_isovar_result, TopiaryResult,
    to_tsv, to_csv, read_tsv, read_csv,
    combine_sources, rank_candidates, evaluate_scores,
)
from topiary.io_isovar import _check_isovar
from topiary.sources import _check_pirlygenes
import topiary.optional_dependencies as optional_dependencies
from .pvacseq_corpus_helpers import REPORTS as PVACSEQ_REPORTS, ROOT as PVACSEQ_ROOT
from topiary import build_sv_interest_report
from topiary.cli.sv_interest import main as sv_interest_main


@dataclass(frozen=True)
class Twin:
    """Two callables that must behave identically on the args they share.

    Parameters
    ----------
    name : str
        Shown in test ids.
    left, right : callable
        Both take a DataFrame first and the shared arguments by keyword.
    shared : dict
        Left's argument name -> right's name for the same quantity. Names
        differ legitimately (``overlapping`` vs ``depth``); what must not
        differ is what they do with the value.
    """

    name: str
    left: Callable
    right: Callable
    shared: Dict[str, str] = field(default_factory=dict)

    def calls(self, arg, value):
        """``((fn, kwargs), (fn, kwargs))`` passing *value* as *arg*."""
        return (
            (self.left, {arg: value}),
            (self.right, {self.shared[arg]: value}),
        )


TWINS = (
    Twin(
        name="rna/dna evidence",
        left=attach_rna_evidence,
        right=attach_dna_evidence,
        # RNA calls the coverage argument `overlapping`, DNA calls it
        # `depth`; they are the same quantity per assay.
        shared={"overlapping": "depth", "vaf": "vaf"},
    ),
)


# Both file formats expose a function and a result method. Drive all four
# through the same metadata validation and write/read battery in test_io.py.
DELIMITED_IO_TWINS = (
    ("tsv", to_tsv, TopiaryResult.to_tsv, read_tsv),
    ("csv", to_csv, TopiaryResult.to_csv, read_csv),
)


# Candidate ranking must use the same feature/prediction semantics as the
# lower-level DSL scorer that downstream consumers already call.
CANDIDATE_SCORE_TWINS = (rank_candidates, evaluate_scores)

# The CLI serializes the same exhaustive SV evidence policy as the public API.
# Driven together in test_consumer_workflows, including absent protein rows.
SV_INTEREST_REPORT_TWINS = (build_sv_interest_report, sv_interest_main)


def _measurement_scores(frame, expression, **options):
    from topiary import parse
    return evaluate_scores(frame, parse(expression), **options)


def _measurement_filter(frame, expression, **options):
    from topiary import apply_filter, parse
    return apply_filter(frame, parse(f"({expression}) < 55"), **options)


def _measurement_sort(frame, expression, **options):
    from topiary import apply_sort, parse
    return apply_sort(frame, [parse(expression)], sort_direction="asc", **options)


def _result_measurement_filter(frame, expression, **options):
    return TopiaryResult(frame).filter_by(f"({expression}) < 55", **options).df


def _result_measurement_sort(frame, expression, **options):
    return TopiaryResult(frame).sort_by(expression, **options).df


# These doors must agree on conflicting, equal and missing measurements.
# The composed battery lives in test_consumer_workflows.py.
DSL_MEASUREMENT_TWINS = (
    ("score", _measurement_scores),
    ("filter", _measurement_filter),
    ("sort", _measurement_sort),
    ("result_filter", _result_measurement_filter),
    ("result_sort", _result_measurement_sort),
)


@pytest.mark.parametrize("expression", ["affinity.value", "n_rna_alt / affinity.value", "n_rna_alt"])
@pytest.mark.parametrize("missing", [False, True])
def test_candidate_ranking_and_dsl_scoring_agree(expression, missing):
    from topiary import parse
    from .test_candidate_tables import source

    frame = source(n_rna_alt=[5, None] if missing else [5, 15])
    combined = combine_sources({"original": frame}, sample_name="p")
    rank, score = CANDIDATE_SCORE_TWINS
    ranked = rank(combined, expression)
    expected = dict(zip(combined.df.candidate_id, score(combined.df, parse(expression))))
    pd.testing.assert_series_equal(
        ranked.candidate_score,
        ranked.candidate_id.map(expected), check_names=False, check_dtype=False,
    )


@pytest.mark.parametrize("axis,labels", [
    ("prediction_method_name", ["first", "second"]),
    ("predictor_version", ["1", "2"]),
    ("kind", ["pMHC_affinity", "pMHC_presentation"]),
])
@pytest.mark.parametrize("values", [[9., 1.], [None, 1.], [1., 1.], [None, None]])
def test_candidate_ranking_and_dsl_reject_ambiguous_columns(axis, labels, values):
    from topiary import Column, parse
    from .test_candidate_tables import source

    original = source(peptides=("SIINFEKL", "SIINFEKL"), values=(50., 50.),
                      percentile_rank=values, original_rank=1, n_rna_alt=5,
                      **{axis: labels}).df
    rank, score = CANDIDATE_SCORE_TWINS
    for ordered in (original, original.iloc[::-1]):
        combined = combine_sources({"original": ordered}, sample_name="p")
        assert combined.df.source_observation_id.nunique() == 1
        for expression in (parse("percentile_rank"), Column("percentile_rank")):
            if ordered.percentile_rank.nunique() > 1:
                with pytest.raises(ValueError, match="Conflicting prediction measurements"):
                    score(combined.df, expression)
                for policy in ("error", "best", "worst"):
                    with pytest.raises(ValueError, match="Conflicting prediction measurements"):
                        rank(combined, expression, duplicates=policy)
            else:
                expected = ordered.percentile_rank.min()
                scored = score(combined.df, expression)
                ranked = rank(combined, expression)
                if pd.isna(expected):
                    assert scored.isna().all()
                    assert ranked.candidate_score.isna().all()
                    assert ranked.ranking_status.eq("missing_score").all()
                else:
                    assert scored.eq(expected).all()
                    assert ranked.candidate_score.eq(expected).all()
                    assert ranked.candidate_rank.tolist() == [1]


# Real aggregated/all-epitopes doors, paired by original run and MHC view.
PVACSEQ_CORPUS_TWINS = tuple(
    (aggregate, next(report for report in PVACSEQ_REPORTS
                     if report["pair"] == aggregate["pair"] and
                     report["category"] == "all_epitopes"))
    for aggregate in PVACSEQ_REPORTS if aggregate["category"] == "aggregated"
)


# Both source-flavor doors receive the same new RNA evidence; neither can
# reinterpret counts/TPM or attach a different transcript's annotation.
PVACSEQ_RNA_OVERLAY_TWINS = PVACSEQ_CORPUS_TWINS

# Both report-reader doors must preserve identical position semantics through
# fragment conversion and the predictor. The composed battery lives in
# test_consumer_workflows.py; fixtures are built by report_geometry_helpers.
PVACSEQ_MUTATION_GEOMETRY_TWINS = ("aggregated", "all_epitopes")


@pytest.mark.parametrize("aggregate,all_epitopes", PVACSEQ_RNA_OVERLAY_TWINS,
                         ids=[a["pair"] for a, _ in PVACSEQ_RNA_OVERLAY_TWINS])
def test_pvacseq_flavors_agree_on_new_rna_overlay(aggregate, all_epitopes):
    import pandas as pd
    from .osteosarc_overlay_helpers import add_rna

    frames = []
    for report in (aggregate, all_epitopes):
        path = PVACSEQ_ROOT / report["file"]
        result = add_rna(read_pvacseq(path), pd.read_csv(path, sep="\t"))
        frames.append(result.df)
    keys = ["rna_t2_allele_key", "transcript"]
    columns = [c for c in frames[0] if c.startswith("rna_t2_") and c not in keys]
    left, right = [df[keys + columns].drop_duplicates().set_index(keys).sort_index()
                   for df in frames]
    pd.testing.assert_frame_equal(left, right.loc[left.index], check_dtype=False)


@pytest.mark.parametrize("aggregate,all_epitopes", PVACSEQ_CORPUS_TWINS,
                         ids=[a["pair"] for a, _ in PVACSEQ_CORPUS_TWINS])
def test_real_pvacseq_flavors_agree_on_selected_candidate(aggregate, all_epitopes):
    import numpy as np

    left = read_pvacseq(PVACSEQ_ROOT / aggregate["file"]).df
    right = read_pvacseq(PVACSEQ_ROOT / all_epitopes["file"]).df
    left = left[left.kind.eq("pMHC_affinity") & left.prediction_method_name.eq("pvacseq")]
    right = right[right.kind.eq("pMHC_affinity") & right.prediction_method_name.eq("pvacseq")]
    keys = ["peptide", "allele", "transcript", "gene"]
    # April 25 has repeated NAV2 candidates under distinct pVAC input indexes.
    # Check every matching row, rather than silently dropping duplicates.
    matched = left.merge(right, on=keys, how="left", indicator=True,
                         suffixes=("_agg", "_all"), validate="one_to_many")
    assert len(left) > 0 and matched["_merge"].eq("both").all()
    for column in ("value", "wt_value", "percentile_rank", "wt_percentile_rank",
                  "pvacseq_tumor_rna_depth", "pvacseq_tumor_rna_vaf",
                  "pvacseq_tumor_dna_vaf", "gene_expression"):
        np.testing.assert_allclose(matched[column + "_agg"].astype(float),
                                   matched[column + "_all"].astype(float),
                                   rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=column)


# Protein windows obey the selected lengths; explicit peptide lists do not.
# The consumer workflow drives each registered live/cache pair identically.
CACHE_LENGTH_TWINS = (
    ("protein windows", RandomBindingPredictor.predict_proteins_dataframe,
     CachedPredictor.predict_proteins_dataframe),
    ("explicit peptides", RandomBindingPredictor.predict_dataframe,
     CachedPredictor.predict_dataframe),
)


# Equal-length, unflanked peptides must have the same coverage policy whether
# supplied to the protein scanner or the whole-peptide API. The shared battery
# lives in test_prediction_batch.py and includes strict and reporting modes.
CACHE_MISS_PREDICTION_TWINS = (
    TopiaryPredictor.predict_from_named_sequences,
    TopiaryPredictor.predict_from_named_peptides,
)


CACHE_PROVENANCE_TWINS = (
    ("dataframe", CachedPredictor.from_dataframe),
    ("topiary_output", CachedPredictor.from_topiary_output),
    ("tsv", CachedPredictor.from_tsv),
    ("directory", CachedPredictor.from_directory),
)


def osteosarc_export_paths(manifest, directory, cache):
    from topiary import osteosarc_fixture_paths
    return osteosarc_fixture_paths(manifest, directory=directory)


def osteosarc_cache_paths(manifest, directory, cache):
    from topiary import osteosarc_fixture_paths
    return osteosarc_fixture_paths(manifest, cache=cache)


# One set of original bytes must give identical scientific answers through
# both the checked-in offline export and another consumer's shared cache.
OSTEOSARC_SOURCE_TWINS = (osteosarc_export_paths, osteosarc_cache_paths)


def mhcflurry_version_twins():
    """Both public entry points must apply the same installed-model rule."""
    from mhctools import mhcflurry_composite_version as upstream
    from topiary import mhcflurry_composite_version as downstream

    return (("mhctools", upstream), ("topiary", downstream))


def fragments_with_creator_options(variants, alignment_file, **options):
    """Public convenience options, with result filters disabled for the test."""
    return fragments_from_variants(
        variants, alignment_file, filter_thresholds={}, filter_flags=[], **options,
    )


def fragments_with_explicit_creator(variants, alignment_file, **options):
    """The existing custom-creator door must produce exactly the same result."""
    from isovar.protein_sequence_creator import ProteinSequenceCreator

    return fragments_from_variants(
        variants, alignment_file, filter_thresholds={}, filter_flags=[],
        protein_sequence_creator=ProteinSequenceCreator(
            variant_sequence_assembly=True, **options,
        ),
    )


# These callables consume variants and a BAM, not a DataFrame. The original
# RNA battery in test_consumer_workflows.py drives this registered pair.
ISOVAR_RECONSTRUCTION_TWINS = Twin(
    name="RNA creator options/custom creator",
    left=fragments_with_creator_options,
    right=fragments_with_explicit_creator,
    shared={name: name for name in (
        "protein_sequence_length", "protein_context_peptide_length",
        "protein_sequence_preference", "min_protein_sequence_support_fraction",
        "min_variant_sequence_coverage",
    )},
)


def fragments_from_original_rna(variants, alignment_file, **kwargs):
    """Diagnostic reconstruction with the public Topiary convenience API."""
    return fragments_from_variants(
        variants, alignment_file, filter_thresholds={}, filter_flags=[], **kwargs)


def fragments_from_upstream_rna(variants, alignment_file, **kwargs):
    """The upstream Isovar result and public adapter must agree."""
    from isovar import run_isovar
    from topiary import fragments_from_isovar_results

    return fragments_from_isovar_results(run_isovar(
        variants, alignment_file, filter_thresholds={}, filter_flags=[], **kwargs))


ISOVAR_HANDOFF_TWINS = (fragments_from_original_rna, fragments_from_upstream_rna)

# Diagnostic adapter and outcome report must agree on the sequence/evidence,
# including filtered reconstructions. Driven by every original-BAM corpus case.
ISOVAR_RESULT_TWINS = (describe_isovar_result, fragment_from_isovar_result)


def predict_fragment_records(fragments):
    return TopiaryPredictor(models=RandomBindingPredictor(
        alleles=["HLA-A*01:01"], default_peptide_lengths=[9]),
        only_novel_epitopes=False).predict_from_fragments(fragments)


FRAGMENT_IDENTITY_TWINS = (unique_fragments, predict_fragment_records)


@pytest.mark.parametrize("changes", [
    {"sequence": "GILGFVFTL"}, {"n_rna_alt_reads": 9},
    {"annotations": {"sample": "T2"}}, {"target_intervals": [(1, 2)]},
    {"field_provenance": {"sequence": "measured"}},
])
def test_fragment_identity_doors_reject_the_same_conflicts(changes):
    first = ProteinFragment(fragment_id="same", sequence="SIINFEKLL")
    second = dataclasses.replace(first, **changes)
    for door in FRAGMENT_IDENTITY_TWINS:
        for records in ([first, second], [second, first]):
            with pytest.raises(ValueError, match="Conflicting.*same"):
                door(iter(records))


def test_fragment_identity_doors_coalesce_identical_records():
    import numpy as np

    first = ProteinFragment(fragment_id="same", sequence="SIINFEKLL",
                            annotations={"a": np.int64(3), "b": [float("nan")]})
    second = dataclasses.replace(first, annotations={"b": [float("nan")], "a": 3})
    for door in FRAGMENT_IDENTITY_TWINS:
        assert len(door(iter([first, second, first]))) == 1
        assert len(door([])) == 0


def test_fragment_identity_doors_accept_one_object_but_not_unstorable_copies():
    first = ProteinFragment(fragment_id="same", sequence="SIINFEKLL", annotations={"custom": object()})
    copy = dataclasses.replace(first)
    for door in FRAGMENT_IDENTITY_TWINS:
        assert len(door([first, first])) == 1
        with pytest.raises(ValueError, match="same.*cannot store"):
            door([first, copy])


@pytest.mark.parametrize("fields", [
    {"annotations": {"count": 5}}, {"annotations": {"nested": {"pair": (1, 2)}}},
    {"annotations": {"missing": float("nan")}}, {"annotations": {1: "a", None: "n", "b": 2}},
    {"gene": ""}, {"effect": "None"}, {"gene": "nan"}, {"transcript_name": 7},
    {"sample_name": "T1"},
])
def test_fragment_identity_doors_coalesce_a_record_with_its_saved_copy(fields, tmp_path):
    """A record and its own fragment-IO round trip are the same observation.

    Saving turns expression ``5`` into ``5.0``, NaN and blank text into
    ``None``, tuples into lists, numbers in text fields into text and mapping
    keys into strings. None of those is a conflict, so a cached fragment file
    can be merged with freshly built fragments.
    """
    fresh = ProteinFragment(
        fragment_id="same", sequence="SIINFEKLL", gene_expression=5,
        transcript_expression=float("nan"), n_rna_alt_reads=3,
        target_intervals=[(1, 2)], **fields)
    path = tmp_path / "fragments.tsv"
    write_fragments([fresh], path)
    restored, = read_fragments(path)
    for door in FRAGMENT_IDENTITY_TWINS:
        assert len(door([fresh, restored])) == len(door([restored, fresh])) == 1


def test_fragment_identity_doors_compare_mapping_keys_as_stored():
    first = ProteinFragment(fragment_id="same", sequence="SIINFEKLL", annotations={1: "a", "b": 2})
    for door in FRAGMENT_IDENTITY_TWINS:
        assert len(door([first, dataclasses.replace(first)])) == 1
        assert len(door([first, dataclasses.replace(first, annotations={"1": "a", "b": 2})])) == 1


def test_fragment_identity_doors_compare_a_subclass_in_either_order():
    @dataclass(frozen=True, eq=False)
    class Scored(ProteinFragment):
        score: float = 0.0

    base = ProteinFragment(fragment_id="same", sequence="SIINFEKLL")
    scored = Scored(fragment_id="same", sequence="SIINFEKLL", score=1.0)
    for door in FRAGMENT_IDENTITY_TWINS:
        for records in ([base, scored], [scored, base],
                        [scored, dataclasses.replace(scored, score=2.0)]):
            with pytest.raises(ValueError, match="differ in (class|score)"):
                door(records)
        assert len(door([scored, dataclasses.replace(scored)])) == 1


def test_fragment_identity_doors_keep_one_candidate_across_samples():
    """One ID is one candidate: samples may differ in evidence, not peptides."""
    first = ProteinFragment(fragment_id="same", sequence="SIINFEKLL", n_rna_alt_reads=3)
    for door in FRAGMENT_IDENTITY_TWINS:
        assert len(door([
            dataclasses.replace(first, sample_name="T1"),
            dataclasses.replace(first, sample_name="T2", n_rna_alt_reads=9),
        ])) == 2
        with pytest.raises(ValueError, match="different candidates.*differ in sequence"):
            door([dataclasses.replace(first, sample_name="T1"),
                  dataclasses.replace(first, sample_name="T2", sequence="GILGFVFTL")])


def test_fragment_identity_conflicts_name_every_differing_field():
    first = ProteinFragment(fragment_id="same", sequence="SIINFEKLL", n_rna_alt_reads=3)
    second = dataclasses.replace(first, n_rna_alt_reads=4, gene="GLIS3")
    for door in FRAGMENT_IDENTITY_TWINS:
        with pytest.raises(ValueError, match="differ in gene, n_rna_alt_reads"):
            door([first, second])

FRAME = pd.DataFrame({"x": [1, 2]}, index=[10, 11])


# ---------------------------------------------------------------------------
# pVACseq's two report flavors
#
# Their schemas differ too much for the generic keyword-argument battery
# above. Register the pair here and drive both public reader branches through
# the same semantic assertion instead.
# ---------------------------------------------------------------------------

PVACSEQ_PRESENTATION_TWINS = (
    (
        "aggregated/all_epitopes presentation",
        "tests/data/pvacseq/mhc_i_aggregated_presentation.tsv",
        "tests/data/pvacseq/mhc_i_all_epitopes_presentation.tsv",
    ),
)


@pytest.fixture
def without_installed_metadata(monkeypatch):
    """Isolate import/API checks from whatever release this machine has."""
    from importlib.metadata import PackageNotFoundError

    def absent(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(optional_dependencies, "version", absent)


# ---------------------------------------------------------------------------
# Optional integrations
#
# Isovar and PirlyGenes are independent doors, but both must distinguish an
# absent optional package from a broken installed package in the same way.
# ---------------------------------------------------------------------------

OPTIONAL_DEPENDENCY_TWINS = (
    (
        "isovar",
        ("run_isovar", "ProteinSequenceCreator"),
        _check_isovar,
        "assembling protein fragments from RNA alignments",
        ">=1.18.1",
    ),
    (
        "pirlygenes",
        ("pan_cancer_expression",),
        lambda: _check_pirlygenes("pan_cancer_expression"),
        "cancer-testis antigen and tissue-expression gene lists",
        ">=5.1.0",
    ),
)


# ---------------------------------------------------------------------------
# ProteinFragment construction
#
# Direct construction and serialized input are two doors onto the same field
# migration contract. Drive every applicable rename through both so adding a
# constructor path cannot make old files and old Python calls mean different
# things again.
# ---------------------------------------------------------------------------

_FRAGMENT_FIELD_NAMES = {
    fragment_field.name for fragment_field in dataclasses.fields(ProteinFragment)
}
FRAGMENT_FIELD_RENAMES = tuple(sorted(
    (old, new) for old, new in RENAMED_COLUMNS.items()
    if new in _FRAGMENT_FIELD_NAMES
))


def _construct_fragment_directly(values):
    return ProteinFragment(fragment_id="direct", sequence="SIINFEKLA", **values)


def _construct_fragment_from_dict(values):
    return ProteinFragment.from_dict({
        "fragment_id": "direct",
        "sequence": "SIINFEKLA",
        **values,
    })


FRAGMENT_CONSTRUCTION_DOORS = (
    ("direct", _construct_fragment_directly),
    ("from_dict", _construct_fragment_from_dict),
)


def fragment_dict_roundtrip(fragment, path):
    return ProteinFragment.from_dict(fragment.to_dict())


def fragment_json_roundtrip(fragment, path):
    return ProteinFragment.from_json(fragment.to_json())


def fragment_tsv_roundtrip(fragment, path):
    write_fragments([fragment], path)
    restored, = read_fragments(path)
    return restored


# The same scalar/type battery and real custom-creator workflow drive all
# three serialization doors in test_consumer_workflows.py.
FRAGMENT_SERIALIZATION_DOORS = (
    ("dict", fragment_dict_roundtrip),
    ("json", fragment_json_roundtrip),
    ("tsv", fragment_tsv_roundtrip),
)


def release_allowed_through_api(project, version):
    from topiary import pypi_release_exists

    try:
        return not pypi_release_exists(project, version)
    except (ValueError, RuntimeError):
        return False


def release_allowed_through_cli(project, version):
    from topiary.cli.release import main

    return main([project, version]) == 0


RELEASE_PREFLIGHT_DOORS = (release_allowed_through_api, release_allowed_through_cli)


def whole_peptides_through_predictor(model, peptides):
    return TopiaryPredictor(models=model).predict_from_named_peptides(
        {str(i): peptide for i, peptide in enumerate(peptides)},
    )


def whole_peptides_through_predictions(model, peptides):
    predictions = [prediction for result in model.predict(peptides) for prediction in result.preds]
    return from_predictions(
        predictions, extra_columns={"source_sequence_name": [str(i) for i in range(len(peptides))]},
    )


# Real mhctools wrappers, with only their external sidecars stubbed, drive
# this pair in test_consumer_workflows.py. It tests integration, not model accuracy.
WHOLE_PEPTIDE_PREDICTION_DOORS = (whole_peptides_through_predictor, whole_peptides_through_predictions)


def stability_through_stdout_cache(path):
    from topiary import CachedPredictor

    return CachedPredictor.from_netmhcstabpan_stdout(path).predict_peptides_dataframe(["SLLQHLIGL"])


def stability_through_native_conversion(path):
    from mhctools.parsing import parse_netmhcstabpan

    return from_predictions([
        prediction.to_pred(kind="pMHC_stability")
        for prediction in parse_netmhcstabpan(path.read_text())
    ])


STABILITY_PREDICTION_DOORS = (
    stability_through_stdout_cache,
    stability_through_native_conversion,
)


@pytest.mark.parametrize(("old", "new"), FRAGMENT_FIELD_RENAMES)
@pytest.mark.parametrize(
    "style", ("legacy", "current", "matching", "empty-current"),
)
def test_fragment_construction_doors_agree_on_renames(old, new, style):
    if style == "legacy":
        values = {old: 12, "field_provenance": {old: APPROXIMATED}}
    elif style == "current":
        values = {new: 12, "field_provenance": {new: APPROXIMATED}}
    elif style == "matching":
        values = {
            old: 12,
            new: 12,
            "field_provenance": {old: APPROXIMATED, new: APPROXIMATED},
        }
    else:
        values = {
            old: 12,
            new: None,
            "field_provenance": {old: APPROXIMATED},
        }

    fragments = [door(values) for _, door in FRAGMENT_CONSTRUCTION_DOORS]
    assert fragments[0].to_dict() == fragments[1].to_dict()
    for fragment in fragments:
        assert getattr(fragment, old) == getattr(fragment, new) == 12
        assert fragment.provenance_of(old) == APPROXIMATED
        assert old not in fragment.to_dict()


@pytest.mark.parametrize(("old", "new"), FRAGMENT_FIELD_RENAMES)
@pytest.mark.parametrize("conflict", ("value", "provenance"))
def test_fragment_construction_doors_agree_on_conflicts(old, new, conflict):
    values = {old: 12, new: 13}
    if conflict == "provenance":
        values = {
            old: 12,
            "field_provenance": {old: MEASURED, new: APPROXIMATED},
        }

    errors = []
    for _, door in FRAGMENT_CONSTRUCTION_DOORS:
        with pytest.raises(ValueError) as raised:
            door(values)
        errors.append(str(raised.value))

    assert errors[0] == errors[1]
    assert "Conflicting ProteinFragment fields" in errors[0]


@pytest.mark.parametrize(
    ("dependency", "required_api", "check", "feature", "specifier"),
    OPTIONAL_DEPENDENCY_TWINS,
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_optional_dependency_metadata_has_one_floor(
    dependency, required_api, check, feature, specifier,
):
    del required_api, check, feature
    declared = [
        Requirement(text) for text in (requires("topiary") or ())
        if Requirement(text).name == dependency
    ]

    assert len(declared) == 1
    assert str(declared[0].specifier) == specifier
    assert declared[0].marker.evaluate({"extra": dependency})
    other = "pirlygenes" if dependency == "isovar" else "isovar"
    assert not declared[0].marker.evaluate({"extra": other})


@pytest.mark.parametrize(
    ("dependency", "required_api", "check", "feature", "specifier"),
    OPTIONAL_DEPENDENCY_TWINS,
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_optional_dependency_missing_errors_match(
    monkeypatch, dependency, required_api, check, feature, specifier,
):
    del required_api, specifier
    original = ModuleNotFoundError(
        f"No module named '{dependency}'", name=dependency,
    )

    def missing(module_name):
        del module_name
        raise original

    monkeypatch.setattr(optional_dependencies, "import_module", missing)

    with pytest.raises(ImportError) as raised:
        check()

    message = str(raised.value)
    assert feature in message
    assert f"pip install 'topiary[{dependency}]'" in message
    assert "installed but" not in message
    assert raised.value.__cause__ is original


@pytest.mark.parametrize(
    ("dependency", "required_api", "check", "feature", "specifier"),
    OPTIONAL_DEPENDENCY_TWINS,
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_optional_dependency_broken_import_errors_match(
    monkeypatch, dependency, required_api, check, feature, specifier,
):
    del required_api, specifier
    original = ModuleNotFoundError(
        "No module named 'broken_transitive_dependency'",
        name="broken_transitive_dependency",
    )

    def broken(module_name):
        del module_name
        raise original

    monkeypatch.setattr(optional_dependencies, "import_module", broken)

    with pytest.raises(ImportError) as raised:
        check()

    message = str(raised.value)
    assert feature in message
    assert "installed but could not be imported" in message
    assert "broken_transitive_dependency" in message
    assert f"pip install --upgrade 'topiary[{dependency}]'" in message
    assert raised.value.__cause__ is original


@pytest.mark.parametrize(
    ("dependency", "required_api", "check", "feature", "specifier"),
    OPTIONAL_DEPENDENCY_TWINS,
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_optional_dependency_capability_errors_match(
    monkeypatch, without_installed_metadata, dependency, required_api, check, feature, specifier,
):
    del feature, specifier
    monkeypatch.setattr(
        optional_dependencies,
        "import_module",
        lambda module_name: SimpleNamespace(__name__=module_name),
    )

    with pytest.raises(ImportError) as raised:
        check()

    message = str(raised.value)
    assert "installed but does not provide the API" in message
    assert all(name in message for name in required_api)
    assert f"pip install --upgrade 'topiary[{dependency}]'" in message


@pytest.mark.parametrize(
    ("dependency", "required_api", "check", "feature", "specifier"),
    OPTIONAL_DEPENDENCY_TWINS,
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_optional_dependency_capabilities_load_through_both_doors(
    monkeypatch, without_installed_metadata, dependency, required_api, check, feature, specifier,
):
    del dependency, feature, specifier
    module = SimpleNamespace(**{name: lambda: None for name in required_api})
    monkeypatch.setattr(
        optional_dependencies, "import_module", lambda module_name: module,
    )

    assert check() is module


@pytest.mark.parametrize(
    ("dependency", "required_api", "check", "feature", "specifier"),
    OPTIONAL_DEPENDENCY_TWINS,
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_optional_dependency_floors_refuse_older_releases_through_both_doors(
    monkeypatch, dependency, required_api, check, feature, specifier,
):
    """An older release imports fine and answers wrongly; both doors refuse it.

    The floor is read from Topiary's own metadata, the one place it is
    declared, so the runtime check cannot drift from the installer's.
    """
    import topiary

    module = SimpleNamespace(**{name: lambda: None for name in required_api})
    monkeypatch.setattr(optional_dependencies, "import_module", lambda module_name: module)
    # Independent of whatever this machine has installed.
    monkeypatch.setattr(optional_dependencies, "requires", lambda name: [
        f'{dependency}{specifier}; extra == "{dependency}"'])
    installed = {"topiary": topiary.__version__}
    monkeypatch.setattr(optional_dependencies, "version", lambda name: installed[name])

    installed[dependency] = "0.0.1"
    with pytest.raises(ImportError) as raised:
        check()
    message = str(raised.value)
    assert f"{dependency} 0.0.1 is installed" in message
    assert feature in message and specifier in message
    assert f"pip install --upgrade 'topiary[{dependency}]'" in message

    installed[dependency] = specifier.removeprefix(">=")
    assert check() is module

    # Metadata from another copy of Topiary may carry another floor.
    installed.update({"topiary": "0.0.0", dependency: "0.0.1"})
    assert check() is module


@pytest.mark.parametrize(
    ("name", "aggregated_path", "all_epitopes_path"),
    PVACSEQ_PRESENTATION_TWINS,
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_pvacseq_flavors_agree_on_aggregate_presentation(
    name, aggregated_path, all_epitopes_path,
):
    """Both reader doors preserve pVACtools' aggregate presentation rank."""
    del name
    rows = []
    for path in (aggregated_path, all_epitopes_path):
        df = read_pvacseq(path).df
        rows.append(df[
            (df["kind"] == "pMHC_presentation")
            & (df["prediction_method_name"] == "pvacseq")
        ].iloc[0])

    for column in (
        "peptide", "allele", "kind", "prediction_method_name",
        "percentile_rank", "wt_percentile_rank",
    ):
        assert rows[0][column] == rows[1][column]


# ---------------------------------------------------------------------------
# The cache's two doors (topiary#231)
#
# Not expressible as a Twin: they take different arguments (a frame vs a
# list of caches) and one is a classmethod. The pairing is still the
# point, so they get their own battery over the same inputs.
# ---------------------------------------------------------------------------


def _cache_row(**overrides):
    row = dict(
        peptide="SIINFEKLA", allele="HLA-A*02:01", peptide_length=9,
        kind="pMHC_affinity", score=0.5, affinity=100.0,
        percentile_rank=1.0, value=100.0,
        prediction_method_name="netmhcpan", predictor_version="4.1",
    )
    row.update(overrides)
    return row


@pytest.mark.parametrize("recorded, supplied, accepted", [
    (None, "2.10", True),
    ("", "2.10", True),
    ("<NA>", "2.10", True),
    ("2.10", "2.10", True),
    ("2.10", None, True),
    ("2.10", "2.11", False),
    (None, None, False),
    (None, " ", False),
])
def test_cache_loader_doors_agree_on_provenance(recorded, supplied, accepted, tmp_path):
    frame = pd.DataFrame([_cache_row(predictor_version=recorded)])
    original = frame.copy(deep=True)
    path = tmp_path / "cache.tsv"
    frame.to_csv(path, sep="\t", index=False)
    outcomes = {}
    versions = {}
    for name, loader in CACHE_PROVENANCE_TWINS:
        source = frame if name == "dataframe" else tmp_path if name == "directory" else path
        try:
            cache = loader(source, predictor_version=supplied)
            outcomes[name] = True
            versions[name] = cache.predictor_version
        except ValueError:
            outcomes[name] = False
    assert set(outcomes.values()) == {accepted}, outcomes
    if accepted:
        assert set(versions.values()) == {"2.10"}, versions
    pd.testing.assert_frame_equal(frame, original)


CACHE_CASES = {
    "identical rows": ([_cache_row()], [_cache_row()], "accept"),
    "differ only in context": (
        [_cache_row(sample_name="a")], [_cache_row(sample_name="b")], "accept",
    ),
    "both leave affinity unstated": (
        [_cache_row(affinity=None)], [_cache_row(affinity=None)], "accept",
    ),
    "disagree on affinity": (
        [_cache_row(affinity=100.0)], [_cache_row(affinity=250.0)], "raise",
    ),
    "disagree on score": (
        [_cache_row(score=0.5)], [_cache_row(score=0.9)], "raise",
    ),
    "one states affinity, one does not": (
        [_cache_row(affinity=None)], [_cache_row(affinity=250.0)], "raise",
    ),
}


@pytest.mark.parametrize("case", sorted(CACHE_CASES), ids=lambda c: c)
def test_the_cache_doors_agree_about_a_repeated_key(case):
    """topiary#231: concat raised on any repeat, the constructor on none.

    `concat` rejecting two shards that share one identical row broke
    `from_directory` on a perfectly consistent cache; the constructor
    accepting a key with two different scores meant a lookup returned
    whichever row came last. Neither is defensible, and they were
    opposite.
    """
    from topiary import CachedPredictor

    left, right, expected = CACHE_CASES[case]

    def through_constructor():
        return CachedPredictor(pd.DataFrame(left + right))

    def through_concat():
        return CachedPredictor.concat([
            CachedPredictor(pd.DataFrame(left)),
            CachedPredictor(pd.DataFrame(right)),
        ])

    outcomes = {}
    caches = {}
    for label, call in (("constructor", through_constructor),
                        ("concat", through_concat)):
        try:
            caches[label] = call()
            outcomes[label] = "accept"
        except ValueError:
            outcomes[label] = "raise"

    assert outcomes["constructor"] == outcomes["concat"], (
        f"{case}: constructor {outcomes['constructor']}s, "
        f"concat {outcomes['concat']}s"
    )
    assert outcomes["constructor"] == expected
    if expected == "accept":
        pd.testing.assert_frame_equal(
            caches["constructor"]._df,
            caches["concat"]._df,
            check_like=True,
        )


def test_every_cache_column_is_classified_as_key_value_or_context():
    """The gap that sank the first attempt at #231.

    It compared a hand-listed subset of value columns, left `affinity`
    out, and merged caches that disagreed about affinity in silence. A
    partition assertion turns adding a cache column into a decision
    about which group it belongs to.
    """
    from topiary.cached import (
        _CACHE_COLUMNS,
        PREDICTION_CONTEXT_COLUMNS,
        PREDICTION_KEY_COLUMNS,
        PREDICTION_VALUE_COLUMNS,
    )

    groups = (
        set(PREDICTION_KEY_COLUMNS)
        | set(PREDICTION_VALUE_COLUMNS)
        | set(PREDICTION_CONTEXT_COLUMNS)
    )
    assert not set(_CACHE_COLUMNS) - groups, "unclassified cache column(s)"
    assert not groups - set(_CACHE_COLUMNS), "classified non-cache column(s)"


def _ids(twin):
    return [f"{twin.name}:{arg}" for arg in twin.shared]


@pytest.mark.parametrize("twin", TWINS, ids=lambda t: t.name)
def test_a_misaligned_series_is_refused_by_both_or_neither(twin):
    """The instance that motivated this file.

    One door aligning while the other assigns positionally loses data
    either way, and the caller is told nothing.
    """
    misaligned = pd.Series([100, 200])  # RangeIndex, frame is [10, 11]

    for arg in twin.shared:
        outcomes = []
        for fn, kwargs in twin.calls(arg, misaligned):
            try:
                fn(FRAME, **kwargs)
                outcomes.append("accepted")
            except Exception as exc:
                outcomes.append(type(exc).__name__)
        assert outcomes[0] == outcomes[1], (
            f"{twin.name}: {arg!r} -> {outcomes[0]}, "
            f"{twin.shared[arg]!r} -> {outcomes[1]}"
        )


@pytest.mark.parametrize("twin", TWINS, ids=lambda t: t.name)
def test_a_wrong_length_sequence_is_refused_by_both_or_neither(twin):
    for arg in twin.shared:
        outcomes = []
        for fn, kwargs in twin.calls(arg, [1, 2, 3]):
            try:
                fn(FRAME, **kwargs)
                outcomes.append("accepted")
            except Exception as exc:
                outcomes.append(type(exc).__name__)
        assert outcomes[0] == outcomes[1], (
            f"{twin.name}: wrong-length {arg!r} -> {outcomes[0]}, "
            f"{twin.shared[arg]!r} -> {outcomes[1]}"
        )


@pytest.mark.parametrize("twin", TWINS, ids=lambda t: t.name)
def test_an_aligned_series_and_a_bare_sequence_agree(twin):
    """A bare sequence has no index to honour, so it is positional.

    Both doors must read it that way, and must agree with the aligned
    Series carrying the same numbers -- otherwise the convenience of
    passing a list quietly means something different per door.
    """
    values = [100, 200]
    aligned = pd.Series(values, index=FRAME.index)

    for twin_arg, other_arg in ((a, twin.shared[a]) for a in twin.shared):
        for fn, arg in ((twin.left, twin_arg), (twin.right, other_arg)):
            from_series = fn(FRAME, **{arg: aligned})
            from_list = fn(FRAME, **{arg: values})
            added = [c for c in from_series.columns if c != "x"]
            for column in added:
                pd.testing.assert_series_equal(
                    from_series[column], from_list[column],
                    check_names=False,
                )


@pytest.mark.parametrize("twin", TWINS, ids=lambda t: t.name)
def test_absent_input_writes_no_column_on_either_side(twin):
    """Omit-not-null, checked as a property of the pair.

    This rule was applied to the DNA side first and reached the RNA side
    two releases later; a pair-level assertion would have failed the day
    they diverged.
    """
    bare_left = twin.left(FRAME)
    bare_right = twin.right(FRAME)
    for out, label in ((bare_left, "left"), (bare_right, "right")):
        nulled = [
            c for c in out.columns
            if c != "x" and not out[c].notna().any()
        ]
        assert not nulled, (
            f"{twin.name} ({label}): wrote all-null columns for absent "
            f"inputs: {nulled}"
        )


@pytest.mark.parametrize("twin", TWINS, ids=lambda t: t.name)
def test_all_null_input_writes_no_column_on_either_side(twin):
    """An all-null Series is the column-level form of absent input."""
    values = pd.Series([None, None], index=FRAME.index)

    for arg in twin.shared:
        for fn, kwargs in twin.calls(arg, values):
            out = fn(FRAME, **kwargs)
            assert list(out.columns) == ["x"]


@pytest.mark.parametrize("twin", TWINS, ids=lambda t: t.name)
def test_coverage_without_a_fraction_has_the_same_subject(twin):
    left = twin.left(FRAME, overlapping=[10, 20])
    right = twin.right(FRAME, depth=[10, 20])

    assert set(left["rna_evidence_subject"]) == {"reads"}
    assert set(right["dna_evidence_subject"]) == {"reads"}


# Native catalogue and historical audit adapter: statuses and diagnostics must
# stay identical even when bad rows precede, follow, or accompany valid ones.
OSTEOSARC_INVENTORY_TWINS = (parse_variants, variant_inventory)


@pytest.mark.parametrize("position", ["10", "010", "", "NA", "1.5", "-1"])
@pytest.mark.parametrize("placement", ["first", "last", "alongside-valid"])
def test_osteosarc_inventory_doors_keep_status_and_diagnostics(position, placement):
    from .test_osteosarc_audit_inventory import HEADER, INDEX, OTHER

    html = INDEX.replace("</table>", "") + OTHER.replace("<table>", "")
    row = f"example\tGENE\tchr1\t{position}\tA\tC\n"
    other = "other\tOTHER\tchr2\t20\tG\tT\n"
    rows = row + other if placement == "first" else other + row
    if placement == "alongside-valid":
        rows += "example\tGENE\tchr1\t10\tA\tC\n"
    native, adapter = OSTEOSARC_INVENTORY_TWINS
    variants = list(native(html, HEADER + rows).select(on_site=True))
    records = adapter(html, HEADER + rows)
    assert [v.id for v in variants] == [r["variant_id"] for r in records]
    for variant, record in zip(variants, records):
        assert record["input_status"] == variant.status
        assert record["candidate_alleles"] == [list(a) for a in variant.alleles]
        assert record.get("parse_errors") == variant.annotations.get("parse_errors")
    assert records[1]["input_status"] == "ready"
    assert records[0]["input_status"] == ("ready" if position in ("10", "010") else "malformed_source_row")


@pytest.mark.parametrize("header", ["variant_id\tgene\tchrom\tref\talt\n", "variant_id\tvariant_id\n"])
def test_osteosarc_inventory_doors_reject_invalid_headers(header):
    from osteosarc import SchemaError
    from .test_osteosarc_audit_inventory import INDEX

    for parse in OSTEOSARC_INVENTORY_TWINS:
        with pytest.raises(SchemaError):
            parse(INDEX, header)


@pytest.mark.parametrize("bad_row", ["example\tGENE\tchr1\t10", "example\tGENE\tchr1\t10\tA\tC\textra"])
@pytest.mark.parametrize("bad_first", [True, False])
def test_osteosarc_inventory_doors_preserve_ragged_row_diagnostics(bad_row, bad_first):
    from .test_osteosarc_audit_inventory import HEADER, INDEX, OTHER

    html = INDEX.replace("</table>", "") + OTHER.replace("<table>", "")
    good = "other\tOTHER\tchr2\t20\tG\tT"
    rows = [bad_row, good] if bad_first else [good, bad_row]
    table = HEADER + "\n".join(rows) + "\n"
    native, adapter = OSTEOSARC_INVENTORY_TWINS
    variants = list(native(html, table).select(on_site=True))
    records = adapter(html, table)
    assert [v.status for v in variants] == [r["input_status"] for r in records] == [
        "malformed_source_row", "ready"]
    assert variants[0].annotations["parse_errors"] == records[0]["parse_errors"]


# The same single input must score identically with and without source tracking.
SOURCE_VIEW_TWINS = (
    TopiaryResult,
    lambda frame: combine_sources({"only": frame}, sample_name="p"),
)


@pytest.mark.parametrize("scope", ["wt", "self", "self_nearest", "shuffled"])
@pytest.mark.parametrize("expression", ["affinity.best_value", "affinity.value / presentation.score"])
def test_source_tracking_preserves_scoped_prediction_aggregation(scope, expression):
    from topiary import parse

    frame = pd.DataFrame(dict(
        peptide=["SIINFEKL"] * 4, source_sequence_name=["orf"] * 4,
        peptide_offset=[0] * 4,
        allele=["HLA-A*02:01", "HLA-B*07:02"] * 2,
        kind=["pMHC_affinity"] * 2 + ["pMHC_presentation"] * 2,
        value=[20., 200., .1, .8], score=[.9, .2, .1, .8],
        prediction_method_name=["model"] * 4,
    ))
    frame[f"{scope}_value"] = [30., 300., .2, .9]
    frame[f"{scope}_score"] = [.8, .1, .2, .9]
    frame[f"{scope}_percentile_rank"] = [1., 10., 2., 20.]
    frame[f"{scope}_peptide"] = "SIINFEKLK"
    plain, combined = (constructor(frame) for constructor in SOURCE_VIEW_TWINS)
    for expr in (expression, expression.replace("affinity.", f"{scope}.affinity.")):
        expected = evaluate_scores(plain.df, parse(expr))
        assert expected.notna().all()
        pd.testing.assert_series_equal(evaluate_scores(combined.df, parse(expr)), expected)
