"""Tabular transport for MHCTools' native per-bond evidence."""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

import pandas as pd
from mhctools import CleavageResult  # type: ignore[import-untyped]

from .serialization import normalize_python_types


_COLUMNS = (
    "result_index", "sequence", "occurrence_id", "source_sequence_name",
    "source_start", "bond", "source_bond", "prediction_method_name",
    "predictor_version", "enzyme", "evidence_type", "scored_endpoint",
    "score_name", "score_units", "status", "reason", "match", "score",
    "substrate_observation", "peptide_input", "model_metadata", "conditions",
)


def from_cleavage_results(
    results: Iterable[CleavageResult | Mapping[str, Any]],
) -> pd.DataFrame:
    """Build a site-evidence table from native or serialized MHCTools results.

    Parameters
    ----------
    results : iterable of mhctools.CleavageResult or mapping
        Already computed evidence, or records from ``CleavageResult.to_dict``.
        Both forms pass through the upstream validation. No model runs here.

    Returns
    -------
    pandas.DataFrame
        One row per reported bond. ``bond=b`` splits the exact input at
        ``sequence[:b] | sequence[b:]``; ``source_bond`` adds ``source_start``.
        ``result_index`` groups rows from one input result in this call, not
        independent biological observations. Input order and repeated source
        occurrences are retained; there is no automatic deduplication.

        Results without sites produce one row with null bond, match and score.
        Their status is ``unsupported`` with the producer's reason, a stated
        whole-substrate observation, or ``unassessed``. A missing bond is never
        a terminal bond or a measured zero. Omitted internal bonds remain
        unassessed; the table does not invent a full cut map.

        ``match`` is a nullable binary motif decision; ``score`` is a native
        quantitative value with its declared endpoint/name/units. Neither is
        converted to a probability or a peptide-MHC score. ``peptide_input``,
        ``model_metadata`` and ``conditions`` preserve chemical form, source,
        assay context, references and limitations through Topiary TSV/CSV IO.
        Empty input returns the same columns with no rows.

    Notes
    -----
    Site evidence is separate from whole-peptide half-life and peptide-MHC
    predictions. Use ordinary DataFrame operations to inspect bond positions;
    do not pass this table to ``CachedPredictor`` or treat it as allele evidence.
    A motif non-match does not establish resistance to cleavage.
    """
    rows = []
    for index, supplied in enumerate(results):
        if isinstance(supplied, CleavageResult):
            supplied = supplied.to_dict()
        if not isinstance(supplied, Mapping):
            raise TypeError("Expected CleavageResult or its serialized mapping")
        result = CleavageResult.from_dict(supplied)
        peptide, model = result.peptide, result.model
        base = dict(
            result_index=index, sequence=peptide.sequence,
            occurrence_id=peptide.occurrence_id,
            source_sequence_name=peptide.source_sequence_name,
            source_start=peptide.source_start,
            prediction_method_name=model.name, predictor_version=model.version,
            enzyme=model.enzyme, evidence_type=model.evidence,
            scored_endpoint=model.scored_endpoint,
            score_name=model.score_name, score_units=model.score_units,
            substrate_observation=result.substrate_observation,
            peptide_input=normalize_python_types(peptide.to_dict()),
            model_metadata=normalize_python_types(model.to_dict()),
            conditions=normalize_python_types(dict(result.conditions)),
        )
        for site in result.sites:
            rows.append(dict(
                deepcopy(base), bond=site.bond,
                source_bond=peptide.source_start + site.bond,
                status=site.status, reason=site.reason,
                match=site.match, score=site.score,
            ))
        if not result.sites:
            rows.append(dict(
                base, bond=None, source_bond=None, match=None, score=None,
                status=("unsupported" if result.unsupported_reason else
                        result.substrate_observation or "unassessed"),
                reason=result.unsupported_reason,
            ))
    frame = pd.DataFrame(rows, columns=_COLUMNS)
    for column in ("result_index", "source_start", "bond", "source_bond", "match"):
        frame[column] = pd.Series(frame[column], dtype="Int64")
    return frame


_MODEL_COLUMNS = ("prediction_method_name", "predictor_version", "enzyme",
                  "evidence_type")
_OCCURRENCE_COLUMNS = ("occurrence_id", "source_sequence_name")


def _interval_rows(sites, start, end, occurrence_id, source_sequence_name):
    """One occurrence's rows, and those whose bond is strictly inside the interval."""
    needed = {"source_bond", *_MODEL_COLUMNS, *_OCCURRENCE_COLUMNS}
    if not isinstance(sites, pd.DataFrame) or needed - set(sites.columns):
        raise ValueError(
            "Expected a table from from_cleavage_results (missing: %s)"
            % ", ".join(sorted(needed - set(getattr(sites, "columns", ())))))
    for name, value in (("start", start), ("end", end)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("%s must be an integer source coordinate" % name)
    if end <= start:
        raise ValueError("end must be greater than start")
    keep = pd.Series(True, index=sites.index)
    for column, wanted in (("occurrence_id", occurrence_id),
                           ("source_sequence_name", source_sequence_name)):
        if wanted is not None:
            keep &= sites[column] == wanted
    chosen = sites[keep]
    distinct = chosen[list(_OCCURRENCE_COLUMNS)].drop_duplicates()
    if len(distinct) > 1:
        raise ValueError(
            "sites hold %d source occurrences; pass occurrence_id and/or "
            "source_sequence_name to choose one" % len(distinct))
    bond = chosen["source_bond"]
    return chosen, chosen[bond.notna() & (bond > start) & (bond < end)]


def select_cleavage_sites(
    sites: pd.DataFrame, start: int, end: int, *,
    occurrence_id: str | None = None, source_sequence_name: str | None = None,
) -> pd.DataFrame:
    """Select the assessed internal bonds of a candidate interval.

    Parameters
    ----------
    sites : pandas.DataFrame
        Output of :func:`from_cleavage_results`.
    start, end : int
        Half-open candidate interval ``[start, end)`` in the source coordinate
        frame (``source_bond``).
    occurrence_id, source_sequence_name : str, optional
        Choose one source occurrence. Required when ``sites`` holds more than
        one, because a cut in one copy of a repeated epitope says nothing about
        another copy; a mixed table raises instead of silently pooling them.

    Returns
    -------
    pandas.DataFrame
        The rows with ``start < source_bond < end``, in input order, with every
        column retained. A bond equal to ``start`` or ``end`` is a boundary of
        the interval, not an internal bond, and is excluded. Rows without a bond
        (unsupported, unassessed or whole-substrate observations) are never
        selected. Empty when nothing was assessed inside the interval.

    Notes
    -----
    This states where evidence lies, not what it means: whether a cut destroys
    every copy of a ligand, or another window is preferable, is the caller's
    decision. Use :func:`cleavage_interval_coverage` to see how much of the
    interval the evidence reaches at all.
    """
    _, inside = _interval_rows(
        sites, start, end, occurrence_id, source_sequence_name)
    return inside.copy()


def cleavage_interval_coverage(
    sites: pd.DataFrame, start: int, end: int, *,
    occurrence_id: str | None = None, source_sequence_name: str | None = None,
) -> pd.DataFrame:
    """Report how many internal bonds of an interval each model assessed.

    Parameters
    ----------
    sites, start, end, occurrence_id, source_sequence_name
        As for :func:`select_cleavage_sites`.

    Returns
    -------
    pandas.DataFrame
        One row per ``(prediction_method_name, predictor_version, enzyme,
        evidence_type)`` present for the chosen occurrence, with
        ``n_internal_bonds`` (``end - start - 1``), ``n_assessed`` (distinct
        bonds with a row), ``n_unassessed`` and ``fraction_assessed``. Repeated
        results for one bond count once. A model with no row for the occurrence
        does not appear: the table cannot name a model it was never given.

    Notes
    -----
    A bond with no row was not scored, usually because the scored context did
    not reach it. That is missing context, not a measured zero and not an
    unfavourable score, so callers must not rank it as one. A model that
    assessed every bond but matched none is reported fully covered; a motif
    non-match does not establish resistance to cleavage.
    """
    chosen, inside = _interval_rows(
        sites, start, end, occurrence_id, source_sequence_name)
    n_bonds = end - start - 1
    keys = list(_MODEL_COLUMNS)
    assessed = inside.groupby(keys, dropna=False)["source_bond"].nunique()
    out = chosen[keys].drop_duplicates().reset_index(drop=True)
    out["n_internal_bonds"] = n_bonds
    out["n_assessed"] = [
        int(assessed.get(tuple(row), 0)) for row in out[keys].itertuples(index=False)]
    out["n_unassessed"] = n_bonds - out["n_assessed"]
    out["fraction_assessed"] = (
        out["n_assessed"] / n_bonds if n_bonds else float("nan"))
    return out
