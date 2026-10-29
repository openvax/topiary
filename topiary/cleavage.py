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
