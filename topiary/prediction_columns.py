"""Shared vocabulary for prediction columns in external reports."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionMetric:
    """Meaning recovered from a model name and metric label.

    Attributes
    ----------
    prediction_method_name : str
        Canonical Topiary method name.
    kind : str
        Canonical prediction kind, such as ``"pMHC_presentation"``.
    field : str
        Canonical long-form field: ``"value"``, ``"score"``, or
        ``"percentile_rank"``.
    sequence : {``"mt"``, ``"wt"``, None}
        Whether the label names the mutant or wildtype sequence. ``None``
        means that the label does not say.
    """

    prediction_method_name: str
    kind: str
    field: str
    sequence: str | None = None


_AFFINITY = "pMHC_affinity"
_PRESENTATION = "pMHC_presentation"
_STABILITY = "pMHC_stability"
_PROCESSING = "antigen_processing"
_IMMUNOGENICITY = "immunogenicity"


def _normalized(text):
    """Lowercase *text* and reduce punctuation/case boundaries to tokens."""
    text = str(text).strip()
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", text)
    text = re.sub(r"%\s*ile", " percentile ", text, flags=re.IGNORECASE)
    text = re.sub(r"%\s*rank", " percentile_rank ", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# Model names supported by pVACtools and/or mhctools. Values are the method
# name Topiary itself emits and the kind to use when a metric says only
# "Score" or "Percentile". A None default means that the base model is
# multi-purpose or needs an explicit metric/mode.
_MODEL_DEFAULTS = {
    # Affinity / binding
    "netmhc": ("netmhc", _AFFINITY),
    "netmhc3": ("netmhc", _AFFINITY),
    "netmhc4": ("netmhc", _AFFINITY),
    "netmhcpan": ("netmhcpan", _AFFINITY),
    "netmhcpan28": ("netmhcpan", _AFFINITY),
    "netmhcpan3": ("netmhcpan", _AFFINITY),
    "netmhcpan4": ("netmhcpan", _AFFINITY),
    "netmhcpan41": ("netmhcpan", _AFFINITY),
    "netmhcpan42": ("netmhcpan", _AFFINITY),
    "netmhcpaniedb": ("netmhcpan", _AFFINITY),
    "netmhciipan": ("netmhciipan", _AFFINITY),
    "netmhciipan3": ("netmhciipan", _AFFINITY),
    "netmhciipan4": ("netmhciipan", _AFFINITY),
    "netmhciipan43": ("netmhciipan", _AFFINITY),
    "netmhciipaniedb": ("netmhciipan", _AFFINITY),
    "netmhccons": ("netmhccons", _AFFINITY),
    "netmhcconsiedb": ("netmhccons", _AFFINITY),
    "mhcflurry": ("mhcflurry", _AFFINITY),
    "mhcflurryaffinity": ("mhcflurry", _AFFINITY),
    "mhcnuggetsi": ("mhcnuggetsi", _AFFINITY),
    "mhcnuggetsii": ("mhcnuggetsii", _AFFINITY),
    "mixmhcpred": ("mixmhcpred", _AFFINITY),
    "nnalign": ("nnalign", _AFFINITY),
    "pickpocket": ("pickpocket", _AFFINITY),
    "smm": ("smm", _AFFINITY),
    "smmiedb": ("smm", _AFFINITY),
    "smmpmbec": ("smmpmbec", _AFFINITY),
    "smmpmbeciedb": ("smmpmbec", _AFFINITY),
    "smmalign": ("smmalign", _AFFINITY),
    "tlbind": ("tlbind", _AFFINITY),
    "random": ("random", _AFFINITY),
    # Presentation / immunogenicity
    "mixmhc2pred": ("mixmhc2pred", _PRESENTATION),
    "prime": ("prime", _IMMUNOGENICITY),
    "deepimmuno": ("deepimmuno", _IMMUNOGENICITY),
    "immuscopeim": ("immuscope_im", _IMMUNOGENICITY),
    "tlimm": ("tlimm", _IMMUNOGENICITY),
    "tlimmuno2": ("tlimmuno2", _IMMUNOGENICITY),
    "calis": ("calis", _IMMUNOGENICITY),
    "eramer": ("eramer", "erap_trimming"),
    "deeptap": ("deeptap", "tap_transport"),
    # Other mhctools predictors. Explicit metric text or a mode suffix can
    # specialize the ones with no safe default.
    "bigmhc": ("bigmhc", None),
    "netmhcstabpan": ("netmhcstabpan", _STABILITY),
    "netchop": ("netchop", "proteasome_cleavage"),
    "netcleave": ("netcleave", None),
    "netcleavei": ("netcleave", "proteasome_cleavage"),
    "netcleaveii": ("netcleave", "endolysosomal_cleavage"),
    "pepsickle": ("pepsickle", "proteasome_cleavage"),
}

_MODE_SUFFIXES = (
    ("immunogenicity", _IMMUNOGENICITY, "im"),
    ("affinity", _AFFINITY, "affinity"),
    ("processing", _PROCESSING, "processing"),
    ("presentation", _PRESENTATION, "presentation"),
    ("proc", _PROCESSING, "processing"),
    ("pres", _PRESENTATION, "presentation"),
    ("aff", _AFFINITY, "affinity"),
    ("ba", _AFFINITY, "ba"),
    ("el", _PRESENTATION, "el"),
    ("im", _IMMUNOGENICITY, "im"),
)


def _model_spec(model_name):
    """Return ``(method, default_kind)`` for a known model/mode spelling."""
    normalized = _normalized(model_name).replace("_", "")
    direct = _MODEL_DEFAULTS.get(normalized)
    if direct is not None:
        return direct

    for suffix, kind, mode_name in _MODE_SUFFIXES:
        if not normalized.endswith(suffix):
            continue
        base = _MODEL_DEFAULTS.get(normalized[:-len(suffix)])
        if base is None:
            continue
        method = base[0]
        # BigMHC exposes EL and IM as distinct predictor method names in
        # mhctools. Other mode suffixes describe the kind of one method.
        if method == "bigmhc":
            method = f"bigmhc_{mode_name}"
        return method, kind
    return None


_KIND_TOKENS = {
    _AFFINITY: frozenset({"affinity", "aff", "binding", "ba", "ic50"}),
    _PRESENTATION: frozenset({
        "presentation", "pres", "el", "elution", "ligand",
    }),
    _PROCESSING: frozenset({"processing", "proc"}),
    _IMMUNOGENICITY: frozenset({
        "immunogenicity", "immunogenic", "immuno", "im",
    }),
    _STABILITY: frozenset({"stability", "stable", "stab", "halflife", "thalf"}),
    "tap_transport": frozenset({"tap"}),
    "erap_trimming": frozenset({"erap"}),
    "proteasome_cleavage": frozenset({"proteasome", "proteasomal"}),
    "endolysosomal_cleavage": frozenset({
        "endolysosomal", "lysosomal",
    }),
}
_RANK_TOKENS = frozenset({
    "percentile", "percentilerank", "perc", "pct", "rank",
})
_SCORE_TOKENS = frozenset({"score", "prediction", "pred"})
_VALUE_TOKENS = frozenset({
    "aff", "affinity", "ic50", "halflife", "thalf", "value",
})
_SEQUENCE_TOKENS = {
    "mt": "mt",
    "mut": "mt",
    "mutant": "mt",
    "wt": "wt",
    "wildtype": "wt",
    "wild": "wt",
}


def parse_prediction_metric(model_name, metric_name) -> PredictionMetric | None:
    """Classify an external prediction metric using a shared vocabulary.

    The parser accepts separators and case interchangeably. Model suffixes
    ``EL`` (presentation), ``BA`` / ``Aff`` / ``Affinity`` (affinity), and
    ``IM`` (immunogenicity) supply a kind when a metric is otherwise only
    ``MT Score``, ``WT Score``, or a percentile. Explicit quantity words in
    the metric take precedence, so ``MHCflurryEL`` + ``Processing WT
    Percentile`` is correctly classified as processing rather than
    presentation.

    Parameters
    ----------
    model_name : str
        Predictor name, optionally carrying a mode suffix.
    metric_name : str
        Metric label. Token order is flexible; MT/WT and score/rank words may
        appear before or after the quantity.

    Returns
    -------
    PredictionMetric or None
        Parsed semantics, or ``None`` when the label does not say enough to
        choose a kind and field without guessing.

    Examples
    --------
    ``parse_prediction_metric("NetMHCpanEL", "MT Score")`` returns a
    presentation score for mutant peptide. ``parse_prediction_metric(
    "MHCflurryEL", "Processing WT Percentile")`` returns a processing
    percentile rank for wildtype peptide.
    """
    if not isinstance(model_name, str) or not isinstance(metric_name, str):
        return None
    model = _model_spec(model_name)
    metric = _normalized(metric_name)
    if not metric:
        return None
    tokens = frozenset(token for token in metric.split("_") if token)

    explicit_kinds = {
        kind for kind, aliases in _KIND_TOKENS.items()
        if tokens & aliases
    }
    # "binding affinity" names the same kind twice, but genuinely different
    # quantities in one label are ambiguous and should not be guessed.
    if len(explicit_kinds) > 1:
        return None
    if explicit_kinds:
        kind = next(iter(explicit_kinds))
    elif model is not None:
        kind = model[1]
    else:
        kind = None
    if kind is None:
        return None

    if tokens & _RANK_TOKENS:
        field = "percentile_rank"
    elif "ic50" in tokens:
        # pVACtools calls nM affinity an "IC50 Score". It is Topiary's raw
        # value, not a normalized higher-is-better score.
        field = "value"
    elif tokens & _SCORE_TOKENS:
        field = "score"
    elif tokens & _VALUE_TOKENS:
        field = "value"
    elif explicit_kinds:
        # Quantity-only shorthands follow mhctools annotate-table semantics.
        field = "value" if kind in {_AFFINITY, _STABILITY} else "score"
    else:
        return None

    sequences = {
        sequence for token, sequence in _SEQUENCE_TOKENS.items()
        if token in tokens
    }
    if len(sequences) > 1:
        return None
    sequence = next(iter(sequences)) if sequences else None

    method = model[0] if model is not None else _normalized(model_name)
    return PredictionMetric(method, kind, field, sequence)
