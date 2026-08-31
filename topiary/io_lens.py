"""LENS (Landscape of Effective Neoantigens Software) report loader.

Reads LENS TSV reports (v1.4, v1.5.1, v1.9) into Topiary's wide-form
schema with a :class:`TopiaryResult` return.  Binding columns are
remapped to Topiary's ``{model}_{kind}_{field}`` convention, or
``{model}_{version}_{kind}_{field}`` where one tool appears at two
versions for the same metric; per-model versions go into the metadata
comment block.  LENS-specific columns
(``erv_*``, ``priority_score_*``, ``b2m_*``, etc.) pass through as
annotation columns for use via ``Column("...")`` in the DSL.

Schema losses worth knowing about:

- ``peptide_offset`` is set to 0 for every row — LENS doesn't record
  the peptide's position within its source protein.
- ``contains_mutant_residues`` / ``mutation_start_in_peptide`` /
  ``mutation_end_in_peptide`` are left NaN. LENS's ``mut_aa_pos``
  semantics are ambiguous across antigen_source types.
- ``n_flank`` / ``c_flank`` are derived from ``pep_context`` only for
  SNV / SPLICE / FUSION rows where the peptide appears exactly once in
  the context. ERV / INDEL / CTA contexts may be the full source ORF,
  so flank derivation is skipped.
- LENS-computed agretopicity / priority scores pass through as columns
  but have no sibling in fresh Topiary predictions — re-predicting
  produces rows with NaN for these fields.
- ``b2m_*`` / ``tap*_*`` / ``hla_allele_*`` are per-sample constants
  that LENS repeats on every row; we carry them the same way. A
  future change may promote these to :class:`Metadata` ``extra``.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .io import Metadata
from .result import TopiaryResult

logger = logging.getLogger(__name__)


# =============================================================================
# Column maps and version markers
# =============================================================================

# LENS binding column → (model, version, kind, wide-field)
#
# The version is extracted here so we can populate ``Metadata.models``.
# The emitted wide-form name is ``{model}_{kind}_{field}`` — no version —
# except where one tool appears with several versions for the same
# metric, when the version is included to keep both, matching what
# ``to_wide`` does in the same situation.
#: LENS names a binding column ``<tool>_<version>.<metric>``. The
#: version is opaque: ``aff_nm`` is an IC50 whether NetMHCpan 4.1, 4.1b
#: or 4.2 produced it. Keying the table on the version too meant a
#: spelling topiary had not seen dropped that predictor's whole axis
#: from the frame, with nothing raised — so the table is keyed on what
#: determines meaning, and the version is recorded rather than matched.
_BINDING_METRICS = {
    ("netmhcpan", "aff_nm"):              ("affinity",           "value"),
    ("netmhcpan", "score_ba"):            ("affinity",           "score"),
    ("netmhcpan", "perc_rank_ba"):        ("affinity",           "rank"),
    ("netmhcpan", "score_el"):            ("presentation",       "score"),
    ("netmhcpan", "perc_rank_el"):        ("presentation",       "rank"),
    ("mhcflurry", "aff"):                 ("affinity",           "value"),
    ("mhcflurry", "aff_perc"):            ("affinity",           "rank"),
    ("mhcflurry", "proc_score"):          ("antigen_processing", "score"),
    ("mhcflurry", "pres_score"):          ("presentation",       "score"),
    ("mhcflurry", "pres_perc"):           ("presentation",       "rank"),
    ("netmhcstabpan", "stab_pred_score"): ("stability",          "score"),
    ("netmhcstabpan", "halflife_hours"):  ("stability",          "value"),
    ("netmhcstabpan", "perc_rank_stab"):  ("stability",          "rank"),
}

#: ``<tool>_<version>.<metric>`` — the shape of a LENS binding column.
#: A column matching this that the table doesn't cover is a predictor
#: topiary hasn't met, which is worth saying out loud rather than
#: leaving as an absence.
_BINDING_COLUMN = re.compile(r"^([A-Za-z][A-Za-z0-9]*)_(\d[\w.]*)\.(.+)$")


def _parse_binding_column(column, binding_metrics=None):
    """``(tool, version, kind, field)`` for a LENS binding column.

    Returns ``None`` when the column isn't predictor-shaped at all, and
    ``(tool, version, None, None)`` when it is but names a tool or
    metric the table doesn't cover — the caller reports those rather
    than dropping them silently.

    *binding_metrics* is a caller override, already normalized and
    merged over the built-in table by :func:`_resolve_binding_metrics`.
    A key mapped to ``None`` there is one the caller has declared is not
    a prediction: it reads as unmapped, but is not reported as a gap.
    """
    if not isinstance(column, str):
        return None
    match = _BINDING_COLUMN.match(column)
    if match is None:
        return None
    tool, version, metric = match.groups()
    table = _BINDING_METRICS if binding_metrics is None else binding_metrics
    key = (tool.lower(), metric.lower())
    if key not in table:
        return tool.lower(), version, None, None
    spec = table[key]
    if spec is None:
        # Acknowledged by the caller as a non-prediction column. It
        # still passes through to the frame as an annotation column;
        # what changes is only that it is no longer reported as a gap.
        return tool.lower(), version, None, _ACKNOWLEDGED
    kind, field = spec
    return tool.lower(), version, kind, field


#: Sentinel in the ``field`` slot for a column the caller declared is
#: not a prediction, to tell "topiary doesn't know this column" from
#: "the caller says there is nothing to know".
_ACKNOWLEDGED = object()


def _resolve_binding_metrics(binding_metrics):
    """Merge a caller's binding-metric overrides over the built-in table.

    Keys are ``(tool, metric)`` — the same key the built-in table uses,
    and the same pair the unmapped-column warning names, so what topiary
    tells you is what you pass back. Deliberately *not* the raw column
    name: that would carry the version, and a mapping that stops working
    when a file spells the version differently is the brittleness #206
    removed.

    Values are ``(kind, field)``, or ``None`` to declare the column a
    non-prediction — which silences the unmapped warning for it and
    leaves the column itself untouched.
    """
    if binding_metrics is None:
        return None
    if not isinstance(binding_metrics, Mapping):
        raise TypeError(
            f"binding_metrics must be a mapping of (tool, metric) -> "
            f"(kind, field), got {type(binding_metrics).__name__}"
        )
    from .wide import WIDE_FIELDS

    known_kinds = _known_short_kinds()
    resolved = dict(_BINDING_METRICS)
    for key, spec in binding_metrics.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(part, str) for part in key)
        ):
            raise ValueError(
                f"binding_metrics keys must be (tool, metric) string "
                f"pairs, got {key!r}"
            )
        tool, metric = (part.strip().lower() for part in key)
        if spec is None:
            resolved[(tool, metric)] = None
            continue
        if not isinstance(spec, tuple) or len(spec) != 2:
            raise ValueError(
                f"binding_metrics[{key!r}] must be a (kind, field) pair, "
                f"or None to declare it a non-prediction column; "
                f"got {spec!r}"
            )
        kind, field = spec
        # Validate up front. An unknown kind or field would emit a
        # wide-form column name that ``_parse_wide_column`` cannot read
        # back, so the data would reach the frame and then vanish on
        # ``to_long()`` — the failure mode this whole area keeps
        # producing, and not one to hand a caller a new way to cause.
        if not isinstance(kind, str) or kind not in known_kinds:
            raise ValueError(
                f"binding_metrics[{key!r}] names an unknown kind "
                f"{kind!r}. Use a short kind name, one of "
                f"{sorted(known_kinds)}."
            )
        if field not in WIDE_FIELDS:
            raise ValueError(
                f"binding_metrics[{key!r}] names an unknown field "
                f"{field!r}. Use one of {sorted(WIDE_FIELDS)}."
            )
        resolved[(tool, metric)] = (str(kind), str(field))
    return resolved


def _known_short_kinds():
    from .wide import _known_kind_short_names
    return _known_kind_short_names()


# LENS metadata column → Topiary column (pass-through rename).
_ANNOTATION_RENAME = {
    "gene_name": "gene",
    "variant_coords": "variant",
    "variant_effect": "effect",  # v1.5.1+; NaN for earlier
    # tpm has special handling (fusion composite strings) — don't blind-rename.
}

# Columns used by version detection.
#: Columns used by version detection.  The v1.4 marker is expressed as
#: a (tool, metric) pair for the same reason the binding table is: a
#: NetMHCstabPan release topiary hasn't seen should not make a v1.4 file
#: undetectable.
_VERSION_MARKERS = [
    ("v1.9",   {"lohhla_allele_loss_pval"}),
    ("v1.5.1", {"snaf_exp"}),
]
_VERSION_MARKER_METRICS = [
    ("v1.4", ("netmhcstabpan", "stab_pred_score")),
]


# =============================================================================
# Public entry points
# =============================================================================


def detect_lens_version(columns) -> str | None:
    """Return ``'v1.4'`` / ``'v1.5.1'`` / ``'v1.9'``, or ``None``."""
    cols = set(columns)
    for version, markers in _VERSION_MARKERS:
        if markers <= cols:
            return version
    present = set()
    for column in cols:
        parsed = _parse_binding_column(column)
        if parsed is not None:
            tool, _, _, _ = parsed
            match = _BINDING_COLUMN.match(column)
            present.add((tool, match.group(3).lower()))
    for version, marker in _VERSION_MARKER_METRICS:
        if marker in present:
            return version
    return None


def read_lens(path, tag: str | None = None, *,
              binding_metrics=None) -> TopiaryResult:
    """Read a LENS TSV report into a :class:`TopiaryResult`.

    Parameters
    ----------
    path : str or Path
        LENS TSV report.
    tag : str, optional
        Source label for :class:`Metadata.sources`. Defaults to the
        filename.
    binding_metrics : mapping, optional
        Overrides for the built-in binding-column map, merged over it
        rather than replacing it — patch one column without restating
        the rest.  Keys are ``(tool, metric)``, the same pair the
        unmapped-column warning names, so what topiary tells you is what
        you pass back::

            read_lens(path, binding_metrics={
                ("netmhcpan", "el_score"): ("presentation", "score"),
                ("sometool", "noisy_metric"): None,   # not a prediction
            })

        Values are ``(kind, field)`` with *kind* a short kind name
        (``"affinity"``, ``"presentation"``, ...) and *field* one of
        ``"value"`` / ``"score"`` / ``"rank"``.  Both parts are
        validated up front, since an unreadable wide-form name would
        put data in the frame that ``to_long()`` then silently drops.

        ``None`` says "this one is not a prediction" — it silences the
        unmapped-column warning without remapping anything.  The column
        is left alone, passing through as an annotation column reachable
        as ``Column("sometool_1.0.noisy_metric")``; overriding a mapping
        does not delete data.

        Keys are deliberately version-free: a mapping keyed on the raw
        column name would stop working the moment a file spelled the
        version differently.

    Returns
    -------
    TopiaryResult
        Wide-form DataFrame with binding columns remapped,
        :class:`Metadata` carrying the detected LENS version and the
        (method → version) map for every binding model found.
    """
    path = Path(path)

    df = pd.read_csv(path, sep="\t", na_values=["NA"])

    version = detect_lens_version(df.columns)
    if version is None:
        logger.warning(
            "Could not detect LENS version from columns of %s; "
            "proceeding with best-effort mapping", path.name,
        )

    df, models, model_key_parts = _remap_binding_columns(
        df, _resolve_binding_metrics(binding_metrics),
    )
    df = _normalize_alleles(df)
    df = _rename_annotations(df)
    df = _handle_tpm(df)
    df = _derive_peptide_columns(df)
    df = _derive_effect_type(df)
    df = _add_source_sequence_name(df)
    df["peptide_offset"] = 0

    source_label = tag or f"lens-{version}" if version else (tag or path.name)

    meta = Metadata(
        form="wide",
        models=models,
        sources=[source_label],
    )
    if version is not None:
        meta.extra["lens_version"] = version
    # ``models`` is keyed by method, so it cannot express a file that
    # carries two versions of one tool. Record what each emitted model
    # key was actually built from, which is what ``from_wide`` reads to
    # split a key back into method and version.
    meta.extra["topiary_model_keys"] = model_key_parts
    df.attrs["topiary_model_keys"] = model_key_parts

    return TopiaryResult(df, meta)


# =============================================================================
# Internals
# =============================================================================


def _remap_binding_columns(df: pd.DataFrame, binding_metrics=None):
    """Rename LENS binding columns to Topiary wide form.

    Returns ``(df, models_dict, model_key_parts)``.  *models_dict* maps
    method name → version for every binding model present, keeping that
    shape even when a method has several versions (it then holds the
    first).  *model_key_parts* maps each emitted model key to its
    ``[method, version]``, which is what lets ``from_wide`` recover the
    two separately.
    """
    parsed_columns = []
    unmapped = []
    versions_by_model = {}
    for column in df.columns:
        parsed = _parse_binding_column(column, binding_metrics)
        if parsed is None:
            continue
        model, version, kind, field = parsed
        if kind is None:
            # A column the caller has acknowledged is not a gap.
            if field is not _ACKNOWLEDGED:
                unmapped.append(column)
            continue
        parsed_columns.append((column, model, version, kind, field))
        versions_by_model.setdefault(model, []).append(version)

    # A LENS table may legitimately carry two versions of one tool. The
    # emitted name has no room for a version, so both would land on the
    # same column and one set of values would be lost — silently, since
    # the collision only surfaces later as a pandas duplicate-column
    # warning that doesn't name the predictor. Qualify the name instead,
    # the way ``to_wide`` does for the same situation.
    #
    # The test is whether two versions claim the *same output column*,
    # not whether two version strings appear anywhere for the tool: a
    # file that spells one run's version inconsistently across metrics
    # (``netmhcpan_4.1b.aff_nm`` beside ``netmhcpan_4.1.score_ba``) has
    # no collision, and splitting it into two predictors would undo
    # exactly what keying on (tool, metric) was for.
    versions_by_output = {}
    for column, model, version, kind, field in parsed_columns:
        versions_by_output.setdefault((model, kind, field), set()).add(version)
    collided_models = {
        model for (model, _, _), versions in versions_by_output.items()
        if len(versions) > 1
    }
    collided = {
        model: sorted(set(versions))
        for model, versions in versions_by_model.items()
        if model in collided_models
    }
    if collided:
        warnings.warn(
            "; ".join(
                f"{model} appears with versions {versions}"
                for model, versions in sorted(collided.items())
            )
            + ". Including the version in the column names so both are "
            "kept — the emitted names are "
            "'{tool}_{version}_{kind}_{field}' for these tools.",
            UserWarning, stacklevel=3,
        )

    rename = {}
    models = {}
    model_key_parts = {}
    for column, model, version, kind, field in parsed_columns:
        if model in collided:
            key = f"{model}_{version}"
            rename[column] = f"{key}_{kind}_{field}"
            model_key_parts[key] = [model, version]
        else:
            rename[column] = f"{model}_{kind}_{field}"
            model_key_parts[model] = [model, version]
        # ``models`` keeps its documented shape — method name → version —
        # so ``models["netmhcpan"]`` still answers for every file. When one
        # method has several versions it can only hold one of them; the
        # full mapping lives in ``model_key_parts``, which is what
        # ``from_wide`` reads to recover method and version separately.
        models.setdefault(model, version)
    if unmapped:
        warnings.warn(
            f"LENS binding column(s) {sorted(unmapped)} look like predictor "
            f"output but name a tool or metric this topiary doesn't know, so "
            f"they are left unnormalized. Their values are still in the "
            f"frame under the original names — nothing was dropped — but no "
            f"kind or field was assigned to them.",
            UserWarning, stacklevel=3,
        )
    return df.rename(columns=rename), models, model_key_parts


def _normalize_alleles(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ``allele`` via mhcgnomes (handles Class I / II / mouse)."""
    if "allele" not in df.columns:
        return df
    import mhcgnomes

    cache: dict[str, str] = {}

    def _norm(raw):
        if not isinstance(raw, str) or not raw:
            return raw
        if raw in cache:
            return cache[raw]
        try:
            out = mhcgnomes.parse(raw).to_string()
        except Exception:  # noqa: BLE001 — mhcgnomes raises many types
            out = raw
        cache[raw] = out
        return out

    df = df.copy()
    df["allele"] = df["allele"].map(_norm)
    return df


def _rename_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the few LENS annotation columns that map 1:1 to Topiary."""
    rename = {src: dst for src, dst in _ANNOTATION_RENAME.items() if src in df.columns}
    return df.rename(columns=rename)


def _handle_tpm(df: pd.DataFrame) -> pd.DataFrame:
    """LENS ``tpm`` is numeric for most rows but a composite string
    ``ENST1:tpm1-ENST2:tpm2`` for fusion rows.  Emit a numeric
    ``gene_tpm`` (NaN for fusion rows) and preserve the raw string in
    ``gene_tpm_raw``.
    """
    if "tpm" not in df.columns:
        return df
    df = df.copy()
    df["gene_tpm_raw"] = df["tpm"]
    df["gene_tpm"] = pd.to_numeric(df["tpm"], errors="coerce")
    df = df.drop(columns=["tpm"])
    # If every raw value coerced cleanly, drop the raw column.
    if df["gene_tpm"].notna().sum() == df["gene_tpm_raw"].notna().sum():
        df = df.drop(columns=["gene_tpm_raw"])
    return df


def _derive_peptide_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``peptide_length``, ``n_flank``, ``c_flank``.

    Flanks are derived from ``pep_context`` only for SNV / SPLICE /
    FUSION where the context is a short window.  ERV / INDEL / CTA
    contexts are the full ORF (or ambiguous), so flanks are left NaN
    for those sources.
    """
    if "peptide" not in df.columns:
        return df
    df = df.copy()
    df["peptide_length"] = df["peptide"].str.len()

    flank_ok_sources = {"SNV", "SPLICE", "FUSION"}
    n_flank = []
    c_flank = []
    has_context = "pep_context" in df.columns
    has_source = "antigen_source" in df.columns
    for idx in df.index:
        pep = df.at[idx, "peptide"] if has_context else None
        ctx = df.at[idx, "pep_context"] if has_context else None
        src = df.at[idx, "antigen_source"] if has_source else None
        if (
            not isinstance(pep, str) or not isinstance(ctx, str)
            or src not in flank_ok_sources
        ):
            n_flank.append(pd.NA)
            c_flank.append(pd.NA)
            continue
        pos = ctx.find(pep)
        # Require exactly one occurrence; ambiguous matches → NaN.
        if pos < 0 or ctx.find(pep, pos + 1) >= 0:
            n_flank.append(pd.NA)
            c_flank.append(pd.NA)
            continue
        n_flank.append(ctx[:pos])
        c_flank.append(ctx[pos + len(pep):])
    df["n_flank"] = n_flank
    df["c_flank"] = c_flank
    return df


# HGVS → Topiary effect_type (varcode-style class names).
# Order matters: more specific patterns first.
_HGVS_EFFECT_RULES = [
    (re.compile(r"fs\b"),        "FrameShift"),
    (re.compile(r"del$|del(ins)?\b"), "Deletion"),
    (re.compile(r"ins\b|dup\b"), "Insertion"),
    (re.compile(r"\*"),          "StopGain"),
    (re.compile(r"="),           "Silent"),
    # Fallback for p.X{pos}{Y} substitutions.
    (re.compile(r"^p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$"), "Substitution"),
]

_ANTIGEN_SOURCE_FALLBACK = {
    "SNV":      "Substitution",
    "INDEL":    "Indel",
    "SPLICE":   "Splice",
    "FUSION":   "Fusion",
    "ERV":      "ERV",
    "CTA/SELF": "Self",
}


def _derive_effect_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``effect_type`` derived from HGVS ``effect`` when present,
    else fall back to ``antigen_source``."""
    df = df.copy()
    has_effect = "effect" in df.columns
    has_source = "antigen_source" in df.columns
    effect_types = []
    for idx in df.index:
        val = None
        if has_effect:
            hgvs = df.at[idx, "effect"]
            if isinstance(hgvs, str):
                for pat, label in _HGVS_EFFECT_RULES:
                    if pat.search(hgvs):
                        val = label
                        break
        if val is None and has_source:
            src = df.at[idx, "antigen_source"]
            if isinstance(src, str):
                val = _ANTIGEN_SOURCE_FALLBACK.get(src)
        effect_types.append(val)
    df["effect_type"] = effect_types
    return df


def _add_source_sequence_name(df: pd.DataFrame) -> pd.DataFrame:
    """Synthesize a ``source_sequence_name`` from ``antigen_source`` +
    ``origin_descriptor`` so Topiary's group key is well-defined.

    For rows without ``origin_descriptor`` we fall back to just the
    ``antigen_source`` (uncommon in practice — most LENS rows have it).
    """
    df = df.copy()
    has_src = "antigen_source" in df.columns
    has_origin = "origin_descriptor" in df.columns
    if not has_src:
        df["source_sequence_name"] = pd.NA
        return df

    def _synth(row):
        src = row.get("antigen_source")
        origin = row.get("origin_descriptor") if has_origin else None
        if not isinstance(src, str):
            return pd.NA
        if isinstance(origin, str) and origin:
            return f"{src}:{origin}"
        return src

    df["source_sequence_name"] = df.apply(_synth, axis=1)
    return df
