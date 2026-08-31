"""LENS (Landscape of Effective Neoantigens Software) report loader.

Reads LENS TSV reports (v1.4, v1.5.1, v1.9) into Topiary's wide-form
schema with a :class:`TopiaryResult` return.  Binding columns are
remapped to Topiary's ``{model}_{kind}_{field}`` convention; per-model
versions go into the metadata comment block.  LENS-specific columns
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
# The version is extracted here so we can populate ``Metadata.models``;
# the emitted Topiary wide-form column name uses just ``{model}_{kind}_{field}``
# (no version), matching the convention of ``to_wide`` / ``from_wide`` when
# there is no multi-version collision within a single file.
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


def _parse_binding_column(column):
    """``(tool, version, kind, field)`` for a LENS binding column.

    Returns ``None`` when the column isn't predictor-shaped at all, and
    ``(tool, version, None, None)`` when it is but names a tool or
    metric the table doesn't cover — the caller reports those rather
    than dropping them silently.
    """
    if not isinstance(column, str):
        return None
    match = _BINDING_COLUMN.match(column)
    if match is None:
        return None
    tool, version, metric = match.groups()
    spec = _BINDING_METRICS.get((tool.lower(), metric.lower()))
    if spec is None:
        return tool.lower(), version, None, None
    kind, field = spec
    return tool.lower(), version, kind, field


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


def read_lens(path, tag: str | None = None) -> TopiaryResult:
    """Read a LENS TSV report into a :class:`TopiaryResult`.

    Parameters
    ----------
    path : str or Path
        LENS TSV report.
    tag : str, optional
        Source label for :class:`Metadata.sources`. Defaults to the
        filename.

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

    df, models = _remap_binding_columns(df)
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

    return TopiaryResult(df, meta)


# =============================================================================
# Internals
# =============================================================================


def _remap_binding_columns(df: pd.DataFrame):
    """Rename LENS binding columns to Topiary wide form.

    Returns ``(df, models_dict)`` where *models_dict* maps method
    name → version for every binding model present.
    """
    parsed_columns = []
    unmapped = []
    versions_by_model = {}
    for column in df.columns:
        parsed = _parse_binding_column(column)
        if parsed is None:
            continue
        model, version, kind, field = parsed
        if kind is None:
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
    collided = {
        model: sorted(set(versions))
        for model, versions in versions_by_model.items()
        if len(set(versions)) > 1
    }
    if collided:
        import warnings
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
    for column, model, version, kind, field in parsed_columns:
        if model in collided:
            rename[column] = f"{model}_{version}_{kind}_{field}"
            models[f"{model}_{version}"] = version
        else:
            rename[column] = f"{model}_{kind}_{field}"
            models[model] = version
    if unmapped:
        import warnings
        warnings.warn(
            f"LENS binding column(s) {sorted(unmapped)} look like predictor "
            f"output but name a tool or metric this topiary doesn't know, so "
            f"they are left unnormalized. Their values are still in the "
            f"frame under the original names — nothing was dropped — but no "
            f"kind or field was assigned to them.",
            UserWarning, stacklevel=3,
        )
    return df.rename(columns=rename), models


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
