"""pVACseq report loader.

Parses both pVACtools output flavors into Topiary long form:

- ``*.all_epitopes.aggregated.tsv`` — one row per variant.  pVACseq picks
  a single Best Peptide × Allele based on its Median IC50 across the
  configured prediction algorithms.
- ``*.all_epitopes.tsv`` — one row per candidate peptide × allele ×
  length.

Both flavors map onto the same long-form schema with
``prediction_method_name="pvacseq"`` and ``kind="pMHC_affinity"``.  The
file's Median MT IC50 / percentile populate the primary
``value`` / ``percentile_rank``; the WT companions populate
``wt_value`` / ``wt_percentile_rank`` so DSL expressions like
``wt(Affinity.value)`` work without further setup.

For missense aggregated rows, the WT peptide sequence is reconstructed
from ``Best Peptide`` + ``Pos`` + ``AA Change`` (the aggregated TSV
itself doesn't carry the WT sequence).  Indel / frameshift / multi-AA
rows leave ``wt_peptide`` NaN; users wanting full WT context for those
should load the unaggregated ``all_epitopes.tsv`` flavor, which has a
``WT Epitope Seq`` column.

Per-algorithm score columns in the all_epitopes flavor (e.g.
"NetMHCpan MT IC50 Score", "MHCflurry WT Percentile") pass through as
snake_cased ``pvacseq_<algo>_{ic50,pct}_{mt,wt}`` annotation columns,
accessible from the DSL via ``Column("...")``.  They aren't melted into
extra ``prediction_method_name`` rows, so the DSL's ``Affinity['netmhcpan']``
selector won't reach them — callers wanting per-algorithm DSL access
should melt them out themselves or re-predict via :class:`TopiaryPredictor`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .io import Metadata
from .result import TopiaryResult

logger = logging.getLogger(__name__)


# =============================================================================
# Format detection
# =============================================================================

_AGG_SIGNATURE = frozenset({"Best Peptide", "IC50 MT", "%ile MT", "Allele"})
_AE_SIGNATURE = frozenset({"MT Epitope Seq", "HLA Allele"})
_AE_MEDIAN = frozenset({"Median MT IC50 Score", "Median MT Percentile"})
_AE_BEST = frozenset({"Best MT IC50 Score", "Best MT Percentile"})


def detect_pvacseq_format(columns) -> str | None:
    """Return ``'aggregated'``, ``'all_epitopes'``, or ``None``."""
    cols = set(columns)
    if _AGG_SIGNATURE <= cols:
        return "aggregated"
    if _AE_SIGNATURE <= cols and (_AE_MEDIAN <= cols or _AE_BEST <= cols):
        return "all_epitopes"
    return None


# =============================================================================
# Allele / sequence helpers
# =============================================================================


def _normalize_alleles(values):
    """Run each allele string through mhcgnomes; unparseable values pass through."""
    import mhcgnomes

    cache: dict[str, str] = {}

    def _norm(raw):
        if not isinstance(raw, str) or not raw:
            return raw
        if raw in cache:
            return cache[raw]
        try:
            out = mhcgnomes.parse(raw).to_string()
        except Exception:  # noqa: BLE001 — mhcgnomes raises many exception types
            out = raw
        cache[raw] = out
        return out

    return [_norm(v) for v in values]


# Matches the pVACseq aggregated AA Change format for a single missense:
# "E806V" → wt='E', mt='V'.  Indels / multi-residue / frameshift formats
# (e.g. "EE764-765EK", "FS342", "SNNDRL233-238S") don't match, so WT
# reconstruction is skipped — there's no unambiguous way to recover the
# WT peptide for them without the upstream variants TSV.
_MISSENSE_AA_CHANGE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _reconstruct_wt_peptide(mt_peptide, pos_in_peptide, aa_change):
    """Recover the WT peptide for a single-AA missense; else return None."""
    if not isinstance(mt_peptide, str) or not isinstance(aa_change, str):
        return None
    m = _MISSENSE_AA_CHANGE.match(aa_change.strip())
    if m is None:
        return None
    wt_aa, _, mt_aa = m.groups()
    try:
        idx = int(pos_in_peptide) - 1  # pVACseq Pos is 1-based
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(mt_peptide):
        return None
    if mt_peptide[idx] != mt_aa:
        # Pos / AA Change disagree with the peptide — skip rather than guess.
        return None
    return mt_peptide[:idx] + wt_aa + mt_peptide[idx + 1:]


# =============================================================================
# Effect-type derivation
# =============================================================================

_VARIANT_TYPE_MAP = {
    "missense":    "Substitution",
    "inframe_ins": "Insertion",
    "inframe_del": "Deletion",
    "frameshift":  "FrameShift",
    "fs":          "FrameShift",
}


def _classify_effect(aa_change=None, variant_type=None):
    """Best-effort varcode-style ``effect_type`` from pVACseq columns.

    ``variant_type`` (from the all_epitopes flavor) wins when set; the
    aggregated flavor only has ``AA Change`` strings, classified by shape.
    """
    if isinstance(variant_type, str):
        mapped = _VARIANT_TYPE_MAP.get(variant_type.strip().lower())
        if mapped is not None:
            return mapped
    if isinstance(aa_change, str):
        s = aa_change.strip()
        if s.upper().startswith("FS") or "fs" in s.lower():
            return "FrameShift"
        if _MISSENSE_AA_CHANGE.match(s):
            return "Substitution"
        if re.search(r"\bdel\b|del$", s, re.IGNORECASE):
            return "Deletion"
        if re.search(r"\bins\b|ins$|\bdup\b", s, re.IGNORECASE):
            return "Insertion"
        # Multi-residue replacement (e.g. "EE764-765EK")
        if re.match(r"^[A-Z]+\d+(-\d+)?[A-Z]+$", s):
            return "Substitution"
    return None


# =============================================================================
# Column maps
# =============================================================================

# (pVACseq column → topiary column) for aggregated-flavor annotations.
_AGG_ANNOTATIONS = {
    "Gene":                    "gene",
    "Best Transcript":         "transcript",
    "ID":                      "variant",
    "RNA Expr":                "rna_transcript_expression",
    "RNA VAF":                 "rna_vaf",
    "Allele Expr":             "allele_expression",
    "RNA Depth":               "rna_depth",
    "DNA VAF":                 "dna_vaf",
    "Tier":                    "pvacseq_tier",
    "Num Passing Transcripts": "pvacseq_num_passing_transcripts",
    "Num Included Peptides":   "pvacseq_num_included_peptides",
    "Num Passing Peptides":    "pvacseq_num_passing_peptides",
    "MANE Select":             "mane_select",
    "Canonical":               "canonical",
    "TSL":                     "transcript_support_level",
    "AA Change":               "aa_change",
    "Mutation Position":       "mutation_position",
    "Ref Match":               "pvacseq_ref_match",
    "Evaluation":              "pvacseq_evaluation",
    "Prob Pos":                "pvacseq_prob_pos",
}

# (pVACseq column → topiary column) for all_epitopes-flavor annotations.
_AE_ANNOTATIONS = {
    "Gene Name":             "gene",
    "Transcript":            "transcript",
    "Gene Expression":       "gene_expression",
    "Transcript Expression": "transcript_expression",
    "Tumor DNA Depth":       "tumor_dna_depth",
    "Tumor DNA VAF":         "tumor_dna_vaf",
    "Tumor RNA Depth":       "tumor_rna_depth",
    "Tumor RNA VAF":         "tumor_rna_vaf",
    "Normal Depth":          "normal_depth",
    "Normal VAF":            "normal_vaf",
    "Protein Position":      "protein_position",
    "HGVSc":                 "hgvsc",
    "HGVSp":                 "hgvsp",
    "Mutation":              "aa_change",
    "Variant Type":          "variant_type",
    "Mutation Position":     "mutation_position",
}

# Matches "<Algo> {MT|WT} {IC50 Score|Percentile}" in all_epitopes.
_PER_ALGO_RE = re.compile(
    r"^(?P<algo>[\w\.\-]+) (?P<mtwt>MT|WT) (?P<field>IC50 Score|Percentile)$"
)
_FIELD_SHORT = {"IC50 Score": "ic50", "Percentile": "pct"}


# =============================================================================
# Per-flavor parsing
# =============================================================================


def _parse_aggregated(df):
    """Aggregated-format DataFrame → parsed-rows DataFrame."""
    out = pd.DataFrame(index=df.index)
    out["peptide"] = df["Best Peptide"]
    out["allele"] = _normalize_alleles(df["Allele"])
    out["value"] = pd.to_numeric(df["IC50 MT"], errors="coerce")
    out["percentile_rank"] = pd.to_numeric(df["%ile MT"], errors="coerce")
    out["wt_value"] = (
        pd.to_numeric(df["IC50 WT"], errors="coerce")
        if "IC50 WT" in df.columns else np.nan
    )
    out["wt_percentile_rank"] = (
        pd.to_numeric(df["%ile WT"], errors="coerce")
        if "%ile WT" in df.columns else np.nan
    )

    pos = df["Pos"] if "Pos" in df.columns else [None] * len(df)
    aa_change = df["AA Change"] if "AA Change" in df.columns else [None] * len(df)
    out["wt_peptide"] = [
        _reconstruct_wt_peptide(pep, p, a)
        for pep, p, a in zip(df["Best Peptide"], pos, aa_change)
    ]
    out["effect_type"] = [_classify_effect(a, None) for a in aa_change]

    for src, dst in _AGG_ANNOTATIONS.items():
        if src in df.columns:
            out[dst] = df[src].values
    return out


def _parse_all_epitopes(df):
    """All-epitopes-format DataFrame → parsed-rows DataFrame."""
    out = pd.DataFrame(index=df.index)
    out["peptide"] = df["MT Epitope Seq"]
    out["allele"] = _normalize_alleles(df["HLA Allele"])

    mt_ic50 = df.get("Median MT IC50 Score", df.get("Best MT IC50 Score"))
    wt_ic50 = df.get("Median WT IC50 Score", df.get("Corresponding WT IC50 Score"))
    mt_pct  = df.get("Median MT Percentile", df.get("Best MT Percentile"))
    wt_pct  = df.get("Median WT Percentile", df.get("Corresponding WT Percentile"))

    out["value"] = pd.to_numeric(mt_ic50, errors="coerce")
    out["percentile_rank"] = pd.to_numeric(mt_pct, errors="coerce")
    out["wt_value"] = pd.to_numeric(wt_ic50, errors="coerce") if wt_ic50 is not None else np.nan
    out["wt_percentile_rank"] = (
        pd.to_numeric(wt_pct, errors="coerce") if wt_pct is not None else np.nan
    )
    out["wt_peptide"] = (
        df["WT Epitope Seq"].values if "WT Epitope Seq" in df.columns else None
    )
    variant_type = (
        df["Variant Type"] if "Variant Type" in df.columns else [None] * len(df)
    )
    out["effect_type"] = [_classify_effect(None, v) for v in variant_type]

    # Stable variant id from chr-coords if "Index" isn't usable.
    if "Index" in df.columns:
        out["variant"] = df["Index"]
    elif {"Chromosome", "Start", "Reference", "Variant"} <= set(df.columns):
        out["variant"] = (
            df["Chromosome"].astype(str) + "-"
            + df["Start"].astype(str) + "-"
            + df["Reference"].astype(str) + "-"
            + df["Variant"].astype(str)
        )

    for src, dst in _AE_ANNOTATIONS.items():
        if src in df.columns:
            out[dst] = df[src].values

    # Per-algorithm score columns pass through as snake_case names.
    for col in df.columns:
        m = _PER_ALGO_RE.match(col)
        if m is None:
            continue
        algo = m.group("algo")
        if algo in ("Median", "Best", "Corresponding"):
            continue
        snake = algo.lower().replace(".", "_").replace("-", "_")
        new_col = (
            f"pvacseq_{snake}"
            f"_{_FIELD_SHORT[m.group('field')]}"
            f"_{m.group('mtwt').lower()}"
        )
        out[new_col] = pd.to_numeric(df[col], errors="coerce")

    return out


# =============================================================================
# Shared finalizer
# =============================================================================

# Final column order — present columns appear in this order; absent columns
# are silently dropped.  Annotation pass-throughs (incl. pvacseq_*) sort
# alphabetically after the canonical columns.
_CANONICAL_ORDER = [
    "source_sequence_name", "gene", "transcript", "variant", "effect_type",
    "peptide", "peptide_offset", "peptide_length",
    "allele", "kind", "score", "value", "affinity", "percentile_rank",
    "prediction_method_name", "predictor_version",
    "wt_peptide", "wt_peptide_length",
    "wt_value", "wt_affinity", "wt_score", "wt_percentile_rank",
    "wt_prediction_method_name", "wt_predictor_version",
]


def _finalize(parsed):
    """Add synthesized constants, mirrors, and column order to a parsed frame."""
    parsed["peptide_length"] = parsed["peptide"].str.len()
    parsed["peptide_offset"] = 0
    parsed["kind"] = "pMHC_affinity"
    parsed["prediction_method_name"] = "pvacseq"
    parsed["predictor_version"] = pd.NA
    parsed["affinity"] = parsed["value"]
    parsed["score"] = parsed["value"]

    parsed["wt_peptide_length"] = parsed["wt_peptide"].map(
        lambda s: len(s) if isinstance(s, str) else pd.NA
    )
    parsed["wt_affinity"] = parsed["wt_value"]
    parsed["wt_score"] = parsed["wt_value"]
    parsed["wt_prediction_method_name"] = "pvacseq"
    parsed["wt_predictor_version"] = pd.NA

    # source_sequence_name: gene + variant when both available, else
    # whichever is present.  Both flavors guarantee at least one, so the
    # column is always populated.
    if "gene" in parsed.columns and "variant" in parsed.columns:
        parsed["source_sequence_name"] = (
            parsed["gene"].astype(object).fillna("?") + ":"
            + parsed["variant"].astype(object).fillna("?")
        )
    elif "variant" in parsed.columns:
        parsed["source_sequence_name"] = parsed["variant"]
    elif "gene" in parsed.columns:
        parsed["source_sequence_name"] = parsed["gene"]

    canonical = [c for c in _CANONICAL_ORDER if c in parsed.columns]
    extra = sorted(c for c in parsed.columns if c not in canonical)
    return parsed[canonical + extra].copy()


# =============================================================================
# Public entry point
# =============================================================================


def read_pvacseq(path, *, tag=None) -> TopiaryResult:
    """Read a pVACseq aggregated or all_epitopes TSV.

    Parameters
    ----------
    path : str or Path
        pVACseq TSV file.  Both ``*.all_epitopes.tsv`` and
        ``*.all_epitopes.aggregated.tsv`` flavors are auto-detected from
        column headers; MHC class (I vs II) is implicit in the allele
        column.
    tag : str, optional
        Source label for :class:`Metadata.sources`.  Defaults to a
        ``pvacseq-{format}:{filename}`` string.

    Returns
    -------
    TopiaryResult
        Long-form DataFrame with one row per (peptide, allele) and
        ``Metadata.extra["pvacseq_format"]`` recording the file flavor.
        Compose multiple files with :func:`topiary.concat`.
    """
    path = Path(path)
    # "X" is pVACseq's sentinel for "this algorithm didn't score this
    # peptide × allele"; pandas already treats "NA" as NaN by default.
    df = pd.read_csv(path, sep="\t", na_values=["X"])
    fmt = detect_pvacseq_format(df.columns)
    if fmt is None:
        raise ValueError(
            f"Could not detect pVACseq format in {path.name}: expected an "
            "aggregated TSV (e.g. *.all_epitopes.aggregated.tsv) or an "
            f"all_epitopes TSV.  Got columns: {sorted(df.columns)[:8]}..."
        )

    parsed = (_parse_aggregated if fmt == "aggregated" else _parse_all_epitopes)(df)
    out = _finalize(parsed)

    meta = Metadata(
        form="long",
        sources=[tag or f"pvacseq-{fmt}:{path.name}"],
    )
    meta.extra["pvacseq_format"] = fmt
    return TopiaryResult(out, meta)
