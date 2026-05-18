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
``wt.Affinity.value`` work without further setup.

Derived columns aligned with :class:`TopiaryPredictor` output so vaxrank
and other downstream consumers don't have to special-case loader source:

- ``mhc_class`` (``"I"`` / ``"II"``) — derived from the allele string.
- ``contains_mutant_residues`` (boolean) — true iff pVACseq's reported
  mutation position falls inside the candidate peptide.
- ``mutation_start_in_peptide`` / ``mutation_end_in_peptide`` (Int64,
  0-based half-open) — derived from pVACseq's 1-based Pos / Mutation
  Position.  Single-residue semantics; indels / frameshifts collapse to
  a representative position.
- ``source`` — per-row provenance label (tag or ``pvacseq-{flavor}:{filename}``),
  matching :func:`read_tsv` convention so multi-file stacks stay
  distinguishable.

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

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .io import Metadata
from .result import TopiaryResult


# =============================================================================
# Format detection
# =============================================================================

_AGG_SIGNATURE = frozenset({"Best Peptide", "IC50 MT", "%ile MT", "Allele"})
_ALL_SIGNATURE = frozenset({"MT Epitope Seq", "HLA Allele"})
_ALL_MEDIAN = frozenset({"Median MT IC50 Score", "Median MT Percentile"})
_ALL_BEST = frozenset({"Best MT IC50 Score", "Best MT Percentile"})


def detect_pvacseq_format(columns) -> str | None:
    """Identify which pVACseq output flavor a column header set represents.

    Used by :func:`read_pvacseq` to dispatch to the right parser, and
    exposed publicly so callers writing their own readers (or routing
    code that needs to choose between flavor-specific handlers) can
    classify a TSV without loading it.

    Parameters
    ----------
    columns : iterable of str
        Column headers from a pVACseq TSV (e.g. the ``columns``
        attribute of a ``pd.read_csv(..., nrows=0)`` DataFrame).

    Returns
    -------
    ``"aggregated"`` for ``*.all_epitopes.aggregated.tsv``;
    ``"all_epitopes"`` for the unaggregated flavor; ``None`` if the
    header doesn't match either signature.
    """
    cols = set(columns)
    if _AGG_SIGNATURE <= cols:
        return "aggregated"
    if _ALL_SIGNATURE <= cols and (_ALL_MEDIAN <= cols or _ALL_BEST <= cols):
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
        # pVACseq frameshifts are "FS<pos>" or "FS<start>-<end>" (e.g. "FS342").
        if re.match(r"^FS\d", s, re.IGNORECASE):
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
# "Pos" is pVACseq's position-of-mutation-within-the-Best-Peptide (1-based)
# in aggregated TSVs; it's the analog of all_epitopes' "Mutation Position".
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
    "Pos":                     "mutation_position",
    "Ref Match":               "pvacseq_ref_match",
    "Evaluation":              "pvacseq_evaluation",
    "Prob Pos":                "pvacseq_prob_pos",
}

# (pVACseq column → topiary column) for all_epitopes-flavor annotations.
_ALL_ANNOTATIONS = {
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


def _first_present_column(df, *candidates):
    """Return the first DataFrame column whose name is in *candidates*, or None."""
    for name in candidates:
        if name in df.columns:
            return df[name]
    return None


def _class_of_allele(allele):
    """Return ``"I"`` / ``"II"`` for an allele string, else ``pd.NA``."""
    if not isinstance(allele, str):
        return pd.NA
    a = allele.upper()
    if a.startswith(("HLA-A", "HLA-B", "HLA-C")):
        return "I"
    if a.startswith("HLA-D") or a.startswith(("DRB", "DPA", "DPB", "DQA", "DQB")):
        return "II"
    return pd.NA


def derive_mhc_class(allele_series: pd.Series) -> pd.Series:
    """Map an allele Series to its MHC class (``"I"`` / ``"II"`` / NA).

    Useful for stamping the ``mhc_class`` column on a DataFrame that
    doesn't already have it (e.g. a fresh ``TopiaryPredictor`` result),
    so the :data:`topiary.class_i` / :data:`topiary.class_ii` filters
    work without rooting through ``Metadata.kind_support``.

    Parameters
    ----------
    allele_series : pandas.Series
        Allele strings (mhcgnomes-normalized or raw).

    Returns
    -------
    pandas.Series
        Same index as the input.  Class I (``HLA-A/B/C``) → ``"I"``;
        any HLA-D* locus → ``"II"``; anything else → ``pd.NA``.
    """
    return allele_series.map(_class_of_allele)


def _summarize_mhc_class(allele_series):
    """File-level MHC class summary: ``"I"`` / ``"II"`` / ``"both"`` / ``None``."""
    classes = {c for c in (_class_of_allele(a) for a in allele_series) if c is not pd.NA}
    if classes == {"I"}:
        return "I"
    if classes == {"II"}:
        return "II"
    if classes == {"I", "II"}:
        return "both"
    return None


def _build_kind_support(mhc_class):
    """Synthesize a kind_support dict for pVACseq's single-allele scoring.

    pVACseq always operates per allele, so ``mhc_dependence`` is
    ``"single_allele"``.  ``mhc_class`` reflects what's actually in the
    file ("I", "II", "both", or "none" when alleles are unrecognized).
    Shape matches ``TopiaryPredictor.kind_support``: dict of
    ``model_key -> {kind -> {mhc_dependence, mhc_class}}``.
    """
    return {
        "pvacseq": {
            "pMHC_affinity": {
                "mhc_dependence": "single_allele",
                "mhc_class": mhc_class or "none",
            },
        },
    }


def _derive_mutation_interval(parsed):
    """Return contains_mutant_residues + 0-based half-open mutation interval
    derived from the 1-based ``mutation_position`` column.

    pVACseq's Pos / Mutation Position is the single-residue position of the
    mutation within the candidate peptide.  Rows where the position is
    missing or falls outside the peptide (flanking-only peptides) get
    ``contains_mutant_residues = False`` and NaN start/end.

    Multi-residue mutations (indels, frameshifts) collapse to a single
    representative position; downstream code wanting full intervals
    should re-derive from the source protein.
    """
    n = len(parsed)
    if "mutation_position" not in parsed.columns:
        return {
            "contains_mutant_residues": pd.array([pd.NA] * n, dtype="boolean"),
            "mutation_start_in_peptide": pd.array([pd.NA] * n, dtype="Int64"),
            "mutation_end_in_peptide": pd.array([pd.NA] * n, dtype="Int64"),
        }
    pos = pd.to_numeric(parsed["mutation_position"], errors="coerce")
    pep_len = parsed["peptide"].str.len()
    valid = pos.notna() & (pos >= 1) & (pos <= pep_len)
    start_int = (pos - 1).where(valid).astype("Int64")
    end_int = pos.where(valid).astype("Int64")
    return {
        "contains_mutant_residues": valid.astype("boolean"),
        "mutation_start_in_peptide": start_int,
        "mutation_end_in_peptide": end_int,
    }


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

    mt_ic50 = _first_present_column(df, "Median MT IC50 Score", "Best MT IC50 Score")
    wt_ic50 = _first_present_column(df, "Median WT IC50 Score", "Corresponding WT IC50 Score")
    mt_pct  = _first_present_column(df, "Median MT Percentile", "Best MT Percentile")
    wt_pct  = _first_present_column(df, "Median WT Percentile", "Corresponding WT Percentile")

    out["value"] = pd.to_numeric(mt_ic50, errors="coerce") if mt_ic50 is not None else np.nan
    out["percentile_rank"] = pd.to_numeric(mt_pct, errors="coerce") if mt_pct is not None else np.nan
    out["wt_value"] = pd.to_numeric(wt_ic50, errors="coerce") if wt_ic50 is not None else np.nan
    out["wt_percentile_rank"] = pd.to_numeric(wt_pct, errors="coerce") if wt_pct is not None else np.nan

    if "WT Epitope Seq" in df.columns:
        out["wt_peptide"] = df["WT Epitope Seq"].values
    else:
        out["wt_peptide"] = pd.NA

    if "Variant Type" in df.columns:
        out["effect_type"] = [_classify_effect(None, v) for v in df["Variant Type"]]
    else:
        out["effect_type"] = pd.NA

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

    for src, dst in _ALL_ANNOTATIONS.items():
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
    "source", "source_sequence_name",
    "gene", "transcript", "variant", "effect_type", "variant_type",
    "peptide", "peptide_offset", "peptide_length",
    "contains_mutant_residues",
    "mutation_start_in_peptide", "mutation_end_in_peptide",
    "allele", "mhc_class",
    "kind", "score", "value", "affinity", "percentile_rank",
    "prediction_method_name", "predictor_version",
    "wt_peptide", "wt_peptide_length",
    "wt_value", "wt_affinity", "wt_score", "wt_percentile_rank",
    "wt_prediction_method_name", "wt_predictor_version",
]


def _build_source_sequence_name(parsed):
    """Compose ``gene:variant`` when both are available; otherwise return
    whichever is present.  Both flavors guarantee at least one."""
    has_gene = "gene" in parsed.columns
    has_variant = "variant" in parsed.columns
    if has_gene and has_variant:
        return (
            parsed["gene"].astype(object).fillna("?") + ":"
            + parsed["variant"].astype(object).fillna("?")
        )
    if has_variant:
        return parsed["variant"]
    if has_gene:
        return parsed["gene"]
    raise KeyError(
        "source_sequence_name needs at least one of 'gene' or 'variant'; "
        "neither column was populated by the flavor parser."
    )


def _finalize(parsed, *, source):
    """Add synthesized constants, mirrors, and column order in one allocation.

    *source* is stamped on every row to match topiary's read_tsv provenance
    convention; downstream stacking across MHC-I and MHC-II files stays
    distinguishable without rooting through Metadata.
    """
    # peptide_offset = 0: pVACseq doesn't ship the source-protein offset
    # of the peptide; LENS loader uses the same convention.
    mutation_cols = _derive_mutation_interval(parsed)
    augmented = parsed.assign(
        source=source,
        peptide_length=parsed["peptide"].str.len(),
        peptide_offset=0,
        kind="pMHC_affinity",
        mhc_class=derive_mhc_class(parsed["allele"]),
        prediction_method_name="pvacseq",
        predictor_version=pd.NA,
        affinity=parsed["value"],
        score=parsed["value"],
        wt_peptide_length=parsed["wt_peptide"]
            .map(lambda s: len(s) if isinstance(s, str) else pd.NA)
            .astype("Int64"),
        wt_affinity=parsed["wt_value"],
        wt_score=parsed["wt_value"],
        wt_prediction_method_name="pvacseq",
        wt_predictor_version=pd.NA,
        source_sequence_name=_build_source_sequence_name(parsed),
        **mutation_cols,
    )

    canonical = [c for c in _CANONICAL_ORDER if c in augmented.columns]
    extra = sorted(c for c in augmented.columns if c not in canonical)
    # DataFrame.__getitem__ with a column list returns a fresh frame —
    # no extra .copy() needed.
    return augmented[canonical + extra]


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
        Compose multiple files with :func:`topiary.stack_results`.
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
    source_label = tag or f"pvacseq-{fmt}:{path.name}"
    out = _finalize(parsed, source=source_label)

    mhc_class = _summarize_mhc_class(out["allele"])
    meta = Metadata(form="long", sources=[source_label])
    meta.extra["pvacseq_format"] = fmt
    # kind_support has the same shape as TopiaryPredictor.kind_support so
    # downstream callers (e.g. vaxrank's evaluate_scores) can pass
    # `r.extra["kind_support"]` through without branching on loader source.
    meta.extra["kind_support"] = _build_kind_support(mhc_class)
    return TopiaryResult(out, meta)


# =============================================================================
# Per-algorithm melt helper
# =============================================================================

_ALGO_COL_RE = re.compile(
    r"^pvacseq_(?P<algo>[\w]+)_(?P<field>ic50|pct)_(?P<mtwt>mt|wt)$"
)


def melt_pvacseq_algorithms(result: TopiaryResult) -> TopiaryResult:
    """Expand per-algorithm columns into one row per (peptide, allele, algorithm).

    The all_epitopes flavor carries individual-algorithm scores as
    ``pvacseq_<algo>_{ic50,pct}_{mt,wt}`` columns (see :func:`read_pvacseq`).
    By default the DSL only sees the Median row
    (``prediction_method_name="pvacseq"``); after melting, each algorithm
    becomes a separate row with ``prediction_method_name=<algo>``, so
    expressions like ``Affinity['mhcflurry'].value`` reach it natively.

    The original Median rows are preserved.  Algorithms that don't have
    any non-null score for a given (peptide, allele) still get a row
    (with NaN value / percentile_rank); the caller can filter those out
    if undesired.

    Parameters
    ----------
    result : TopiaryResult
        A long-form result from :func:`read_pvacseq` (typically the
        all_epitopes flavor; aggregated TSVs only carry Median scores
        and round-trip unchanged).

    Returns
    -------
    TopiaryResult
        Same metadata, longer DataFrame.  ``Metadata.extra["kind_support"]``
        is extended to register each melted algorithm under the same
        ``mhc_class`` as ``"pvacseq"``.
    """
    df = result.df
    algos = sorted({
        m.group("algo")
        for m in (_ALGO_COL_RE.match(c) for c in df.columns)
        if m is not None
    })
    if not algos:
        return result

    rows_per_algo = []
    for algo in algos:
        row = df.copy()
        row["prediction_method_name"] = algo
        mt_ic50 = f"pvacseq_{algo}_ic50_mt"
        mt_pct = f"pvacseq_{algo}_pct_mt"
        wt_ic50 = f"pvacseq_{algo}_ic50_wt"
        wt_pct = f"pvacseq_{algo}_pct_wt"
        if mt_ic50 in df.columns:
            row["value"] = df[mt_ic50]
            row["affinity"] = df[mt_ic50]
            row["score"] = df[mt_ic50]
        if mt_pct in df.columns:
            row["percentile_rank"] = df[mt_pct]
        if wt_ic50 in df.columns:
            row["wt_value"] = df[wt_ic50]
            row["wt_affinity"] = df[wt_ic50]
            row["wt_score"] = df[wt_ic50]
        if wt_pct in df.columns:
            row["wt_percentile_rank"] = df[wt_pct]
        row["wt_prediction_method_name"] = algo
        rows_per_algo.append(row)

    combined = pd.concat([df] + rows_per_algo, ignore_index=True)

    # Preserve & extend kind_support so DSL evaluations of
    # `Affinity['<algo>'].value` find consistent metadata for each.
    new_kind_support = dict(result.extra.get("kind_support", {}))
    template = new_kind_support.get("pvacseq", {}).get("pMHC_affinity")
    if template is not None:
        for algo in algos:
            new_kind_support.setdefault(algo, {"pMHC_affinity": dict(template)})

    new_extra = dict(result.extra)
    new_extra["kind_support"] = new_kind_support
    new_extra["pvacseq_algorithms_melted"] = ",".join(algos)

    return TopiaryResult(
        combined,
        topiary_version=result.topiary_version,
        form=result.form,
        models=result.models,
        sources=result.sources,
        extra=new_extra,
    )
