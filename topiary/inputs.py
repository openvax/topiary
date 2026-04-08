"""
Parse peptide and protein sequence inputs from CSV and FASTA files.

Supports regions of interest, exclusion filtering, and two FASTA modes
(protein scanning vs. peptide list).

Example — CTA-like targets with vital-organ exclusion::

    from topiary import TopiaryPredictor, Affinity, Presentation
    from topiary.sources import tissue_expressed_sequences
    from topiary.inputs import exclude_by
    from mhctools import NetMHCpan

    # Targets: genes expressed in reproductive tissues
    target_seqs = tissue_expressed_sequences(["testis", "placenta", "ovary"])

    # Vital organ proteome (for exclusion)
    vital_seqs = tissue_expressed_sequences(["heart_muscle", "lung", "liver"])

    # Predict
    predictor = TopiaryPredictor(
        models=NetMHCpan, alleles=["A0201", "A0301"],
        filter=Affinity <= 500,
        rank_by=Presentation.score,
    )
    df = predictor.predict_from_named_sequences(target_seqs)

    # Substring exclusion: heart 8-mer inside CTA 9-mer → excluded
    df = exclude_by(df, vital_seqs, mode="substring")

    # Or exact match only:
    df = exclude_by(df, vital_seqs, mode="exact")
"""

import logging

import pandas as pd


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------


def read_peptide_csv(path):
    """Read a CSV of peptides to predict across all specified alleles.

    Required column: ``peptide``.
    Optional columns: ``name``.

    Returns
    -------
    dict : name -> peptide sequence
    """
    df = pd.read_csv(path)
    _require_columns(df, ["peptide"], path)
    if "name" not in df.columns:
        df["name"] = df["peptide"]
    return dict(zip(df["name"], df["peptide"]))


def read_sequence_csv(path):
    """Read a CSV of protein sequences to scan with sliding window.

    Required column: ``sequence``.
    Optional column: ``name``.

    Returns
    -------
    dict : name -> amino acid sequence
    """
    df = pd.read_csv(path)
    _require_columns(df, ["sequence"], path)
    if "name" not in df.columns:
        df["name"] = [f"seq_{i}" for i in range(len(df))]
    return dict(zip(df["name"], df["sequence"]))


# ---------------------------------------------------------------------------
# FASTA readers
# ---------------------------------------------------------------------------


def read_fasta(path):
    """Read a FASTA file of protein/amino acid sequences.

    Use this for full-length protein sequences that will be scanned
    with a sliding window by the predictor.

    Returns
    -------
    dict : name -> amino acid sequence
    """
    sequences = _parse_fasta(path)
    if not sequences:
        logging.warning("No sequences found in FASTA file: %s", path)
    return sequences


def read_peptide_fasta(path):
    """Read a FASTA file where each entry is a single peptide.

    Unlike :func:`read_fasta`, these are predicted as-is (no sliding
    window scanning). Each entry should be a short peptide sequence.

    Returns
    -------
    dict : name -> peptide sequence
    """
    sequences = _parse_fasta(path)
    if not sequences:
        logging.warning("No peptides found in FASTA file: %s", path)
    return sequences


def _parse_fasta(path):
    """Low-level FASTA parser. Returns dict of name -> sequence."""
    sequences = {}
    current_name = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    sequences[current_name] = "".join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_name is not None:
        sequences[current_name] = "".join(current_seq)
    return sequences


# ---------------------------------------------------------------------------
# Region slicing
# ---------------------------------------------------------------------------


def slice_regions(sequences, regions):
    """Extract subsequences at specified half-open intervals.

    This generalizes both full-protein scanning (no regions = use full
    sequence) and targeted analysis (e.g. mutational hotspots, domains).

    Parameters
    ----------
    sequences : dict
        name -> amino acid sequence

    regions : dict
        name -> list of (start, end) half-open intervals.
        Names not in this dict are included in full.
        Names mapped to an empty list are excluded entirely.

    Returns
    -------
    dict : name -> subsequence (with region suffix for multi-region entries)

    Examples
    --------
    >>> seqs = {"spike": "M" * 1273, "orf1a": "A" * 4405}
    >>> # RBD only for spike, full orf1a
    >>> result = slice_regions(seqs, {"spike": [(319, 541)]})
    >>> "spike:319-541" in result
    True
    >>> "orf1a" in result
    True
    """
    result = {}
    for name, seq in sequences.items():
        if name not in regions:
            # No regions specified → include full sequence
            result[name] = seq
            continue

        intervals = regions[name]
        if not intervals:
            # Explicitly empty → exclude this sequence
            continue

        if len(intervals) == 1:
            start, end = intervals[0]
            result[f"{name}:{start}-{end}"] = seq[start:end]
        else:
            for start, end in intervals:
                result[f"{name}:{start}-{end}"] = seq[start:end]

    return result


# ---------------------------------------------------------------------------
# Exclusion filtering
# ---------------------------------------------------------------------------


def exclude_by(df, reference_sequences, mode="substring", min_kmer=8):
    """Remove predicted peptides found in reference protein sequences.

    Parameters
    ----------
    df : pandas.DataFrame
        Predictions DataFrame (must have a ``peptide`` column).

    reference_sequences : dict
        name -> amino acid sequence.  Any source: vital-organ proteome,
        germline, non-CTA proteins, etc.  Combine multiple sources
        before calling::

            ref = {**vital_organ_seqs, **germline_seqs}

    mode : ``"substring"`` or ``"exact"``
        **substring** (default): exclude a predicted peptide if *any*
        ``min_kmer``-length window from the reference appears as a
        contiguous substring.  Example: an 8-mer from heart tissue
        inside a 9-mer CTA peptide → excluded.

        **exact**: exclude only when the full predicted peptide appears
        as a k-mer in the reference (at matching length).

    min_kmer : int
        k-mer length for the reference index (default 8).  In substring
        mode, shorter k-mers catch more containment.  In exact mode,
        the reference is indexed at the predicted peptide lengths instead.

    Returns
    -------
    pandas.DataFrame
        Filtered copy with excluded peptides removed.
    """
    if df.empty or not reference_sequences:
        return df

    if mode == "exact":
        lengths = sorted(df["peptide"].str.len().unique())
        ref_kmers = _build_kmer_set(reference_sequences, lengths)
        mask = df["peptide"].isin(ref_kmers)
    elif mode == "substring":
        ref_kmers = _build_kmer_set(reference_sequences, [min_kmer])
        mask = df["peptide"].apply(lambda p: _contains_any(p, ref_kmers, min_kmer))
    else:
        raise ValueError(f"mode must be 'substring' or 'exact', got {mode!r}")

    n_removed = mask.sum()
    if n_removed:
        logging.info("Excluded %d/%d predictions (%s mode)", n_removed, len(df), mode)
    return df[~mask].reset_index(drop=True)


def _build_kmer_set(sequences, lengths):
    """Enumerate all k-mers at specified lengths from a dict of sequences."""
    kmers = set()
    for seq in sequences.values():
        for k in lengths:
            for i in range(len(seq) - k + 1):
                kmers.add(seq[i:i + k])
    return kmers


def _contains_any(peptide, kmer_set, k):
    """Check if any k-length window of peptide is in kmer_set."""
    if len(peptide) < k:
        return False
    for i in range(len(peptide) - k + 1):
        if peptide[i:i + k] in kmer_set:
            return True
    return False




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_columns(df, required, path):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s) %s in %s. Found: %s"
            % (missing, path, list(df.columns))
        )
