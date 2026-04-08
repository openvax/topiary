"""
Parse peptide and protein sequence inputs from CSV and FASTA files.

Supports regions of interest, exclusion filtering against reference
proteomes, and two FASTA modes (protein scanning vs. peptide list).

Example with a viral proteome::

    from topiary import TopiaryPredictor, Affinity, Presentation
    from topiary.inputs import (
        read_fasta, slice_regions, build_exclusion_set,
        exclude_self_peptides,
    )
    from mhctools import NetMHCpan

    # Load SARS-CoV-2 proteome
    proteins = read_fasta("sars_cov2.fasta")

    # Focus on regions of interest (half-open intervals)
    regions = slice_regions(proteins, {
        "spike": [(319, 541)],           # RBD
        "nucleocapsid": [(0, 50), (350, 419)],
    })

    # Build exclusion set from human proteome
    human = read_fasta("human_proteome.fasta")
    self_peptides = build_exclusion_set(human, lengths=[8, 9, 10, 11])

    # Predict
    predictor = TopiaryPredictor(
        models=NetMHCpan(alleles=["A0201", "A0301", "B0702"]),
        ranking=(Affinity <= 500) | (Presentation.rank <= 2.0),
    )
    df = predictor.predict_from_named_sequences(regions)

    # Remove self-peptides
    df = exclude_self_peptides(df, self_peptides)
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


def build_exclusion_set(sequences, lengths):
    """Build a set of all k-mer peptides from reference sequences.

    Use this to construct a self-peptide set from e.g. the human proteome,
    then pass to :func:`exclude_self_peptides` to remove them from
    predictions.

    Parameters
    ----------
    sequences : dict
        name -> amino acid sequence (e.g. from :func:`read_fasta`)

    lengths : list of int
        Peptide lengths to enumerate (e.g. [8, 9, 10, 11])

    Returns
    -------
    set of str
    """
    peptides = set()
    for seq in sequences.values():
        for length in lengths:
            for i in range(len(seq) - length + 1):
                peptides.add(seq[i:i + length])
    return peptides


def exclude_self_peptides(df, exclusion_set, peptide_column="peptide"):
    """Remove predictions whose peptide appears in the exclusion set.

    Parameters
    ----------
    df : pandas.DataFrame
        Predictions DataFrame with a peptide column.

    exclusion_set : set of str
        Peptides to exclude (e.g. from :func:`build_exclusion_set`).

    peptide_column : str
        Column name containing peptide sequences.

    Returns
    -------
    pandas.DataFrame
        Filtered copy with self-peptides removed.
    """
    mask = ~df[peptide_column].isin(exclusion_set)
    n_removed = (~mask).sum()
    if n_removed > 0:
        logging.info(
            "Excluded %d predictions matching %d-peptide exclusion set",
            n_removed, len(exclusion_set),
        )
    return df[mask].reset_index(drop=True)


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
