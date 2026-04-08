"""
Parse peptide and protein sequence inputs from CSV and FASTA files.

Supports regions of interest, exclusion filtering against reference
proteomes, and two FASTA modes (protein scanning vs. peptide list).

Example with a viral proteome::

    from topiary import TopiaryPredictor, Affinity, Presentation
    from topiary.inputs import read_fasta, slice_regions, build_exclusion_set
    from mhctools import NetMHCpan

    # Load SARS-CoV-2 proteome
    proteins = read_fasta("sars_cov2.fasta")

    # Focus on regions of interest (half-open intervals)
    regions = slice_regions(proteins, {
        "spike": [(319, 541)],           # RBD
        "nucleocapsid": [(0, 50), (350, 419)],
    })

    # Build exclusion set — combine multiple sources
    excluded = (
        build_exclusion_set(read_fasta("non_cta.fasta"), lengths=[8, 9, 10, 11])
        | build_exclusion_set(read_fasta("germline.fasta"), lengths=[8, 9, 10, 11])
    )

    # Predict
    predictor = TopiaryPredictor(
        models=NetMHCpan(alleles=["A0201", "A0301", "B0702"]),
        ranking=(Affinity <= 500) | (Presentation.rank <= 2.0),
    )
    df = predictor.predict_from_named_sequences(regions)

    # Remove peptides containing any excluded substring
    df = df[~peptides_contained_in(df, excluded)]
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


def build_exclusion_set(sequences, lengths=None, min_length=8):
    """Build a set of all k-mer peptides from reference sequences.

    The exclusion source can be anything you consider "background" —
    non-CTA proteins, vital-organ proteomes, a patient's germline, etc.
    Combine multiple sources by taking the union::

        excluded = (
            build_exclusion_set(read_fasta("vital_organs.fasta"))
            | build_exclusion_set(read_fasta("germline.fasta"))
        )

    When used with :func:`peptide_is_excluded`, even shorter k-mers from
    the exclusion set that appear as *substrings* of a longer predicted
    peptide will cause exclusion.  For example, an 8-mer from heart
    tissue contained within a 9-mer from a CTA gene will exclude that
    9-mer.

    Parameters
    ----------
    sequences : dict
        name -> amino acid sequence (e.g. from :func:`read_fasta`)

    lengths : list of int, optional
        Peptide lengths to enumerate. If None, uses ``[min_length]``.

    min_length : int
        Shortest k-mer to enumerate (default 8). Shorter k-mers enable
        substring containment checking against longer peptides.

    Returns
    -------
    set of str
    """
    if lengths is None:
        lengths = [min_length]
    peptides = set()
    for seq in sequences.values():
        for length in lengths:
            for i in range(len(seq) - length + 1):
                peptides.add(seq[i:i + length])
    return peptides


def peptides_contained_in(df, exclusion_set, peptide_column="peptide",
                          substring=True):
    """Boolean mask: True for peptides matching the exclusion set.

    Parameters
    ----------
    df : pandas.DataFrame
    exclusion_set : set of str
    peptide_column : str
    substring : bool
        If True (default), a peptide is excluded when *any* k-mer from
        the exclusion set appears as a contiguous substring.  An 8-mer
        from heart tissue inside a 9-mer CTA peptide → excluded.

        If False, only exact whole-peptide matches are excluded
        (equivalent to ``df[peptide_column].isin(exclusion_set)``).

    Returns
    -------
    pandas.Series of bool
        True = peptide is excluded.
    """
    if not exclusion_set:
        return pd.Series(False, index=df.index)

    if not substring:
        return df[peptide_column].isin(exclusion_set)

    excl_lengths = {len(s) for s in exclusion_set}

    def _is_excluded(peptide):
        for k in excl_lengths:
            if k > len(peptide):
                continue
            for i in range(len(peptide) - k + 1):
                if peptide[i:i + k] in exclusion_set:
                    return True
        return False

    return df[peptide_column].apply(_is_excluded)




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
