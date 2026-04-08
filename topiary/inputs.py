"""
Parse peptide and protein sequence inputs from CSV and FASTA files.

Supported formats:

- **Peptide CSV**: ``peptide`` column required. Optional: ``name``,
  ``n_flank``, ``c_flank``, ``source_protein``.
- **pMHC CSV**: ``allele`` and ``peptide`` columns required.
  Optional: ``name``, ``n_flank``, ``c_flank``, ``source_protein``.
- **Sequence CSV**: ``sequence`` column required.
  Optional: ``name``, ``source_protein``.
- **FASTA**: standard FASTA with ``>name`` headers and amino acid sequences.
"""

import logging

import pandas as pd


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


def read_fasta(path):
    """Read a FASTA file of protein sequences.

    Returns
    -------
    dict : name -> amino acid sequence
    """
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
    if not sequences:
        logging.warning("No sequences found in FASTA file: %s", path)
    return sequences


def _require_columns(df, required, path):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s) %s in %s. Found: %s"
            % (missing, path, list(df.columns))
        )
