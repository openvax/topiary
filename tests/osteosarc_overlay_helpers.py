"""Pinned RNA data and the study's production annotation handoff."""

import json
from pathlib import Path

from .sid_data import sid_data_root

import pandas as pd

from scripts.osteosarc_rna_overlay import annotation_table
from topiary import join_annotations


ROOT = sid_data_root("osteosarc_rna_overlay")
PROVENANCE = {
    "sample": "January 2025 UCLA resection tumor RNA (T2)",
    "expression_unit": "TPM", "count_unit": "sequenced segments",
    "acquisition": json.loads((ROOT / "acquisition.json").read_text()),
}


def add_rna(result, raw):
    evidence = pd.read_csv(ROOT / "transcript-evidence.tsv", sep="\t")
    return join_annotations(result, annotation_table(raw, evidence),
                            on=["variant", "transcript"], prefix="rna_t2", provenance=PROVENANCE)
