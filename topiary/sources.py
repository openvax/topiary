"""
Built-in protein sequence sources from Ensembl and PirlyGenes.

Provides functions to load protein sequences for prediction from:
- Full Ensembl proteome (any species/release)
- Gene lists (by name or ID)
- Transcript lists (by name or ID)
- Cancer-testis antigen (CTA) genes
- Non-CTA genes (for exclusion sets)
- Tissue-expressed genes (via PirlyGenes expression data)

All functions return ``dict[str, str]`` (name -> amino acid sequence),
compatible with :meth:`TopiaryPredictor.predict_from_named_sequences`.

Ensembl data is downloaded on first use via pyensembl.
"""

import logging

from pyensembl import EnsemblRelease, ensembl_grch38

from .optional_dependencies import require_optional_dependency


# ---------------------------------------------------------------------------
# Ensembl proteome
# ---------------------------------------------------------------------------


def ensembl_proteome(release=None, species="human"):
    """All protein sequences from an Ensembl release.

    Parameters
    ----------
    release : int, optional
        Ensembl release number. Defaults to the latest installed GRCh38
        release (93 for human).

    species : str
        Species name (default "human").

    Returns
    -------
    dict : protein_id -> amino acid sequence
    """
    genome = _get_genome(release, species)
    sequences = {}
    for protein_id in genome.protein_ids():
        seq = genome.protein_sequence(protein_id)
        if seq:
            sequences[protein_id] = seq
    logging.info("Loaded %d protein sequences from Ensembl %s", len(sequences), genome)
    return sequences


# ---------------------------------------------------------------------------
# Gene / transcript lookups
# ---------------------------------------------------------------------------


def sequences_from_gene_names(gene_names, release=None, species="human"):
    """Protein sequences for a list of gene names.

    For each gene, uses the longest protein-coding transcript.

    Parameters
    ----------
    gene_names : list of str
        e.g. ["BRAF", "TP53", "EGFR"]

    Returns
    -------
    dict : "GENE_NAME|TRANSCRIPT_ID" -> amino acid sequence
    """
    if not gene_names:
        raise ValueError("gene_names must be a non-empty list")
    genome = _get_genome(release, species)
    return _sequences_for_genes(genome, gene_names, by="name")


def sequences_from_gene_ids(gene_ids, release=None, species="human"):
    """Protein sequences for a list of Ensembl gene IDs.

    Parameters
    ----------
    gene_ids : list of str
        e.g. ["ENSG00000157764", "ENSG00000141510"]
    """
    if not gene_ids:
        raise ValueError("gene_ids must be a non-empty list")
    genome = _get_genome(release, species)
    return _sequences_for_genes(genome, gene_ids, by="id")


def sequences_from_transcript_ids(transcript_ids, release=None, species="human"):
    """Protein sequences for a list of Ensembl transcript IDs.

    Parameters
    ----------
    transcript_ids : list of str
        e.g. ["ENST00000288602", "ENST00000269305"]
    """
    if not transcript_ids:
        raise ValueError("transcript_ids must be a non-empty list")
    genome = _get_genome(release, species)
    sequences = {}
    for tid in transcript_ids:
        try:
            t = genome.transcript_by_id(tid)
        except ValueError:
            logging.warning("Transcript %s not found in Ensembl %s", tid, genome)
            continue
        seq = t.protein_sequence
        if seq:
            sequences[f"{t.gene.name}|{tid}"] = seq
    return sequences


def sequences_from_transcript_names(transcript_names, release=None, species="human"):
    """Protein sequences for a list of transcript names.

    Parameters
    ----------
    transcript_names : list of str
        e.g. ["BRAF-001", "TP53-001"]
    """
    if not transcript_names:
        raise ValueError("transcript_names must be a non-empty list")
    genome = _get_genome(release, species)
    sequences = {}
    for tname in transcript_names:
        try:
            transcripts = genome.transcripts_by_name(tname)
        except ValueError:
            logging.warning("Transcript %s not found in Ensembl %s", tname, genome)
            continue
        for t in transcripts:
            seq = t.protein_sequence
            if seq:
                sequences[f"{t.gene.name}|{t.id}"] = seq
    return sequences


# ---------------------------------------------------------------------------
# CTA / non-CTA (human only, via PirlyGenes)
# ---------------------------------------------------------------------------


def cta_sequences(release=None):
    """Protein sequences from cancer-testis antigen (CTA) genes.

    Human only. Uses CTA gene list from PirlyGenes.

    Returns
    -------
    dict : "GENE_NAME|TRANSCRIPT_ID" -> amino acid sequence
    """
    gene_ids = _pirlygenes_cta_gene_ids()
    genome = _get_genome(release, "human")
    return _sequences_for_genes(genome, list(gene_ids), by="id")



def non_cta_sequences(release=None):
    """Protein sequences from all non-CTA genes.

    Human only. Useful as an exclusion source — peptides from non-CTA
    proteins are likely presented on normal tissues.

    Returns
    -------
    dict : protein_id -> amino acid sequence
    """
    cta_ids = _pirlygenes_cta_gene_ids()
    genome = _get_genome(release, "human")
    sequences = {}
    for protein_id in genome.protein_ids():
        try:
            gene_id = genome.gene_id_of_protein_id(protein_id)
        except ValueError:
            continue
        if gene_id not in cta_ids:
            seq = genome.protein_sequence(protein_id)
            if seq:
                sequences[protein_id] = seq
    logging.info(
        "Loaded %d non-CTA protein sequences (excluded %d CTA genes)",
        len(sequences), len(cta_ids),
    )
    return sequences


# ---------------------------------------------------------------------------
# Tissue-expressed genes (via PirlyGenes expression data)
# ---------------------------------------------------------------------------


def tissue_expressed_gene_ids(tissues, min_ntpm=1.0):
    """Gene IDs expressed in specified tissues above a threshold.

    Parameters
    ----------
    tissues : list of str
        Tissue names, e.g. ["heart_muscle", "lung", "brain"].
        Use :func:`available_tissues` to see the full list.

    min_ntpm : float
        Minimum normalized TPM to consider a gene "expressed".

    Returns
    -------
    set of str : Ensembl gene IDs
    """
    if not tissues:
        raise ValueError("tissues must be a non-empty list")
    pirlygenes = _check_pirlygenes("pan_cancer_expression")
    pce = pirlygenes.pan_cancer_expression()

    cols = _tissue_columns(pce, tissues)

    mask = pce[cols].max(axis=1) >= min_ntpm
    return set(pce.loc[mask, "Ensembl_Gene_ID"])


def tissue_expressed_sequences(tissues, min_ntpm=1.0, release=None):
    """Protein sequences of genes expressed in specified tissues.

    Parameters
    ----------
    tissues : list of str
    min_ntpm : float
    release : int, optional

    Returns
    -------
    dict : "GENE_NAME|TRANSCRIPT_ID" -> amino acid sequence
    """
    gene_ids = tissue_expressed_gene_ids(tissues, min_ntpm=min_ntpm)
    genome = _get_genome(release, "human")
    return _sequences_for_genes(genome, list(gene_ids), by="id")


def available_tissues():
    """List all available tissue names from PirlyGenes expression data."""
    pirlygenes = _check_pirlygenes("pan_cancer_expression")
    return sorted(_tissue_column_map(pirlygenes.pan_cancer_expression()))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _get_genome(release, species):
    if release is None and species == "human":
        return ensembl_grch38
    if release is None:
        raise ValueError("Must specify release for non-human species")
    return EnsemblRelease(release=release, species=species)


def _sequences_for_genes(genome, identifiers, by="name"):
    """Get protein sequences for genes, picking the longest transcript."""
    sequences = {}
    for ident in identifiers:
        try:
            if by == "name":
                genes = genome.genes_by_name(ident)
            else:
                genes = [genome.gene_by_id(ident)]
        except ValueError:
            logging.warning("Gene %s not found in Ensembl %s", ident, genome)
            continue
        for gene in genes:
            best_transcript = None
            best_len = 0
            for t in gene.transcripts:
                seq = t.protein_sequence
                if seq and len(seq) > best_len:
                    best_transcript = t
                    best_len = len(seq)
            if best_transcript:
                key = f"{gene.name}|{best_transcript.id}"
                sequences[key] = best_transcript.protein_sequence
    return sequences


# PirlyGenes has named its per-tissue normalized-TPM columns both ways:
# the prefix form (nTPM_lung) through 5.1, and the suffix form
# (lung_nTPM) in current releases, matching the suffix convention its
# own FPKM/TPM helpers accept.  Read tissue names under either spelling
# rather than pinning topiary to one of them.
_NTPM_PREFIX = "nTPM_"
_NTPM_SUFFIX = "_nTPM"


def _tissue_column_map(pce):
    """Map tissue name -> expression column, under either naming form."""
    columns = {}
    for col in pce.columns:
        if not isinstance(col, str):
            continue
        if col.endswith(_NTPM_SUFFIX):
            columns.setdefault(col[:-len(_NTPM_SUFFIX)], col)
        elif col.startswith(_NTPM_PREFIX):
            columns.setdefault(col[len(_NTPM_PREFIX):], col)
    return columns


def _tissue_columns(pce, tissues):
    """Resolve tissue names to expression columns, or explain what's there."""
    columns = _tissue_column_map(pce)
    bad = [t for t in tissues if t not in columns]
    if bad:
        raise ValueError(
            "Unknown tissue(s): %s. Available: %s"
            % (bad, sorted(columns))
        )
    return [columns[t] for t in tissues]


def _check_pirlygenes(*required_callables):
    """Load the PirlyGenes API used for CTA and tissue-expression data."""
    return require_optional_dependency(
        "pirlygenes",
        feature="cancer-testis antigen and tissue-expression gene lists",
        required_callables=required_callables,
    )


def _pirlygenes_cta_gene_ids():
    gene_sets = require_optional_dependency(
        "pirlygenes.gene_sets_cancer",
        feature="cancer-testis antigen gene lists",
        extra="pirlygenes",
        required_callables=("CTA_gene_ids",),
    )
    return set(gene_sets.CTA_gene_ids())
