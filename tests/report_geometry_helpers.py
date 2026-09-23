"""Reproducible source-report fixtures for imported mutation geometry.

Build both pVACseq flavors from the same peptide/position facts. Keeping this
builder beside the tests makes the coordinate convention and WT derivation
reviewable without external analysis directories or generated binary inputs.
"""

import pandas as pd


def write_pvacseq_geometry_report(path, flavor, position=2):
    """Write paired 12-aa sequences and their reported changed positions.

    The default is C→R at residue two. Multi-position cases change the
    corresponding residues as well; missing/unsupported positions deliberately
    omit reliable localization while retaining the observed sequence.
    """
    mutant, reference = "ARDEFGHIKLMN", "ACDEFGHIKLMN"
    aa_change = "C42R"
    if position == "2,11":
        mutant, aa_change = "ARDEFGHIKLAN", "CM42-51RA"
    elif position == "2-3":
        mutant, aa_change = "ARAEFGHIKLMN", "CD42-43RA"
    elif position == 99:
        mutant = reference  # Historical flank-only report; mutation outside the window.
    if flavor == "aggregated":
        row = {
            "Best Peptide": mutant, "Allele": "HLA-A*02:01",
            "IC50 MT": 100, "%ile MT": 1, "Gene": "GENE",
            "Best Transcript": "ENST1", "ID": "chr1:1C>G",
            "Pos": position, "AA Change": aa_change,
        }
    else:
        row = {
            "MT Epitope Seq": mutant, "WT Epitope Seq": reference,
            "HLA Allele": "HLA-A*02:01", "Median MT IC50 Score": 100,
            "Median MT Percentile": 1, "Gene Name": "GENE",
            "Transcript": "ENST1", "Index": "chr1:1C>G",
            "Mutation Position": position, "Mutation": aa_change,
            "Variant Type": "missense",
        }
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False)
    return path
