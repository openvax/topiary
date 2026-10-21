"""Fail CI setup unless both human references used by the suite are usable."""

from pyensembl import ensembl_grch37, ensembl_grch38


def main():
    # Use the same canonical objects as the GRCh37 protein-change and GRCh38
    # sequence-source workflows. Both indexed annotation and proteins must
    # work; a successful install of a different mirror identity is not enough.
    for genome, gene, transcript_id in (
        (ensembl_grch37, "EGFR", "ENST00000275493"),
        (ensembl_grch38, "BRAF", "ENST00000496384"),
    ):
        transcript = genome.transcript_by_id(transcript_id)
        if not transcript.protein_sequence:
            raise RuntimeError(f"Missing {gene} protein sequence in {genome}")
        print(f"Verified {genome}: {len(transcript.protein_sequence)} {gene} residues")


if __name__ == "__main__":
    main()
