"""Fail CI setup unless the dependency-selected human reference is usable."""

from pyensembl import ensembl_grch38


def main():
    # The same BRAF transcript is exercised by Topiary's sequence-source tests.
    # This lookup needs both the indexed annotation and its sequence data.
    transcript = ensembl_grch38.transcript_by_id("ENST00000496384")
    if not transcript.protein_sequence:
        raise RuntimeError(f"Missing BRAF protein sequence in {ensembl_grch38}")
    print(f"Verified {ensembl_grch38}: {len(transcript.protein_sequence)} BRAF residues")


if __name__ == "__main__":
    main()
