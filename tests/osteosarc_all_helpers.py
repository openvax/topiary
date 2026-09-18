"""Original-annotation and RNA translation checks, independent of Isovar code."""

from collections import defaultdict
import gzip
from itertools import product
import re

from .osteosarc_helpers import transcript_offset


CODONS = ["".join(b) for b in product("TCAG", repeat=3)]
TABLES = {
    1: dict(zip(CODONS, "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")),
    2: dict(zip(CODONS, "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG")),
}
STARTS = {1: {"TTG", "CTG", "ATG"}, 2: {"ATT", "ATC", "ATA", "ATG", "GTG"}}


def translate(sequence, table, annotated_start=False):
    """NCBI tables 1/2, separating initiation from ordinary codon translation."""
    amino_acids = []
    for offset in range(0, len(sequence) - 2, 3):
        codon = sequence[offset:offset + 3]
        residue = TABLES[table].get(codon, "X")
        if offset == 0 and annotated_start and codon in STARTS[table]:
            residue = "M"
        if residue == "*":
            return "".join(amino_acids), True
        amino_acids.append(residue)
    return "".join(amino_acids), False


def reference_models(root):
    """Parse original GTF/FASTA; independently validate complete coding models."""
    import pysam

    with pysam.FastxFile(str(root / "reference.cdna.fa.gz")) as handle:
        cdnas = {r.name.split(".")[0]: r.sequence for r in handle}
    with pysam.FastxFile(str(root / "reference.pep.fa.gz")) as handle:
        proteins = {re.search(r"transcript:(ENST\d+)", r.comment)[1]: r.sequence for r in handle}
    features = defaultdict(lambda: defaultdict(list))
    with gzip.open(root / "reference.gtf.gz", "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            match = re.search(r'transcript_id "([^"]+)"', fields[8])
            if match:
                features[match[1]][fields[2]].append(fields)
    result = {}
    for tid, rows in features.items():
        if not rows["start_codon"] or tid not in cdnas or tid not in proteins:
            continue
        strand = rows["transcript"][0][6]
        exons = [(int(r[3]), int(r[4])) for r in rows["exon"]]
        start = min(transcript_offset(exons, strand, p) for r in rows["start_codon"]
                    for p in range(int(r[3]), int(r[4]) + 1))
        table = 2 if rows["transcript"][0][0] == "MT" else 1
        if translate(cdnas[tid][start:], table, True)[0] == proteins[tid]:
            result[tid] = dict(strand=strand, exons=exons, cds_start=start,
                               cdna=cdnas[tid], protein=proteins[tid], table=table)
    return result


def validate_rna_protein(result, models):
    """Verify frame, literal RNA translation, stop and mutation interval.

    Additional RNA changes are allowed: the isolated nominated reference edit
    is not substituted for the observed RNA haplotype. No production translation
    or genomic-to-transcript implementation participates in this oracle.
    """
    protein = result.top_protein_sequence
    if protein is None:
        return
    variant = result.variant
    position, ref, alt = variant.start, variant.ref, variant.alt
    # Varcode stores minimally trimmed variants, but verify rather than rely on
    # an anchored source spelling matching the internal coordinate convention.
    while ref and alt and ref[0] == alt[0]:
        position, ref, alt = position + 1, ref[1:], alt[1:]
    for translation in protein.translations:
        orf = translation.variant_orf
        checked = 0
        for transcript in translation.reference_context.transcripts:
            model = models[transcript.id]
            reverse = model["strand"] == "-"
            if ref:
                offsets = sorted(transcript_offset(model["exons"], model["strand"], p)
                                 for p in range(position, position + len(ref)))
                assert offsets == list(range(offsets[0], offsets[0] + len(ref)))
                variant_offset = offsets[0]
            else:
                left, right = [transcript_offset(model["exons"], model["strand"], p)
                               for p in (position - 1, position)]
                assert abs(left - right) == 1
                variant_offset = max(left, right)
            oriented_ref = ref.translate(str.maketrans("ACGT", "TGCA"))[::-1] if reverse else ref
            oriented_alt = alt.translate(str.maketrans("ACGT", "TGCA"))[::-1] if reverse else alt
            assert model["cdna"][variant_offset:variant_offset + len(ref)] == oriented_ref
            start = variant_offset - len(orf.reference_cdna_sequence_before_variant)
            assert model["cdna"][start:variant_offset] == orf.reference_cdna_sequence_before_variant
            frame = model["cds_start"] - start if start < model["cds_start"] else (model["cds_start"] - start) % 3
            assert frame == orf.offset_to_first_complete_codon
            prefix, end = orf.variant_cdna_interval_start, orf.variant_cdna_interval_end
            assert orf.cdna_sequence[prefix:end] == oriented_alt
            sequence, stop = translate(orf.cdna_sequence[frame:], model["table"], start + frame == model["cds_start"])
            if len(sequence) > 49:
                sequence, stop = sequence[:49], False
            assert (translation.amino_acids, translation.ends_with_stop_codon) == (sequence, stop)
            changed_start = (prefix - frame) // 3
            frameshift = (len(ref) - len(alt)) % 3 != 0
            changed_end = len(sequence) if frameshift else (prefix + len(alt) - frame + 2) // 3
            assert translation.frameshift == frameshift
            assert (translation.mutation_start_idx, translation.mutation_end_idx) == (changed_start, min(changed_end, 49))
            checked += 1
        assert checked > 0
