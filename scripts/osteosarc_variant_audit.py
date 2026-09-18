"""Account for every website variant; retain data and explicit negative outcomes.

Acquisition is an explicit local operation, never a CI side effect. Original
source bytes are cached with receipts. No historical pVAC prediction is changed.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
from datetime import datetime, timezone
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import requests

from scripts.osteosarc_rna_overlay import BAM, digest, write_json


METADATA = {
    "variants.html": "https://osteosarc.com/variants/",
    "vafs.tsv": "https://osteosarc.com/variants/variant_vafs_long.tsv",
    "vafs-columns.tsv": "https://osteosarc.com/variants/variant_vafs_long.columns.tsv",
}
EXTRA_INDELS = [
    ("CGNL1", "chr15", 57451531, "TATTGGAACAGAAAAGCA", "T"),
    ("GTF3C5", "chr9", 133057893, "GGAGGAGGAGGAA", "G"),
    ("GLIS3", "chr9", 3856149, "CTGATGTGG", "C"),
    ("RNF213", "chr17", 80327830, "ATAC", "A"),
    ("ACSL6", "chr5", 131988563, "G", "GTA"),
    ("KTN1", "chr14", 55627965, "G", "GTT"),
    ("PIP5K1A", "chr1", 151242178, "AG", "A"),
    ("TAF7L", "chrX", 101286650, "GT", "G"),
    ("SPAG1", "chr8", 100213209, "CG", "C"),
    ("FGFR3", "chr4", 1805817, "GC", "G"),
]


def fetch_snapshot(url, path):
    """Cache unmodified source bytes and refuse mismatched existing receipts."""
    path = Path(path)
    receipt_path = path.with_name(path.name + ".receipt.json")
    if path.exists() or receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["url"] != url or receipt["sha256"] != digest(path):
            raise ValueError(f"Cached input changed: {path}")
        return receipt
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise ValueError(f"Inspect unfinished acquisition before retrying: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        with partial.open("xb") as handle:
            for block in response.iter_content(1024 * 1024):
                handle.write(block)
        receipt = dict(url=url, sha256=digest(partial), bytes=partial.stat().st_size,
                       acquired_at=datetime.now(timezone.utc).isoformat(),
                       etag=response.headers.get("ETag"))
    partial.rename(path)
    write_json(receipt_path, receipt)
    return receipt


def upstream_structural_evidence(root):
    """Cache immutable upstream original-read evidence, not only a PR summary."""
    base = ("https://raw.githubusercontent.com/openvax/isovar/"
            "fa44889fcf1e946450d91875fcaf5886b1723e79/")
    names = ["tests/data/osteosarc/figure_comparisons/" + suffix for suffix in (
        "EXTENDED_RNA.md", "CORPUS_OUTCOMES.md", "SOFT_CLIPS.md",
        "corpus/dlg5-manifest.json", "corpus/dlg5.json.gz",
        "corpus/extended-footprints-manifest.json", "corpus/extended-footprints.json.gz",
        "corpus/insertion-boundaries-manifest.json", "corpus/insertion-boundaries.json.gz",
        "dlg5.py", "extended_footprints.py")]
    for name in names:
        fetch_snapshot(base + name, root / "source/upstream-isovar-1.18.1" / Path(name).name)


class VariantIndex(HTMLParser):
    """Read exact website entry IDs, including entries lacking genomic alleles."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.current = None
        self.cells = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr" and "data-vaccines" in attrs:
            vaccines = (attrs["data-vaccines"] or "").strip()
            if not vaccines.isdigit():
                raise ValueError(f"Website variant table schema changed: data-vaccines={vaccines!r}")
            self.current = {"vaccine_count": int(vaccines)}
            self.cells = []
        elif self.current is not None:
            if tag == "td":
                self.cell = []
            elif tag == "a" and attrs.get("href", "").startswith("/variant/"):
                self.current["variant_id"] = attrs["href"].rstrip("/").split("/")[-1]

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.current is not None and self.cell is not None:
            self.cells.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.current is not None:
            if "variant_id" not in self.current or len(self.cells) != 9:
                raise ValueError("Website variant table schema changed")
            self.rows.append(dict(self.current, gene=self.cells[0],
                                  source_location=self.cells[1], source_protein_label=self.cells[2]))
            self.current = None


#: Columns the long VAF table must provide. A missing one means the source
#: schema changed, which no per-entry status can describe.
VAF_ALLELE_COLUMNS = ["variant_id", "chrom", "pos", "ref", "alt"]


def literal_alleles(vafs_text):
    """Distinct candidate alleles per website ID, parsed as a typed table.

    Every cell is read as text and nothing is coerced while parsing, so one
    malformed row cannot abort the others. ``position`` is ``pos`` as an
    integer only where ``pos`` is a positive whole number written in digits,
    and missing otherwise. ``literal`` also requires ``ref`` and ``alt`` to be
    ACGT bases. Rows that are not literal remain, so their entries can be
    reported with an explicit status instead of disappearing.

    Raises
    ------
    ValueError
        A required column is missing.
    """
    table = pd.read_csv(io.StringIO(vafs_text), sep="\t", dtype=str, keep_default_na=False)
    missing = [column for column in VAF_ALLELE_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"VAF table schema changed: missing columns {missing}")
    alleles = table[VAF_ALLELE_COLUMNS].apply(lambda column: column.str.strip()).drop_duplicates()
    digits = alleles["pos"].str.fullmatch(r"[1-9][0-9]*")
    alleles["position"] = pd.to_numeric(alleles["pos"].where(digits)).astype("Int64")
    alleles["literal"] = (digits & alleles["ref"].str.fullmatch("[ACGT]+")
                          & alleles["alt"].str.fullmatch("[ACGT]+"))
    return alleles.sort_values(
        ["variant_id", "chrom", "position", "pos", "ref", "alt"], na_position="last")


def variant_inventory(index_html, vafs_text):
    """Resolve literal alleles without discarding ambiguous or unavailable inputs.

    Returns one dictionary per website entry. A complete, consistent GRCh38
    allele has ``input_status=ready``. Otherwise the original entry and its
    candidate alleles remain available for investigation, with an explicit
    non-ready status: ``missing_literal_allele``, ``ambiguous_literal_allele``,
    ``non_literal_allele`` (including an unusable position) or
    ``conflicting_coordinates``. Protein labels never substitute for a
    nucleotide allele.
    """
    index = VariantIndex()
    index.feed(index_html)
    if not index.rows or len({r["variant_id"] for r in index.rows}) != len(index.rows):
        raise ValueError("Missing or duplicate website variant IDs")
    alleles = literal_alleles(vafs_text)
    candidates = alleles.groupby("variant_id", sort=False)
    entries = pd.DataFrame(index.rows)
    entries["n_candidates"] = entries["variant_id"].map(candidates.size()).fillna(0).astype(int)
    only = alleles[alleles["variant_id"].map(candidates.size()).eq(1)].set_index("variant_id")
    entries = entries.join(only, on="variant_id")
    entries["input_status"] = np.select(
        [entries["n_candidates"].eq(0), entries["n_candidates"].gt(1),
         ~entries["literal"].eq(True),
         entries["source_location"].ne(entries["chrom"] + ":" + entries["pos"])],
        ["missing_literal_allele", "ambiguous_literal_allele", "non_literal_allele",
         "conflicting_coordinates"],
        default="ready",
    )

    # A candidate's position is an integer where usable, else the source text.
    listed = {
        variant_id: [
            [a["chrom"], a["pos"] if pd.isna(a["position"]) else int(a["position"]), a["ref"], a["alt"]]
            for a in group.to_dict("records")
        ]
        for variant_id, group in candidates
    }
    result = []
    for entry, status in zip(index.rows, entries["input_status"]):
        record = dict(entry, source_url=f"https://osteosarc.com/variant/{entry['variant_id']}/",
                      assembly="GRCh38", candidate_alleles=listed.get(entry["variant_id"], []))
        if len(record["candidate_alleles"]) == 1:
            chrom, pos, ref, alt = record["candidate_alleles"][0]
            record.update(chrom=chrom, pos=pos, ref=ref, alt=alt)
        record["input_status"] = str(status)
        if status == "ready":
            record["allele_key"] = f"GRCh38:{record['chrom']}:{record['pos']}:{record['ref']}>{record['alt']}"
        result.append(record)
    return result


def ensembl_contig(chrom):
    """The Ensembl contig for a UCSC chromosome name; ``chrM`` is ``MT``."""
    return "MT" if chrom == "chrM" else chrom.removeprefix("chr")


def audit_variant(record, genome):
    """The varcode variant for one checked inventory record.

    The one place a record becomes a variant, so every stage (audit,
    diagnosis, tests) uses the same contig naming.
    """
    from varcode import Variant

    return Variant(ensembl_contig(record["chrom"]), record["pos"], record["ref"],
                   record["alt"], ensembl=genome)


def check_mutation_windows(frame, fragments):
    """Verify each prediction is a window of its fragment covering the edit.

    Checked from the fragments' own sequences and target intervals, not from
    the predictor's ``contains_mutant_residues``, which already selected these
    rows and so cannot catch its own mistake.

    Raises
    ------
    ValueError
        A row's peptide is not its fragment's sequence at that offset, or the
        window does not overlap the fragment's mutation interval.
    """
    by_id = {fragment.fragment_id: fragment for fragment in fragments}
    rows = frame[["fragment_id", "peptide", "peptide_offset", "peptide_length"]]
    for fragment_id, peptide, offset, length in rows.itertuples(index=False):
        fragment = by_id[fragment_id]
        start, end = int(offset), int(offset) + int(length)
        if fragment.sequence[start:end] != peptide:
            raise ValueError(f"{peptide} is not {fragment_id}[{start}:{end}]")
        if not any(start < high and low < end for low, high in fragment.target_intervals or ()):
            raise ValueError(f"A non-mutant peptide escaped selection: {peptide} in {fragment_id}")


def inventory(root):
    """Pin the full website index and retain additional named exact indels."""
    source = root / "source"
    receipts = {name: fetch_snapshot(url, source / name) for name, url in METADATA.items()}
    variants = variant_inventory((source / "variants.html").read_text(), (source / "vafs.tsv").read_text())
    existing = {r.get("allele_key") for r in variants}
    for gene, chrom, pos, ref, alt in EXTRA_INDELS:
        key = f"GRCh38:{chrom}:{pos}:{ref}>{alt}"
        if key not in existing:
            variants.append(dict(variant_id=f"{gene}-{chrom}-{pos}-{ref}-{alt}",
                                 gene=gene, chrom=chrom, pos=pos, ref=ref, alt=alt,
                                 allele_key=key, assembly="GRCh38", input_status="ready",
                                 source_url="https://osteosarc.com/oncoanalyser/tables/snv_top.tsv",
                                 membership="additional_candidate_report", vaccine_count=None))
            existing.add(key)
    write_json(root / "inventory.json", dict(sources=receipts, variants=variants))
    return variants


def acquire(root, index_path):
    """Verify reference alleles and fetch a union of original regional RNA records."""
    variants = json.loads((root / "inventory.json").read_text())["variants"]
    source = root / "source"

    def check(record):
        if record["input_status"] != "ready":
            return record
        chrom, pos, ref = record["chrom"], record["pos"], record["ref"]
        url = (f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chrom}"
               f";start={pos - 1};end={pos - 1 + len(ref)}")
        path = source / "reference" / (record["variant_id"] + ".json")
        receipt = fetch_snapshot(url, path)
        sequence = json.loads(path.read_text())["dna"].upper()
        return dict(record, reference_check=receipt, observed_reference=sequence,
                    input_status="ready" if sequence == ref else "reference_mismatch")

    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(check, variants))
    write_json(root / "checked-inventory.json", dict(variants=checked))
    regions = sorted({f"{v['chrom']}:{max(1, v['pos'] - 100)}-{v['pos'] + len(v['ref']) + 100}"
                      for v in checked if v["input_status"] == "ready"})
    bam = source / "t2-all-variant-regions.bam"
    receipt_path = source / "bam.receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt["regions"] != regions or receipt["sha256"] != digest(bam)
                or receipt["index_sha256"] != digest(str(bam) + ".bai")):
            raise ValueError("Regional alignment scope or bytes changed")
        return
    if bam.exists():
        raise ValueError("Inspect unreceipted BAM before retrying")
    command = ["samtools", "view", "--no-PG", "-b", "-M", "-X", BAM,
               str(index_path), *regions, "-o", str(bam)]
    subprocess.run(command, check=True)
    subprocess.run(["samtools", "index", str(bam)], check=True)
    write_json(receipt_path, dict(url=BAM, command=command, regions=regions,
                                 sha256=digest(bam), index_sha256=digest(str(bam) + ".bai"),
                                 source_index_sha256=digest(index_path), bytes=bam.stat().st_size,
                                 acquired_at=datetime.now(timezone.utc).isoformat()))


def build_reference(root, ensembl):
    """Retain unmodified Ensembl records for all transcripts at checked loci."""
    variants = json.loads((root / "checked-inventory.json").read_text())["variants"]
    by_contig = defaultdict(list)
    for record in variants:
        if record["input_status"] == "ready":
            by_contig[ensembl_contig(record["chrom"])].append(
                (record["pos"], record["pos"] + len(record["ref"]) - 1))
    transcripts, genes = set(), set()
    gtf = ensembl / "Homo_sapiens.GRCh38.87.gtf.gz"
    with gzip.open(gtf, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if fields[2] != "transcript":
                continue
            if any(int(fields[3]) <= end and int(fields[4]) >= start
                   for start, end in by_contig[fields[0]]):
                attrs = dict(re.findall(r'(\S+) "([^"]*)";', fields[8]))
                transcripts.add(attrs["transcript_id"])
                genes.add(attrs["gene_id"])
    destination = root / "reference"
    destination.mkdir(exist_ok=True)
    files = {}
    for kind, source in (
        ("gtf", gtf),
        ("cdna", ensembl / "Homo_sapiens.GRCh38.cdna.all.fa.gz"),
        ("pep", ensembl / "Homo_sapiens.GRCh38.pep.all.fa.gz"),
    ):
        chunks, keep = [], False
        with gzip.open(source, "rb") as handle:
            for line in handle:
                if kind == "gtf":
                    attrs = dict(re.findall(rb'(\S+) "([^"]*)";', line))
                    keep = (line.startswith(b"#") or attrs.get(b"transcript_id", b"").decode() in transcripts
                            or (line.split(b"\t")[2:3] == [b"gene"]
                                and attrs.get(b"gene_id", b"").decode() in genes))
                elif line.startswith(b">"):
                    match = (re.match(rb">(ENST\d+)", line) if kind == "cdna" else
                             re.search(rb"transcript:(ENST\d+)", line))
                    keep = bool(match and match[1].decode() in transcripts)
                if keep:
                    chunks.append(line)
        path = destination / ("reference." + kind + (".gz" if kind == "gtf" else ".fa.gz"))
        packed = gzip.compress(b"".join(chunks), mtime=0)
        if path.exists() and path.read_bytes() != packed:
            raise ValueError(f"Reference subset changed: {path}")
        path.write_bytes(packed)
        category = "gtf/homo_sapiens/" if kind == "gtf" else f"fasta/homo_sapiens/{kind}/"
        files[path.name] = dict(sha256=digest(path), source_sha256=digest(source),
                               source_url="https://ftp.ensembl.org/pub/release-87/" + category + source.name)
    write_json(destination / "manifest.json", dict(files=files, transcripts=sorted(transcripts),
                                                   assembly="GRCh38", ensembl_release=87))
    print("Pinned", len(transcripts), "transcripts at", len(genes), "genes", flush=True)


def reference_genome(root, cache=None):
    """Load the pinned subset into its own cache without mutating shared indices."""
    from pyensembl import Genome

    reference = root / "reference"
    manifest = json.loads((reference / "manifest.json").read_text())
    for name, receipt in manifest["files"].items():
        if digest(reference / name) != receipt["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {name}")
    genome = Genome(
        reference_name="GRCh38-osteosarc-all-" + digest(reference / "manifest.json")[:16],
        annotation_name="osteosarc-ensembl-subset", annotation_version=87,
        gtf_path_or_url=str(reference / "reference.gtf.gz"),
        transcript_fasta_paths_or_urls=[str(reference / "reference.cdna.fa.gz")],
        protein_fasta_paths_or_urls=[str(reference / "reference.pep.fa.gz")],
        copy_local_files_to_cache=True, cache_directory_path=str(cache or root / "reference-index"))
    genome.index()
    return genome


def audit(root):
    """Run unchanged result filters and keep one outcome for every input entry."""
    import isovar
    import pysam
    from topiary import describe_isovar_result, fragment_from_isovar_result, write_fragments
    from scripts.osteosarc_rna_overlay import POLICY

    genome = reference_genome(root)
    variants = json.loads((root / "checked-inventory.json").read_text())["variants"]
    bam_path = root / "source/t2-all-variant-regions.bam"
    receipt = json.loads((root / "source/bam.receipt.json").read_text())
    if (digest(bam_path) != receipt["sha256"]
            or digest(str(bam_path) + ".bai") != receipt["index_sha256"]):
        raise ValueError("Regional BAM checksum mismatch")
    outcomes = root / "outcomes"
    outcomes.mkdir(exist_ok=True)
    creator = isovar.ProteinSequenceCreator(protein_context_peptide_length=25, variant_sequence_assembly=True)
    collector = isovar.ReadCollector(**POLICY)
    run = dict(isovar_version=isovar.__version__, collection_policy=POLICY,
               protein_context_peptide_length=25, variant_sequence_assembly=True,
               bam_sha256=receipt["sha256"],
               inventory_sha256=digest(root / "checked-inventory.json"),
               reference_sha256=digest(root / "reference/manifest.json"))
    run_path = root / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run:
        raise ValueError("Run inputs/settings changed; use a separate audit directory")
    write_json(run_path, run)
    for record in variants:
        path = outcomes / (record["variant_id"] + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["input"] != record or (record["input_status"] == "ready" and (
                    saved["isovar_version"] != isovar.__version__ or saved["collection_policy"] != POLICY)):
                raise ValueError(f"Outcome inputs/version changed: {path}")
            continue
        if record["input_status"] != "ready":
            write_json(path, dict(input=record, status=record["input_status"], rna=None))
            continue
        variant = audit_variant(record, genome)
        print("AUDIT", record["variant_id"], flush=True)
        with pysam.AlignmentFile(bam_path) as bam:
            result, = isovar.run_isovar([variant], bam, read_collector=collector,
                                       protein_sequence_creator=creator)
        description = describe_isovar_result(result)
        fragment = fragment_from_isovar_result(result)
        if fragment is not None:
            # Retain diagnostic sequences separately even when filters reject them.
            destination = root / ("accepted" if result.passes_all_filters else "filtered")
            destination.mkdir(exist_ok=True)
            fragment.annotations.update(audit_variant_id=record["variant_id"],
                                        audit_filter_status=description["status"],
                                        audit_sample="T2 January 2025 UCLA resection")
            write_fragments([fragment], destination / (record["variant_id"] + ".tsv"))
        write_json(path, dict(input=record, status=description["status"], rna=description,
                             isovar_version=isovar.__version__, collection_policy=POLICY))
        print("RESULT", record["variant_id"], description["status"], flush=True)


def predict(root, hla_input):
    """Score only accepted RNA fragments with the archived class-I genotype.

    These are new MHCflurry affinity predictions, not historical pVAC scores,
    measured presentation, or a vaccine ranking. Model provenance is required.
    """
    import yaml
    from mhctools import MHCflurry_Affinity
    from topiary import TopiaryPredictor, TopiaryResult, read_fragments, read_tsv

    configuration = yaml.safe_load(hla_input.read_text())
    alleles, lengths = configuration["alleles"], configuration["epitope_lengths"]
    files = sorted((root / "accepted").glob("*.tsv"))
    fragments = [fragment for path in files for fragment in read_fragments(path)]
    if not fragments:
        raise ValueError(
            f"No accepted RNA fragment TSVs in {root / 'accepted'} (audits before "
            "topiary 5.63 named these TSV files *.json)")
    model = MHCflurry_Affinity(alleles=alleles, default_peptide_lengths=lengths)
    if not model.predictor_version:
        raise ValueError("MHCflurry model provenance is unknown")
    predictor = TopiaryPredictor(models=model, only_novel_epitopes=True)
    frame = predictor.predict_from_fragments(fragments)
    check_mutation_windows(frame, fragments)
    observed = set(frame.audit_variant_id)
    expected = {f.annotations["audit_variant_id"] for f in fragments}
    if observed != expected:
        raise ValueError(f"Accepted fragments lost during prediction: {expected - observed}")
    provenance = dict(
        interpretation="New predicted class-I affinity; not measured presentation or clinical eligibility",
        hla_source_sha256=digest(hla_input), alleles=alleles, peptide_lengths=lengths,
        predictor_version=model.predictor_version,
        fragments={str(path.relative_to(root)): digest(path) for path in files},
        reconstruction=json.loads((root / "run.json").read_text()),
        mitochondrial_origin="MT-ND5 mitochondrial translation is conditional on origin; NUMTs not excluded",
    )
    path = root / "new-mhcflurry-affinity.tsv"
    TopiaryResult(frame, extra={"osteosarc_audit": provenance}).to_tsv(path)
    restored = read_tsv(path)
    if set(restored.df.audit_variant_id) != expected or restored.extra["osteosarc_audit"] != provenance:
        raise ValueError("Prediction identity/provenance lost on round trip")
    write_json(root / "prediction-receipt.json", dict(provenance, sha256=digest(path), rows=len(frame)))
    print("Predicted", len(frame), "rows for", len(expected), "accepted observations", flush=True)


def diagnose_untranslated(root):
    """Record which public reconstruction stage returns no coding sequence."""
    import isovar
    import pysam
    from scripts.osteosarc_rna_overlay import POLICY

    class TracedCreator(isovar.ProteinSequenceCreator):
        """Observe public stage boundaries without changing their answers."""

        def __init__(self):
            super().__init__(protein_context_peptide_length=25, variant_sequence_assembly=True)
            self.stages = {}

        def variant_sequences_from_reads(self, variant, reads):
            reads = list(reads)
            self.stages["compatible_alt_read_objects"] = len(reads)
            sequences = super().variant_sequences_from_reads(variant, reads)
            self.stages["rna_sequence_count"] = len(sequences)
            return sequences

        def all_pairs_translations(self, variant_sequences, reference_contexts):
            self.stages["reference_context_count"] = len(reference_contexts)
            translations = list(super().all_pairs_translations(variant_sequences, reference_contexts))
            self.stages["translation_count"] = len(translations)
            return translations

    genome = reference_genome(root)
    details = []
    for path in sorted((root / "outcomes").glob("*.json")):
        outcome = json.loads(path.read_text())
        if outcome["status"] not in ("no_protein_sequence", "no_predicted_coding_change"):
            continue
        record = outcome["input"]
        variant = audit_variant(record, genome)
        creator = TracedCreator()
        with pysam.AlignmentFile(root / "source/t2-all-variant-regions.bam") as bam:
            result, = isovar.run_isovar([variant], bam, read_collector=isovar.ReadCollector(**POLICY),
                                       protein_sequence_creator=creator)
        if result.top_protein_sequence is not None:
            raise ValueError("Diagnostic tracing changed reconstruction")
        stages = creator.stages
        if "rna_sequence_count" not in stages:
            reason = "no_coding_reference_context"
        elif not stages["compatible_alt_read_objects"]:
            reason = "no_alt_reads_compatible_with_annotated_coding_paths"
        elif not stages["rna_sequence_count"]:
            reason = "no_rna_sequence_meets_context_and_coverage_requirements"
        else:
            reason = "no_translation_passes_reference_matching"
        details.append(dict(variant_id=record["variant_id"], reason=reason, stages=stages))
    write_json(root / "untranslated-diagnostics.json", details)


def additional_indels(root, hla_input):
    """Replay the already pinned T1/T2 GLIS3/KTN1 libraries without pooling them."""
    import isovar
    import pysam
    from topiary import (
        describe_isovar_result, fragment_from_isovar_result, fragments_for_sample, write_fragments,
    )
    from tests.osteosarc_helpers import load_osteosarc, assert_expected_fragment

    output = root / "additional-indels"
    output.mkdir(exist_ok=True)
    variants, bams, expected = load_osteosarc(output / "reference-and-alignments", "osteosarc_indels")
    rows = []
    for name, bam_path in sorted(bams.items()):
        gene, sample = name.split(".")
        for secondary in (True, False):
            policy = "default_placements" if secondary else "primary_placements"
            identifier = name + "." + policy
            with pysam.AlignmentFile(bam_path) as bam:
                result, = isovar.run_isovar(
                    [variants[gene]], bam,
                    read_collector=isovar.ReadCollector(use_secondary_alignments=secondary),
                    protein_sequence_creator=isovar.ProteinSequenceCreator(
                        protein_context_peptide_length=25, variant_sequence_assembly=True))
            description = describe_isovar_result(result)
            fragment = fragment_from_isovar_result(result)
            if fragment is not None:
                assert_expected_fragment(fragment, expected[gene])
                # Each library and placement policy is its own observation.
                fragment, = fragments_for_sample([fragment], f"{sample}.{policy}")
                destination = output / ("accepted" if result.passes_all_filters else "filtered")
                destination.mkdir(exist_ok=True)
                fragment.annotations.update(audit_variant_id=identifier, audit_sample=sample,
                                            audit_filter_status=description["status"],
                                            audit_placement_policy=policy)
                write_fragments([fragment], destination / (identifier + ".tsv"))
            rows.append(dict(case_id=identifier, gene=gene, sample=sample, placement_policy=policy,
                             status=description["status"], rna=description))
    source = Path(__file__).resolve().parents[1] / "tests/data/osteosarc_indels/manifest.json"
    write_json(output / "outcomes.json", rows)
    write_json(output / "run.json", dict(isovar_version=isovar.__version__,
               fixture_manifest_sha256=digest(source),
               collection_policy="Isovar defaults, changing only use_secondary_alignments per labelled row",
               sample="Four separately labelled T1/T2 ONT/oncoanalyser libraries; not the T2 STAR overlay"))
    predict(output, hla_input)


def report(root):
    """Render one source-linked outcome per inventory entry, without hiding gaps."""
    variants = json.loads((root / "checked-inventory.json").read_text())["variants"]
    outcomes = [json.loads((root / "outcomes" / (r["variant_id"] + ".json")).read_text()) for r in variants]
    diagnostics = {r["variant_id"]: r["reason"] for r in json.loads((root / "untranslated-diagnostics.json").read_text())}
    predictions = json.loads((root / "prediction-receipt.json").read_text())
    extras = [r["gene"] for r in variants if r.get("membership") == "additional_candidate_report"]
    statuses = Counter(r["input_status"] for r in variants)
    ready = statuses.pop("ready", 0)
    lines = ["# Osteosarc: accountable variant outcomes", "",
             f"All {len(variants) - len(extras)} website entries plus {' and '.join(extras) or 'no others'}. "
             f"{ready} literal alleles are GRCh38-reference-verified; the other {sum(statuses.values())} "
             "entries keep explicit input statuses: "
             + (", ".join(f"{count} {status}" for status, count in sorted(statuses.items())) or "none") + ".",
             "This main pass uses January-2025 UCLA T2 STAR RNA only; a negative here is not a negative across timepoints.",
             "Read counts are sequenced segments, not independent molecules. Recorded filters are unchanged.",
             "Reference: Ensembl 87, with original GTF/cDNA/protein records and regional BAM retained locally.",
             "Public collection labels associate samples; no independent DNA/RNA fingerprint is claimed.", "",
             "## T2 outcomes", "", "| Outcome | Entries |", "| --- | ---: |"]
    lines.extend(f"| {status} | {count} |" for status, count in sorted(Counter(r["status"] for r in outcomes).items()))
    lines.extend(["", f"New class-I affinity predictions: **{predictions['rows']:,} rows**, model `{predictions['predictor_version']}`.",
                  "These are predictions, not measured presentation, tumor specificity or clinical eligibility.",
                  "MT-ND5 translation assumes mitochondrial origin; NUMTs have not been excluded.",
                  "Historical pVAC scores and tiers remain unchanged in their separate archive.", "",
                  "## Every entry", "", "| Variant / source | Outcome | Ref / alt / other reads | Protein aa | Reason / failed filters |",
                  "| --- | --- | --- | ---: | --- |"])
    for value in outcomes:
        record, rna = value["input"], value["rna"]
        counts = (" / ".join(str(rna[f"num_{k}_reads"]) for k in ("ref", "alt", "other")) if rna else "unavailable")
        reason = diagnostics.get(record["variant_id"], "; ".join(rna["failed_filters"]) if rna else "Exact literal allele not supplied")
        length = len(rna["protein_sequence"]) if rna and rna["protein_sequence"] else "—"
        lines.append(f"| [{record['variant_id']}]({record['source_url']}) | {value['status']} | {counts} | {length} | {reason} |")
    supplemental = root / "additional-indels/outcomes.json"
    if supplemental.exists():
        lines.extend(["", "## Other libraries: GLIS3 / KTN1", "",
                      "Distinct libraries and placement policies, not extra independent observations to add together.", "",
                      "| Case | Outcome | Alt reads / templates | Protein aa | Failed filters |", "| --- | --- | ---: | ---: | --- |"])
        for row in json.loads(supplemental.read_text()):
            rna = row["rna"]
            lines.append(f"| {row['case_id']} | {row['status']} | {rna['num_alt_reads']} / {rna['num_alt_fragments']} | "
                         f"{len(rna['protein_sequence']) if rna['protein_sequence'] else '—'} | {'; '.join(rna['failed_filters'])} |")
    lines.extend(["", "## Structural and larger-deletion leads", "",
                  "See the source-pinned [RNA path audit](../../docs/osteosarc-rna-audit.md).",
                  "GABBR1–SLC29A1 and OTUD7A–FMN1 have observed RNA paths but unresolved coding frames; no peptide is invented.",
                  "DLG5's sequence-resolved DNA junction audit is complete (Isovar PR #298); RNA junction and coding frame remain unresolved. No internal-deletion protein is inferred for a canonical-start-loss event.",
                  "AFF3/KEAP1 ordinary intron-spanning RNA cannot discriminate the intronic deletion from reference.",
                  "SPRED1, the suspicious TGFBR2 calls and organoid-only TCF7L2 remain unvalidated structural leads, not literal coding alleles.", ""])
    upstream = root / "source/upstream-isovar-1.18.1/EXTENDED_RNA.md"
    if upstream.exists():
        lines.extend(["The [completed extended upstream audit](source/upstream-isovar-1.18.1/EXTENDED_RNA.md) and its compact original-read evidence are cached locally.",
                      "It queries 14 RNA products without pooling them. DLG5 has no recovered exact RNA junction signature, but this does not exclude a spliced product or subclonality.",
                      "The additional T3 GABBR1 path has unresolved endpoint/base-quality limitations, not a newly validated fusion protein.", ""])
    (root / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inventory", "acquire", "reference", "audit", "predict", "diagnose", "additional", "upstream", "report"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--ensembl", type=Path)
    parser.add_argument("--hla-input", type=Path)
    args = parser.parse_args()
    if args.stage == "inventory":
        records = inventory(args.root)
        print(len(records), dict(Counter(r["input_status"] for r in records)))
    elif args.stage == "acquire":
        if args.index is None:
            parser.error("acquire requires --index")
        acquire(args.root, args.index)
    elif args.stage == "reference":
        if args.ensembl is None:
            parser.error("reference requires --ensembl")
        build_reference(args.root, args.ensembl)
    elif args.stage == "audit":
        audit(args.root)
    elif args.stage == "diagnose":
        diagnose_untranslated(args.root)
    elif args.stage == "upstream":
        upstream_structural_evidence(args.root)
    elif args.stage == "report":
        report(args.root)
    else:
        if args.hla_input is None:
            parser.error("predict requires --hla-input")
        if args.stage == "additional":
            additional_indels(args.root, args.hla_input)
        else:
            predict(args.root, args.hla_input)
