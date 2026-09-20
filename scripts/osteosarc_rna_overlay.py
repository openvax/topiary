"""Reproduce the January-2025 RNA sidecar; never overwrite historical pVAC data.

Run with the Isovar extra installed, from a Topiary checkout. Acquisition uses
public HTTP sources and samtools indexed range requests, not whole BAM downloads.
The two local roots must be separate. See docs/osteosarc-rna-overlay.md.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import pandas as pd
from osteosarc import digest

from topiary import join_annotations, read_tsv


REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/data/pvacseq/osteosarc"
BASE = "https://sid-sijbrandij-osteosarc-dataset.s3.us-west-2.amazonaws.com/"
RNA = BASE + "genomics/genomics-bulk/2025.01.06/RNA/2025.01.06.rna.ucla-core/processed/"
RSEM = RNA + "RSEM/sj.rna.2025.01.resection.ucla.align.tcga.protocol.dr32.rsem."
BAM = RNA + "STAR/25.03.23.rna.ucla.2025.01.resection.tcga.d32.protocolAligned.sorted.bam"
SOURCES = {"t2.genes.results": RSEM + "genes.results",
           "t2.isoforms.results": RSEM + "isoforms.results"}
POLICY = dict(use_secondary_alignments=False, use_duplicate_reads=False,
              min_mapping_quality=20, use_soft_clipped_bases=False,
              merge_overlapping_fragments=True)
COORDS = ["chromosome_name", "start", "stop", "reference", "variant"]


def write_json(path, value):
    """Write deterministic, human-readable audit metadata."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def panel():
    """Load the pinned historical input and explicit SNV/deletion coordinate map.

    This study contains only SNVs and left-anchored deletions. pVACtools 5.3.0
    reports SNV Start=VCF POS-1, but deletion Start=VCF POS. Both spellings
    are verified against GRCh38 before counting. Other allele forms are refused.
    """
    table = pd.read_csv(FIXTURES / "2025.04.27.variant-input.tsv", sep="\t")
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    # The source table is a complete, unmodified fixture, not a sampled output.
    entry = manifest["variant_input"]
    if digest(FIXTURES / entry["file"]) != entry["sha256"]:
        raise ValueError("Historical variant-input checksum changed")
    unique = table[COORDS].drop_duplicates().copy()
    snv = unique.reference.str.len().eq(1) & unique.variant.str.len().eq(1)
    deletion = unique.variant.str.len().eq(1) & unique.reference.str.len().gt(1)
    deletion &= pd.Series([r.startswith(a) for r, a in zip(unique.reference, unique.variant)],
                          index=unique.index)
    if not (snv | deletion).all():
        raise ValueError("Panel contains an unsupported allele representation")
    unique["vcf_pos"] = unique.start + snv.astype(int)
    expected_stop = unique.start + unique.reference.str.len() - deletion.astype(int)
    if not unique.stop.eq(expected_stop).all():
        raise ValueError("Unexpected pVAC coordinate interval")
    unique["allele_key"] = [f"{c}:{p}:{r}>{a}" for c, p, r, a in zip(
        unique.chromosome_name, unique.vcf_pos, unique.reference, unique.variant)]
    table = table.merge(unique, on=COORDS, how="left", validate="many_to_one")
    return table, unique


def acquire(root, *, cache=None, alignment_source=BAM, index_path=None):
    """Acquire expression/reference inputs and selected reads through Osteosarc.

    Parameters
    ----------
    root : pathlib.Path
        Audit directory; existing acquisitions must have unchanged receipts.
    cache : osteosarc.Cache, optional
        Shared source cache. Offline mode requires already imported sources.
    alignment_source : str or path-like
        Original indexed RNA alignment, or a local alignment for replay.
    index_path : path-like, optional
        Explicit alignment index. Otherwise Osteosarc resolves the index.

    Returns
    -------
    None
        Write source files, an indexed regional BAM and acquisition receipts.
        Historical panel alleles are retained without applying corrections.
    """
    import pysam
    from osteosarc import Cache, Region, extract_reads
    from scripts.osteosarc_variant_audit import fetch_snapshot

    cache = cache if cache is not None else Cache()
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    receipts = {name: fetch_snapshot(url, source / name, cache=cache)
                for name, url in SOURCES.items()}
    _, variants = panel()

    def reference(row):
        filename = row.allele_key.replace(":", "_").replace(">", "_") + ".json"
        path = source / filename
        url = ("https://api.genome.ucsc.edu/getData/sequence?genome=hg38"
               f";chrom={row.chromosome_name};start={row.vcf_pos - 1}"
               f";end={row.vcf_pos - 1 + len(row.reference)}")
        fetch_snapshot(url, path, cache=cache)
        sequence = json.loads(path.read_text())["dna"].upper()
        if sequence != row.reference:
            raise ValueError(f"Reference mismatch for {row.allele_key}: {sequence}")
        return dict(allele_key=row.allele_key, file=filename, url=url,
                    sha256=digest(path), sequence=sequence)

    with ThreadPoolExecutor(max_workers=4) as pool:
        references = list(pool.map(reference, variants.itertuples(index=False)))
    regions = [f"{r.chromosome_name}:{r.vcf_pos - 100}-{r.vcf_pos + len(r.reference) + 100}"
               for r in variants.itertuples(index=False)]
    bam = source / "t2-pvac-regions.bam"
    receipt_path = source / "bam.receipt.json"
    if receipt_path.exists():
        bam_receipt = json.loads(receipt_path.read_text())
        if (bam_receipt["url"] != str(alignment_source) or bam_receipt["regions"] != regions
                or digest(bam) != bam_receipt["sha256"]
                or digest(str(bam) + ".bai") != bam_receipt["index_sha256"]):
            raise ValueError("Regional alignment scope or bytes changed")
    else:
        if bam.exists():
            raise ValueError("Inspect unreceipted BAM before retrying")
        subset = extract_reads(alignment_source,
                               [Region.from_samtools(r, assembly="GRCh38") for r in regions],
                               index=index_path, cache=cache)
        shutil.copyfile(subset.path, bam)
        shutil.copyfile(subset.index_path, str(bam) + ".bai")
        with pysam.AlignmentFile(bam) as handle:
            (source / "t2-pvac-regions.header.sam").write_text(str(handle.header))
        bam_receipt = dict(url=str(alignment_source), regions=regions,
                           sha256=digest(bam), index_sha256=digest(str(bam) + ".bai"),
                           header_sha256=digest(source / "t2-pvac-regions.header.sam"),
                           bytes=bam.stat().st_size, osteosarc_extraction=subset.receipt)
        write_json(receipt_path, bam_receipt)
    write_json(root / "acquisition.json", dict(sources=receipts, references=references,
                                               alignment=bam_receipt))


def build(root):
    """Recount sequenced segments and join uniquely matched, version-labelled TPM."""
    import isovar
    import pysam
    from isovar import ReadCollector
    from isovar.allele_read_helpers import allele_reads_from_locus_reads
    from isovar.read_evidence import ReadEvidence
    from isovar.read_identity import count_reads, source_read_ids
    from isovar.variant_helpers import trim_variant, base0_interval_for_variant_fields
    from varcode import Variant

    table, variants = panel()
    acquisition = json.loads((root / "acquisition.json").read_text())
    source = root / "source"
    for filename, receipt in acquisition["sources"].items():
        if digest(source / filename) != receipt["sha256"]:
            raise ValueError(f"Source checksum mismatch: {filename}")
    bam_path = source / "t2-pvac-regions.bam"
    if digest(bam_path) != acquisition["alignment"]["sha256"]:
        raise ValueError("Regional BAM checksum mismatch")
    reference_checks = {r["allele_key"]: r for r in acquisition["references"]}
    for row in variants.itertuples(index=False):
        check = reference_checks[row.allele_key]
        if (digest(source / check["file"]) != check["sha256"]
                or check["sequence"] != row.reference):
            raise ValueError(f"Reference check changed: {row.allele_key}")
    collector = ReadCollector(**POLICY)
    counts = []
    with pysam.AlignmentFile(bam_path) as bam:
        if bam.get_reference_length("chr1") != 248956422:
            raise ValueError("BAM is not the expected GRCh38 reference")
        for row in variants.itertuples(index=False):
            variant = Variant(row.chromosome_name, int(row.vcf_pos), row.reference,
                              row.variant, ensembl=87, convert_ucsc_contig_names=True)
            # Coordinate-level collection requires no gene annotation download.
            # Use the public trimming, collection and classification functions;
            # expression/gene identity comes from the source table, not Ensembl 87.
            position, ref, alt = trim_variant(variant)
            start, end = base0_interval_for_variant_fields(position, ref, alt)
            reads = collector.get_locus_reads(
                bam, row.chromosome_name, start, end,
                trimmed_base1_start=position, trimmed_ref=ref, trimmed_alt=alt)
            evidence = ReadEvidence.from_variant_and_allele_reads(
                variant, allele_reads_from_locus_reads(reads))
            groups = [getattr(evidence, category + "_reads") for category in ("ref", "alt", "other")]
            identities = [{key for read in group for key in source_read_ids(read)} for group in groups]
            if any(identities[i] & identities[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise ValueError(f"Conflicting segment classifications: {row.allele_key}")
            ref, alt, other = map(count_reads, groups)
            depth = ref + alt + other
            counts.append(dict(allele_key=row.allele_key, ref_reads=ref, alt_reads=alt,
                               other_reads=other, depth_reads=depth,
                               vaf=alt / depth if depth else None,
                               coverage_status="covered" if depth else "zero_usable_reads"))
    counts = pd.DataFrame(counts)
    counts.to_csv(root / "allele-evidence.tsv", sep="\t", index=False, na_rep="NA")
    table = table.merge(counts, on="allele_key", validate="many_to_one")
    for level, filename, key, target in (
        ("gene", "t2.genes.results", "gene_id", "ensembl_gene_id"),
        ("transcript", "t2.isoforms.results", "transcript_id", "transcript_name"),
    ):
        expression = pd.read_csv(source / filename, sep="\t")
        expression["stable_id"] = expression[key].str.replace(r"\.\d+$", "", regex=True)
        selected = expression[expression.stable_id.isin(table[target])].copy()
        if selected.stable_id.duplicated().any():
            raise ValueError(f"Ambiguous {level} stable IDs")
        selected = selected.set_index("stable_id")
        table[level + "_tpm"] = table[target].map(selected.TPM)
        table[level + "_rsem_id"] = table[target].map(selected[key])
        table[level + "_match"] = table[level + "_rsem_id"].notna().map(
            {True: "unique_stable_id", False: "unmatched"})
        if level == "transcript":
            versioned = table.hgvsc.str.split(":").str[0]
            exact = versioned.eq(table.transcript_rsem_id)
            table.loc[exact, "transcript_match"] = "exact_hgvsc_version"
    table.to_csv(root / "transcript-evidence.tsv", sep="\t", index=False, na_rep="NA")
    provenance = dict(
        sample="January 2025 UCLA resection tumor RNA (T2)",
        sample_pairing="Public collection labels/date; not an independent DNA/RNA fingerprint",
        reference="GRCh38; BAM STAR header names GENCODE v36 reference",
        expression="Source RSEM TPM; versioned identifiers retained; no version averaging",
        count_unit="sequenced segments, not independent molecules",
        vaf_denominator="ref_reads + alt_reads + other_reads; missing at zero depth",
        interpretation="Locus allele support, not assembled protein or peptide support",
        method="Isovar ReadCollector", isovar_version=isovar.__version__, policy=POLICY,
        acquisition=acquisition,
    )
    write_json(root / "provenance.json", provenance)


def annotation_table(raw, evidence):
    """Map this study's source report identities to explicit RNA sidecar rows."""
    fields = ["allele_key", "ensembl_gene_id", "ref_reads", "alt_reads", "other_reads",
              "depth_reads", "vaf", "coverage_status", "gene_tpm", "transcript_tpm",
              "gene_rsem_id", "transcript_rsem_id", "gene_match", "transcript_match"]
    if raw.empty:
        return pd.DataFrame(columns=["variant", "transcript", *fields])
    if "ID" in raw and "Best Transcript" in raw:
        keys = raw[["ID", "Best Transcript"]].rename(
            columns={"ID": "variant", "Best Transcript": "transcript"}).drop_duplicates()
        panel_ids = evidence[COORDS].astype(str).agg("-".join, axis=1)
    else:
        keys = raw[["Index", "Transcript"]].rename(
            columns={"Index": "variant", "Transcript": "transcript"}).drop_duplicates()
        # Index is run-specific. Resolve via that report's explicit coordinates,
        # never by the April 27 row number or by gene symbol alone.
        coords = ["Chromosome", "Start", "Stop", "Reference", "Variant"]
        source = raw[["Index", "Transcript"] + coords].drop_duplicates()
        if source.duplicated(["Index", "Transcript"]).any():
            raise ValueError("Source pVAC identity maps to multiple genomic alleles")
        source["panel_key"] = source[coords].astype(str).agg("-".join, axis=1)
        keys = source.rename(columns={"Index": "variant", "Transcript": "transcript"})[
            ["variant", "transcript", "panel_key"]]
        panel_ids = evidence[COORDS].astype(str).agg("-".join, axis=1)
    lookup = evidence[fields].copy()
    lookup["panel_key"] = panel_ids
    lookup["transcript"] = evidence.transcript_name
    if "panel_key" not in keys:
        keys["panel_key"] = keys.variant
    joined = keys.merge(lookup, on=["panel_key", "transcript"], how="left", validate="many_to_one")
    if joined.allele_key.isna().any():
        raise ValueError("A source variant/transcript has no RNA panel identity")
    return joined.drop(columns="panel_key")


def apply(root, archive):
    """Write and verify all 21 enriched copies, leaving historical files intact."""
    evidence = pd.read_csv(root / "transcript-evidence.tsv", sep="\t")
    provenance = json.loads((root / "provenance.json").read_text())
    manifest = json.loads((archive / "archive-manifest.json").read_text())
    imports = json.loads((archive / "import-manifest.json").read_text())
    if not imports.get("complete"):
        raise ValueError("Historical import manifest is not complete")
    import_hashes = {r["source"]: r["output_sha256"] for r in imports["reports"]}
    outputs = []
    for item in manifest["objects"]:
        if item["category"] == "supporting":
            continue
        relative = item["relative_path"]
        raw_path = archive / "raw" / relative
        if digest(raw_path) != item["sha256"]:
            raise ValueError(f"pVAC source changed: {relative}")
        raw = pd.read_csv(raw_path, sep="\t", na_values=["NA", "None"])
        annotations = annotation_table(raw, evidence)
        imported_path = archive / "topiary" / relative
        if digest(imported_path) != import_hashes[relative]:
            raise ValueError(f"Historical Topiary import changed: {relative}")
        original = read_tsv(imported_path)
        enriched = join_annotations(original, annotations, on=["variant", "transcript"],
                                    prefix="rna_t2", provenance=provenance)
        pd.testing.assert_frame_equal(enriched.df[original.df.columns], original.df)
        path = root / "topiary" / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            enriched.to_tsv(path)
        # Existing generated files are verified, never overwritten. This also
        # permits resuming an interrupted 21-file run without discarding data.
        restored = read_tsv(path)
        pd.testing.assert_frame_equal(restored.df, enriched.df.reset_index(drop=True),
                                      check_dtype=False, rtol=1e-12, atol=1e-12)
        if restored.extra["annotation_overlays"] != enriched.extra["annotation_overlays"]:
            raise ValueError("Overlay provenance did not survive serialization")
        outputs.append(dict(file=relative, rows=len(enriched), sha256=digest(path),
                            original_sha256=digest(archive / "topiary" / relative)))
        print(relative, len(enriched), flush=True)
    write_json(root / "overlay-manifest.json", dict(complete=True, reports=outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["acquire", "build", "apply"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.stage == "apply":
        if args.archive is None or args.root.resolve() == args.archive.resolve():
            parser.error("apply requires --archive and a separate --root")
        apply(args.root, args.archive)
    else:
        {"acquire": acquire, "build": build}[args.stage](args.root)
