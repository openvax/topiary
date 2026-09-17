"""Offline, source-grounded regression coverage for all 21 archived reports."""

import hashlib

import pandas as pd

from .pvacseq_corpus_helpers import MANIFEST, REPORTS, ROOT


def test_pinned_corpus_provenance_and_coverage():
    assert len(REPORTS) == 21
    assert sum(r["source_rows"] for r in REPORTS) == 131209
    assert sum(len(r["source_data_rows"]) for r in REPORTS) == 454
    variants = set()
    for report in REPORTS:
        data = (ROOT / report["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == report["sha256"]
        lines = data.splitlines(keepends=True)[1:]
        assert [hashlib.sha256(line).hexdigest() for line in lines] == report["row_sha256"]
        assert len(lines) == len(report["source_data_rows"])
        assert report["source_data_rows"] == sorted(set(report["source_data_rows"]))
        assert all(1 <= n <= report["source_rows"] for n in report["source_data_rows"])
        raw = pd.read_csv(ROOT / report["file"], sep="\t")
        if report["category"] == "filtered":
            assert raw.empty and report["source_rows"] == 0
        if report["category"] == "all_epitopes":
            variants.update(raw[["Chromosome", "Start", "Stop", "Reference", "Variant"]]
                            .itertuples(index=False, name=None))
    assert len(variants) == 20  # Not all website variants, and not independent report views.
    assert ("chr9", 133057893, 133057905, "GGAGGAGGAGGAA", "G") in variants
    assert ("chr17", 80327830, 80327833, "ATAC", "A") in variants
    assert not any(v[0:2] in (("chr9", 3856149), ("chr14", 55627965)) for v in variants)


def test_rna_annotation_gap_precedes_prediction_and_import():
    source = MANIFEST["variant_input"]
    data = (ROOT / source["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == source["sha256"]
    raw = pd.read_csv(ROOT / source["file"], sep="\t", keep_default_na=False)
    assert len(raw) == source["rows"] == 63
    for column in ("trna_depth", "trna_vaf", "gene_expression", "transcript_expression"):
        assert raw[column].eq("NA").all()
    # This does not mean sequencing was absent, or that these later RNA samples
    # can be attached to the historical specimen without establishing identity.
    for report in REPORTS:
        raw = pd.read_csv(ROOT / report["file"], sep="\t", keep_default_na=False)
        columns = ("RNA Depth", "RNA VAF", "RNA Expr") if report["category"] == "aggregated" else (
            "Tumor RNA Depth", "Tumor RNA VAF", "Gene Expression")
        for column in columns:
            assert raw[column].eq("NA").all()
