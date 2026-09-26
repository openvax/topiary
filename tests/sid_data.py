"""Topiary's Sid fixtures: checked-in files, plus reads from openvax-v1.

Everything except the reads is checked in under ``tests/data`` and verified
offline against ``tests/data/manifest.json``. The reads come from openvax-v1,
the OpenVax libraries' shared Sid test data (iskandr/osteosarc#56): each file
listed in ``SHARED_READS`` is the openvax-v1 member ``topiary/<path>``, holding
exactly its original records. The first read downloads the bundle into the
osteosarc cache if needed; each read file is exported there once and reused.
"""

from functools import lru_cache
import json
from pathlib import Path

from topiary import osteosarc_fixture_paths


ROOT = Path(__file__).parent / "data"
SHARED_BUNDLE = "openvax-v1"
SHARED_READS = (
    "isovar_repeats/reads.bam",
    "osteosarc/bulk_star_t0.sam.gz",
    "osteosarc/ont_t1.sam.gz",
    "osteosarc_all_variants/source/t2-all-variant-regions.bam",
    "osteosarc_indels/GLIS3.T1-ONT-dedup.sam.gz",
    "osteosarc_indels/GLIS3.T1-short.sam.gz",
    "osteosarc_indels/GLIS3.T2-ONT-dedup.sam.gz",
    "osteosarc_indels/GLIS3.T2-short.sam.gz",
    "osteosarc_indels/KTN1.T1-ONT-dedup.sam.gz",
    "osteosarc_indels/KTN1.T1-short.sam.gz",
    "osteosarc_indels/KTN1.T2-ONT-dedup.sam.gz",
    "osteosarc_indels/KTN1.T2-short.sam.gz",
    "osteosarc_rna_overlay/source/t2-pvac-regions.bam",
    "osteosarc_shared/vaccine-rna-v1/28-NTF3-chr12-5494381-53f498a544883d51.bam",
)


@lru_cache(maxsize=1)
def sid_fixture_paths():
    """Verify the checked-in fixture files once per test worker; never fetch."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    return osteosarc_fixture_paths(manifest, directory=ROOT)


def sid_data_root(dataset):
    """Return one checked-in fixture directory after verifying the entire export."""
    paths = sid_fixture_paths()
    if not any(name.startswith(dataset + "/") for name in paths):
        raise ValueError(f"Unknown Sid fixture group: {dataset}")
    return ROOT / dataset


def sid_read(name):
    """The Sid read file ``tests/data/<name>`` as a read-only, indexed BAM.

    Whatever the original format, the BAM (``.bai`` alongside) holds exactly
    the original file's records, repeats included, coordinate-sorted; the order
    of records at one position may differ.
    """
    if name not in SHARED_READS:
        raise KeyError(f"Not a shared Sid read file: {name}")
    import osteosarc

    return Path(osteosarc.bundle_file(SHARED_BUNDLE, "topiary/" + name))
