# Shared Osteosarc inputs and regional audit continuity

Topiary uses **Osteosarc 0.1.0** for new audit downloads, indexed regional read
extraction and access to the shared OpenVax source objects. Install the optional
tooling on Python 3.10+ with `pip install -e '.[isovar,osteosarc]'`; extraction
also needs `samtools` on PATH. The base Topiary package remains Python 3.9+ and
does not import Osteosarc or acquire data on import.

## Original reads, derived proteins, prediction caches

`tests/data/osteosarc_shared/vaccine-rna-v1` is the unchanged portable source
manifest adopted by [Vaxrank #486](https://github.com/openvax/vaxrank/pull/486).
It contains 49 cases covering 44 original vaccine loci, with 98 original
BAM/index files totaling 4,126,730 bytes. All objects are pinned to Isovar commit
`0cad5b275c852263a1c77722aa463fe5568b2c76`, with sizes and SHA-256 checksums.
Native alleles, original source products, region digests and selection rules
remain in the manifest. These are selected regression reads, not full-source
coverage or independent biological replicates. The original public data are
[CC0](https://registry.opendata.aws/sid-osteosarc/).

`translation-v1.json` separately pins a real NTF3 RNA reconstruction, including
its exact protein window and target interval, transcripts, gene, species,
reference identity, read counts, filter outcomes and reconstruction policy.
Its parent is the shared source manifest digest. NTF3's adjacent RNA change
is retained: `NKLSKQMVDVSENYQSTLPK` is not the isolated reference A>G edit.
The regression checks the original cDNA/GTF frame and translates the observed
RNA independently of Isovar's translation code.

`prediction-contract-v1.json` is a separately versioned **synthetic score**
fixture for that real sequence. It tests transport, coverage, filtering and
ranking, not binding accuracy. Its keys retain predictor name/version, kind,
allele, peptide, length and flank/genotype context; its parent is the translated
fixture digest. The source hash alone never supplies prediction coverage.

The historical source manifest still contains the old MAP2 deletion.
Osteosarc's corrected complex allele is not substituted during acquisition.
Changing that allele and reviewing its protein expectations requires a new,
explicitly reviewed data revision. The existing 184-entry T2 audit and its
historical pVAC predictions likewise retain their reviewed inputs.

## Acquire, share, verify and reproduce

From a checkout:

```sh
# Explicit acquisition; never part of ordinary tests.
python -m scripts.osteosarc_test_data --cache-root /path/to/shared-openvax

# Offline export to a NEW directory, using those verified shared objects.
python -m scripts.osteosarc_test_data --cache-root /path/to/shared-openvax \
  --offline --output /path/to/new/export

# Read-only verification of the checked-in original bytes.
python -m scripts.osteosarc_test_data \
  --verify tests/data/osteosarc_shared/vaccine-rna-v1

# Regenerate derived fixtures offline into a NEW directory for comparison.
PYTHONPATH=. python tests/data/osteosarc_shared/regenerate.py /path/to/new/derived
```

The default root is Osteosarc's `openvax` cache, overridable through
`OPENVAX_DATA_CACHE` or an explicit `osteosarc.Cache(root)`. Objects use the
shared `objects/sha256/<sha256><original suffixes>` convention. Verified objects
created by Vaxrank/datacache are adopted through Osteosarc's public
`Cache.import_file` without rewriting their bytes or requiring Vaxrank.
Corrupt objects fail visibly; this workflow never repairs them implicitly.
An interrupted export has no `manifest.json`; remove or inspect it explicitly
before choosing a fresh output directory.

The public `topiary.osteosarc_fixture_paths(manifest, directory=...)` verifies an
offline export without cache writes. With `cache=Cache(..., offline=True)` it
resolves the same assets from the shared cache. **Pass the returned BAI path
explicitly** when opening/extracting its BAM: the two content hashes differ,
so their cache filenames are not adjacent `file.bam` / `file.bam.bai` names.
Tests drive both paths through extraction, native Varcode conversion, Isovar,
fragment IO, prediction, filtering, ranking and prediction IO.

New regional audit acquisition uses Osteosarc's public `extract_reads` with the
pinned inventory's original coordinates and the existing read-selection scope.
Receipts retain the full extraction request and source/index identities.
Complete SAM records are compared before/after migration, including qualities,
flags, mate fields, CIGARs, tags and repeated-record multiplicity. Topiary retains
its historical inventory parser and scientific selection/reconstruction policy.

## Missing regional annotation

The regional reference intentionally retains only overlapping transcripts.
A contig can exist in the original assembly/alignment without appearing in
that subset. The audit now records `no_regional_annotation`, with **null RNA
evidence**, and continues to subsequent variants. It does not call that a zero
read count. Annotation-dependent collection is tracked in
[Isovar #295](https://github.com/openvax/isovar/issues/295).
`alignment_contig_unavailable` and `outside_alignment_contig` are separate input
outcomes. Other execution failures still raise; they are not relabelled as
biological negatives.

The #348 regression retains unmodified FAM157A GTF records from
[Ensembl release 87](https://ftp.ensembl.org/pub/release-87/gtf/homo_sapiens/Homo_sapiens.GRCh38.87.gtf.gz).
The gene starts at 198153287, beyond the recovered insertion at chr3:198153259;
the historical RefSeq claim remains input provenance. No coding effect is
invented. The test builds the regional reference, runs the audit past this
entry to a real DYNC1H1 reconstruction, and generates the report. An entirely
unannotated selection also produces explicit outcomes without a fabricated GTF.

Reports can be generated before diagnosis/prediction and state which stage
has not run. Audit schema 2 changes resumable run identity: use a new audit
directory for a replay of a pre-change run rather than mixing outcomes.
