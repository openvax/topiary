# Shared Osteosarc inputs and regional audit continuity

Topiary uses **Osteosarc 0.1.1** for new audit downloads, indexed regional read
extraction and access to the shared OpenVax source objects. Osteosarc and its
read-extraction dependency are installed by `pip install topiary` on Python
3.10+. From a checkout, use `pip install -e .`; add `.[isovar]` for RNA
reconstruction. Extraction also needs `samtools` on PATH. Importing Topiary
does not acquire data.

## Original reads, derived proteins, prediction caches

`tests/data/manifest.json` describes all seven Sid fixture groups: six-locus
RNA, GLIS3/KTN1 indels, rearrangements, the RNA overlay, the 184-entry audit,
shared NTF3 reads and historical pVACseq reports. Tests verify every bundled
file through Osteosarc without downloading anything. The 104 assets total
10,623,870 bytes; alignments and indexes account for 7,133,515 bytes.
They ship in the **source distribution**, alongside the tests and generator;
the installed wheel retains its existing runtime-only layout.

The checked-in recipe, `tests/data/sid-fixtures.json`, pins original regional
inputs and zero-based half-open test intervals. `generate_sid_fixtures` uses
Osteosarc's cache and `extract_reads` to select the union of those intervals.
Allele intervals use Osteosarc's `Variant.region(padding=2)`, intersected
with the archived extraction scope for the indel fixtures. These tight
windows retain the bases needed at indel boundaries without broadening the
historical input selection. Original SAM records,
including qualities, flags, tags and mate fields, are never rewritten. Reads
outside the selected intervals are omitted, and no entire remote BAM is
fetched. The six-locus fixtures retain their original template-selection and
quality-stress policies. The rearrangement fixtures already contain only the
nine tested junction paths and their 18 original SAM records.

The audit BAM contains 55,610 records instead of 124,046; the overlay contains
8,312 instead of 16,497. All 184 audit outcomes and the overlay evidence remain
unchanged. These are selected regression reads, not full-source coverage or
independent biological replicates. The original public data are
[CC0](https://registry.opendata.aws/sid-osteosarc/).

`tests/data/osteosarc_shared/vaccine-rna-v1` retains just the NTF3 case exercised
by Topiary's shared reconstruction/ranking workflow: 15 reads, one BAM and its
index (43,966 bytes), byte-identical to the objects adopted by
[Vaxrank #486](https://github.com/openvax/vaxrank/pull/486). The other 48 cases
are omitted. Its subset manifest retains the upstream manifest's digest,
original source products, native allele and selection policy. Input objects
remain pinned to Isovar commit `0cad5b275c852263a1c77722aa463fe5568b2c76`.

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

Historical MAP2 fixtures keep their old deletion. Osteosarc's corrected
complex allele is not substituted during acquisition. Scientific corrections
require separate fixture/outcome review; the existing 184-entry T2 audit and
historical pVAC predictions retain their reviewed inputs.

[Osteosarc #4](https://github.com/iskandr/osteosarc/issues/4) is fixed in 0.1.1:
Topiary now delegates catalogue parsing to `parse_variants`. Malformed VAF
rows become `malformed_source_row` with original `parse_errors` diagnostics;
later valid variants still run. Raw pinned inputs do not apply catalogue
corrections implicitly. [Osteosarc #5](https://github.com/iskandr/osteosarc/issues/5)
still tracks five unresolved alleles.

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
  --verify tests/data

# Generate the minimal corpus from pinned regional archives (first run fetches).
# The recipe, including source hashes and requested loci, is checked in.
python -m scripts.generate_sid_fixtures --output /path/to/new/sid-fixtures \
  --cache-root /path/to/shared-openvax

# Repeat without any acquisition after the original input objects are cached.
python -m scripts.generate_sid_fixtures --output /path/to/new/reproduction \
  --cache-root /path/to/shared-openvax --offline

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
its scientific selection/reconstruction policy while Osteosarc owns parsing.

The generator's `--source-directory` accepts the historical input tree named
by the recipe's source commit; it does not accept already-trimmed outputs as
original inputs. `--live-alignments` additionally compares complete selected
SAM records with indexed extraction from the original remote BAMs. Default
regeneration uses immutable archived regional inputs, making it independent of
changes to the live site. Complete-record comparisons include repeated-record
multiplicity. Tests inject an irrelevant read and prove extraction removes it.
BAM compression can vary between tool versions; scientific equality is checked
using canonical complete-record digests as well as the published file hashes.

The built source distribution is checked against every asset's size and hash,
its exact alignment-file membership, and an 11 MiB corpus budget. Full BAMs and
unused vaccine read sets cannot enter a release silently. Receipts distinguish
the archived broader source from the new fixture selection; historical audit
run provenance still identifies the original run.

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
