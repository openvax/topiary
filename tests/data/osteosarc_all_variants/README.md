# Full osteosarc index: offline original-RNA regression corpus

184 entries: the pinned 182-entry website index (44 vaccine-nominated), including
all 20 historical pVAC alleles, plus ACSL6 and KTN1 from the candidate report.
174 literal GRCh38 alleles are reference-verified; ten incomplete inputs have
explicit statuses and null RNA results, never invented zero counts.

The BAM, the `openvax-v1` member
`topiary/osteosarc_all_variants/source/t2-all-variant-regions.bam` (see
`../README.md`), retains unmodified records from the union of ±100-base source
intervals in January-2025 UCLA T2 STAR RNA. Reference files retain original Ensembl 87
GTF/cDNA/protein records for all transcripts at the exact alleles. Manifests
record original source URLs, full-source hashes, subset hashes, extraction
command and read/reconstruction settings. Public collection labels establish
the association; no independent genotype fingerprint or deduplication claim.

`expected.json` pins all 184 dispositions/counts/proteins/filters, but is not
the sole scientific oracle: tests independently parse raw GTF and FASTA,
derive reading frames, translate observed RNA and check edit intervals/stop
codons using NCBI codes 1/2. Synthetic prediction scores are transport tests
only. Real model predictions are kept separately in the local audit archive.

Generate from an acquired audit (not during CI):

```sh
PYTHONPATH=. python tests/data/osteosarc_all_variants/regenerate.py AUDIT_ROOT
```

For scope, outcomes, limitations and the other original-read indel/fusion
fixtures, see `docs/osteosarc-variant-outcomes.md` and
`docs/osteosarc-rna-audit.md`. No result here pools samples or declares a peptide
clinically suitable. Source: https://osteosarc.com/variants/.

The historical `expected.json` and `run.json` remain unchanged.
`deletion-label-replay.json` records a complete 174-allele replay with Isovar
1.62.0 and Varcode 10.22.0, checking the published selected-read multiset and
independent raw-reference translation oracle. Only four annotation labels
changed: Varcode 10.21.2 corrected one-based deletion positions and used endpoint
notation; 10.21.3 restored full-sequence notation while retaining the correction
([#644](https://github.com/openvax/varcode/pull/644),
[#647](https://github.com/openvax/varcode/pull/647)). Tests select those recorded
label vocabularies by Varcode version and still compare every other RNA, filter,
sequence and interval field exactly. `replay_deletion_labels.py NEW_OUTPUT_DIR`
reproduces the comparison without rewriting the historical evidence.
