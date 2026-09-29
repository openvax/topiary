# Reading native Exacto output

`read_exacto` imports native Exacto TSV files into `TopiaryResult` without
running Exacto, reconstructing reads or executing a prediction model. Supply
`sample_name`: these files do not identify the patient/sample themselves.

```python
from topiary import read_exacto

exacto = read_exacto(
    "sample_exacto_peptide_variants.tsv",
    sample_name="patient-01",
    primary_structures="sample_exacto_primary_structures.tsv",
    tag="exacto-run-01",
)
```

## Supported native schemas

The tested producer is [Exacto 0.4.6, commit
307c086](https://github.com/pirl-unc/exacto/tree/307c08670d5e706734bddf393bcebc84db497f9f).
Topiary identifies required column sets rather than assuming that a filename
or the producer's package version establishes a schema. `EXACTO_SCHEMAS`
exports these column sets; `schema=` can select one explicitly. Input may be a
path, gzip-compressed path or open text stream. Extra columns and original
records remain in `result.extra["exacto"]`, including companion tables.

| Schema | Native output | Normalized observations |
| --- | --- | --- |
| `peptide_variants` | `call-peptide-vars` TSV | Reported peptide sequences and parent IDs; optional primary structures supply exact occurrences and context |
| `primary_structures` | `translate-structs` TSV | One translated ORF hypothesis per native peptide ID, with producer-annotated changed intervals |
| `translations` | `translate-seqs` TSV/TSV.gz | Every reported RNA/ORF translation, including partial alternatives |

`transcript_read_support` is a companion table, supplied with the argument of
that name. It is accepted with primary structures, or peptide variants plus
primary structures. Provide `library_id` and `read_set_id`; stream inputs also
require a `tag` identifying the Exacto run. Local transcript model IDs are
scoped by this tag, or by the input path. Use the same run tag when importing
different files from one run. Distinct read names supply a transcript-level
read count with explicit membership. Missing models remain unknown. These
counts are not variant support or ORF-specific abundance; repeated peptide
rows do not create independent RNA observations.

The native corpus in `tests/data/exacto` pins source paths and SHA-256 digests:
228 peptide records, six primary structures and 77 translations. Unsupported
headers, primary record types, sequence-bearing event rows, invalid codons or
inconsistent peptide/parent geometry raise `ValueError`. This reader does not
import arbitrary Exacto tables or reconstruct genomic events from local IDs.

## What the evidence means

Native DNA and RNA call IDs, transcript model IDs, reference transcript IDs and
all source records are retained. Local numeric IDs do not establish agreement
with another caller's genomic events. Translation ORF bounds are converted from
zero-based inclusive coordinates to zero-based half-open coordinates; original
values remain in metadata. Translation IDs include both RNA and peptide IDs,
since the producer reuses peptide IDs across RNA records.

With primary structures, peptide coordinates and flanks are checked against
the reconstructed translation. Primary indices count base and event records,
so they are not treated as amino-acid coordinates. Changed amino-acid intervals
come from Exacto's explicit `amino_acid_change` labels. Without primary
structures, context and novelty intervals remain unknown and the reported
peptide is still available.

`sequence_source="predicted_from_observed_rna"` distinguishes inferred protein
sequence from a protein-expression measurement. A complete `protein_sequence`
requires a start codon and terminal stop. Partial translations remain in
`protein_hypothesis_sequence`. Native files do not establish matched-normal
specificity or comparator protein sequences: these stay unknown. A changed
sequence is not automatically tumor-specific.

## Combine with reported candidates

```python
from topiary import (
    combine_sources, evidence_views, melt_pvacseq_algorithms, rank_candidates,
    read_lens, read_pvacseq, reconcile_evidence,
)

combined = reconcile_evidence(combine_sources({
    "exacto": exacto,
    "lens": read_lens("lens.tsv"),
    "pvacseq": melt_pvacseq_algorithms(read_pvacseq("all_epitopes.tsv")),
}, sample_name="patient-01"))
ranked = rank_candidates(
    combined, "affinity['netmhcpan'].value", ascending=True, duplicates="best",
)
by_source = rank_candidates(
    combined, "affinity['netmhcpan'].value", ascending=True, duplicates="best",
    strata=["source_label", "candidate_mhc_class"],
)
combined.to_tsv("combined-evidence.tsv")
links = evidence_views(combined)["candidate_occurrences"]
```

Exacto's native files contain no HLA assignments or pMHC predictions. Therefore
its rows retain null `allele`, `value` and `candidate_id`; ORF-only rows also
have no peptide. The ranking universe comes from already reported pMHCs.
`candidate_occurrences` links same-sample, identical peptide occurrences to
those queries, with `reported_candidate=False` for a sequence match. This does
not transfer expression, scores or eligibility to a different observation.
All source records and native metadata survive long/wide CSV and TSV.

Choose compatible original measurements explicitly. [Combined-source
ranking](combined-sources.md) explains conflicts, missing scores and additive
`rescore_candidates` calls; rescoring alone does not invent HLA combinations.

## Optional new windows

`read_exacto_fragments(path, **kwargs)` accepts the same inputs and produces
`ProteinFragment` records with the recovered sequence, novelty intervals and
native provenance. It calls no model. Empty native tables return an empty list.

Pass these fragments to `TopiaryPredictor.predict_from_fragments` to explicitly
scan new peptide windows. This changes the candidate universe and is separate
from rescoring reported candidates. Select lengths/alleles deliberately and
use `overlaps_target` to filter recovered changed intervals. Missing intervals
remain unknown. Fragment conversion preserves each native observation, so
several reported peptides from one parent may carry the same parent sequence.

The general reader uses no Varcode fusion reconstruction: that adapter handles
selected DNA-SV-linked linear RNA models, whereas these native translated
sequences can be imported directly without genomic anchors or raw reads.
