# Self-sequence and presentation evidence

`SelfProteome` searches an explicit reference corpus. Sequence similarity,
predicted binding and observed presentation are separate facts; none establishes
TCR recognition. Peptide-loaded or minigene recognition can also differ from
recognition of native protein processing and presentation ([primary study](https://www.nature.com/articles/s41541-023-00713-y)).

Use [all-match evidence](#all-match-evidence) for complete vaccine-window searches
or multiple similar candidates. The existing `nearest()` and `self_nearest_*`
predictor columns remain a single-nearest-sequence view.

> **Status.** `SelfProteome` currently exposes the sequence-nearest
> axis: substitutions plus 1aa indel neighbors against a scoped self
> proteome (`include="all"`, `"non_cta"`, `"protected_tissues"`, or a
> callable). Binding-aware axes (`self_mimic_*`, `self_strongest_nearby_*`)
> remain [#412](https://github.com/openvax/topiary/issues/412).
> All-match evidence accepts supplied observations and predictions; it does not
> run models, rank by binding or infer TCR recognition.

## All-match evidence

```python
from topiary import SelfProteome, match_self_peptides, self_matches_in_windows

# Synthetic reference. In production, retain an unfiltered corpus so a shared
# sequence keeps both CTA and non-CTA origins. oncoref owns CTA membership.
reference = SelfProteome.from_peptides(
    {"CTA": "GILGFVFTL", "healthy": "GILGFVFTL", "near": "SIINFEKM"},
    peptide_lengths=[8, 9],
)
windows = {"full": "GILGFVFTLGGGSIINFEKLELAGIGILT", "trimmed": "SIINFEKLELAGIGILT"}
exact = self_matches_in_windows(
    windows, reference, peptide_lengths=[8, 9],
    alleles={name: ["A0201", "B0702"] for name in windows},
    excluded_gene_ids={"CTA"},  # illustrative caller-resolved exclusion
)
similar = match_self_peptides(reference, ["SIINFEKL"], max_mismatches=1, alleles=["A0201"])
exact.to_tsv("window-self-evidence.tsv")
```

`reference.match_candidates(...)` delegates to `match_self_peptides`. Queries
retain order and duplicates through `query_index`; every matched sequence and
gene/transcript/reference-offset origin has a row. `self_match_id` identifies the
reference occurrence across queries. Window output also carries `window_id`,
`window_sequence` and every zero-based `peptide_offset`. The exact adapter checks
all requested lengths and offsets, not only mutation-overlapping peptides.

Exclusions flag `self_in_scope=False`; they never delete origins. A reference
already filtered with `include="non_cta"` cannot recover removed CTA origins.
For an Ensembl corpus use `include="all"` and resolve the exclusion set with
`cta_gene_ids(source="oncoref")` for the appropriate species/tier. A sequence
match alone is not evidence that the peptide was presented.

Both functions accept `observations=` and `predictions=` as tables or record
lists. They preserve multiple records in typed list/dict columns:

| Input | Required fields | Interpretation |
|---|---|---|
| Observations | `evidence_id`, `peptide`, `source`, `evidence_kind`, `allele_assignment` | `evidence_kind` is `observed` or `predicted`; allele assignment is `confirmed`, `predicted` or `unknown`. Confirmed/predicted assignments require an explicit `allele`. |
| Predictions | `peptide`, `allele`, `kind`, `value`, `prediction_method_name`, `predictor_version` | Preserve supplied model facts, including unknown versions, missing values and conflicts; no averaging or model choice. |

Observation records may carry tissue, assay, source version and other
JSON-compatible provenance. An optional `gene_id` restricts attribution to that
origin; its absence does not identify the producing gene. An unresolved
`allele_set` remains a genotype, not confirmed restriction. Alleles are parsed
with mhcgnomes, including non-human alleles.

`self_observations` and `self_predictions` retain the records.
`self_same_allele_predictions` contains only finite numeric `value` measurements
at the query allele with per-allele MHC scope. Haplotype presenter labels do not
become per-allele evidence. This reports any supplied measurement, not complete
coverage of a desired model panel. Other-allele records remain in
`self_predictions`. Observation flags distinguish confirmed and inferred
same-allele assignments; unreported observations remain unknown.

Coverage is explicit:

| Field | Meaning |
|---|---|
| `self_search_status` | `matched`, `no_match_in_scope`, or `unassessed`. |
| `self_search_complete`, `self_search_reason` | Whether the requested canonical, same-length search is complete; unavailable lengths and unsupported query/reference sequences have explicit reasons. A match can coexist with incomplete coverage. |
| `self_observation_status` | Observation input absent, a record reported, or no record reported for this match/origin. |
| `self_prediction_status` | Prediction input absent, query allele unknown, same-allele measurement reported, or not reported. |

`no_match_in_scope` is negative evidence only within the declared reference,
lengths and Hamming radius. It does not mean no presentable peptide or no risk.
Indels, substitutions outside the radius and unsearched reference data are not
assessed. Comparisons use chunks of 65,536 reference rows; exact queries reuse
the reference index. Metadata under `extra['self_search']` records the reference
identity, parameters, coverage counts and hashes of normalized supplied evidence.
`extra['self_window_coverage']` also records requests too short for a peptide.

CSV/TSV preserves these fields and nested records. This is an evidence table,
not a fabricated prediction table: join it to candidate measurements at the
explicit query/allele identity before using a prediction policy. Preserve all
matches and their IDs when there are several per candidate. The composed test
in `tests/test_consumer_workflows.py` checks named-criterion unknown handling and
exact replay after joining and saving the evidence.

Vaxrank owns window selection. `scripts/check_vaxrank_candidates.py` consumes its
released `builtin:openvax-v1` bundle and shows that self evidence changes a window
choice while retaining both intended epitopes and their alleles. That synthetic
fixture uses 100% target-score retention and checks every required target; it
does not establish that a general aggregate-score threshold protects each target.

## Basic usage

```python
from topiary import SelfProteome, TopiaryPredictor
from mhctools import NetMHCpan

ref = SelfProteome.from_ensembl(species="human")
# Default include="non_cta" strips CTAs using oncoref membership via pirlygenes.

predictor = TopiaryPredictor(
    models=NetMHCpan,
    alleles=["HLA-A*02:01"],
    self_proteome=ref,
)
df = predictor.predict_from_variants(variants)

# Output gains:
#   self_nearest_peptide
#   self_nearest_peptide_length
#   self_nearest_edit_distance
#   self_nearest_gene_id
#   self_nearest_transcript_id
#   self_nearest_reference_offset
#   self_nearest_reference_version
```

The columns join onto the predictor output on `peptide`. They're
attached *before* `filter_by` and `sort_by` evaluate, so you can
reference them in DSL expressions:

```python
predictor = TopiaryPredictor(
    models=NetMHCpan,
    alleles=["HLA-A*02:01"],
    self_proteome=ref,
    filter_by=(Affinity <= 500) & (Column("self_nearest_edit_distance") >= 3),
)
```

## Scope

Three construction modes:

| `include=` | Behavior | Configuration |
|---|---|---|
| `"all"` | Whole proteome, no filter | — |
| `"non_cta"` (default for human Ensembl) | Remove CTA genes | `cta_source="pirlygenes"` default; set / callable accepted |
| `"protected_tissues"` | Keep only genes expressed in named tissues | `tissues=[…]`, `min_tissue_ntpm=…` for human HPA data; or explicit `tissue_gene_ids={…}` for any species |
| callable | Arbitrary `gene_id → bool` filter | — |

Tissue expression selects the reference genes; it is not evidence of peptide
presentation. Supply presentation observations separately when available.

**Human users** get zero-config `include="non_cta"` via pirlygenes:

```python
ref = SelfProteome.from_ensembl(species="human")
```

**Non-human users** must either use `include="all"` or supply their own
CTA source, because oncoref's CTA membership is human-only today:

```python
ref = SelfProteome.from_ensembl(species="mouse", release=102, include="all")

# Or with a custom CTA list:
ref = SelfProteome.from_ensembl(
    species="mouse",
    release=102,
    include="non_cta",
    cta_source={"ENSMUSG0001", "ENSMUSG0002", ...},
)
```

A non-human `include="non_cta"` call without `cta_source=` raises at
construction — silent unfiltered results would misstate the reference scope.

## Non-Ensembl sources

For users whose reference proteome isn't in Ensembl, `from_fasta` takes
a protein-FASTA file directly:

```python
ref = SelfProteome.from_fasta("my_reference.fa")
# include="all" by default; callable scope also works.
# include="non_cta" isn't available here — FASTA has no gene metadata.
```

For test or programmatic use:

```python
ref = SelfProteome.from_peptides(
    {"geneA": "MASIINFEKLGGG", "geneB": "QPRSTVWYACDEF"},
    peptide_lengths=[8, 9, 10, 11],
)
```

## Reference version

Every row of the output carries a `self_nearest_reference_version`
string. Matching strings identify the same indexed reference; comparing results
also requires the same queries, metric and search settings. Its form is
`{source}-{species}[-{release}]+include-{scope}+sha256:{digest}`:

```
ensembl-human-115+include-non_cta+cta-oncoref-VERSION+sha256:3f2a9c1e7b40
ensembl-mouse-102+include-all+sha256:91d0c4e2a8b7
fasta-fasta+include-callable-keep_named+sha256:0be51d2c9f63
peptides-synthetic+include-all+sha256:bd8a4e854d1c
```

`source` is the constructor: `ensembl`, `fasta` or `peptides`. When
`from_ensembl` is called without a release, the string records the release
pyensembl selected. The leading parts describe the proteome for a reader;
the digest identifies it. It covers every record the index was built from
(gene id, transcript id and sequence, in order) and the lengths indexed, so
two proteomes share a string only when they hold the same sequences under
the same identifiers. A callable filter is labelled by its qualified name,
which is the same on every run; the digest tells two filters apart.

## Algorithm

Reference and query peptides use the public `encode_amino_acids()` integer
encoding. The default metric uses NCBI's
[BLOSUM62 matrix](https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62).
Canonical pairs retain Topiary's established distance behavior. NCBI's B/J/Z
ambiguity scores use a symmetric conservative transformation, while O/U/X/*
and unrecognized characters receive the symmetric worst-case canonical
distance (15). This prevents missing substitution evidence from looking like
an exact match. `metric="hamming"` counts encoded residue mismatches instead.
For a query of length L, the same-length search is vectorized against that
reference bucket, with 1aa insertion/deletion neighbors checked separately
when enabled.

### Performance notes

- Construction: one pass over the reference proteome extracts every
  L-mer for each configured length, dedupes per length, and encodes
  into a `(M, L) int8` array. Corpus size depends on the release, scope and lengths.
- Lookup: vectorized comparisons are chunked to bound temporary memory.
  Runtime depends on the corpus and query count; exact matching reuses a hash index.

The existing `nearest()` path checks one-residue indel neighbors. The new
all-match API is limited to same-length Hamming candidates; broader candidate
search and binding-ranked axes remain [#412](https://github.com/openvax/topiary/issues/412).
