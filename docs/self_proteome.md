# Cross-reactivity: `self_nearest_*` via `SelfProteome`

Given a mutant peptide, find its closest match in a reference proteome
of healthy self peptides. The distance (and the source gene of the
match) is a cross-reactivity risk signal: mutant neoantigens that look
a lot like a real self-peptide may trigger T-cell cross-reactivity
against healthy tissue.

This page covers the `SelfProteome` class and the `self_nearest_*`
columns it adds to `TopiaryPredictor` output.

> **Scope note.** This PR ships the **core architecture and one
> nearest-by-sequence axis**: `include="all"` and `include="non_cta"`,
> substitutions only (no 1aa indels yet), single `self_nearest_peptide`
> scalar. The three-axis story (sequence-nearest, binding-similar,
> strongest-binder), 1aa indel candidates, the full candidate-set
> structured column, and `include="protected_tissues"` are tracked for
> follow-up PRs under [#124](https://github.com/openvax/topiary/issues/124).

## Basic usage

```python
from topiary import SelfProteome, TopiaryPredictor
from mhctools import NetMHCpan

ref = SelfProteome.from_ensembl(species="human")
# Default include="non_cta" strips CTAs via pirlygenes.

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

Three construction modes (this PR ships the first two):

| `include=` | Behavior | Configuration |
|---|---|---|
| `"all"` | Whole proteome, no filter | — |
| `"non_cta"` (default for human Ensembl) | Remove CTA genes | `cta_source="pirlygenes"` default; `"tsarina"` reserved; set / callable accepted |
| `"protected_tissues"` *(PR B)* | Keep only genes expressed in named tissues | `tissues=[…]`, `tissue_source="hpa"`/`"gtex"`, `min_expression=…` |
| callable | Arbitrary `gene → bool` filter | — |

**Human users** get zero-config `include="non_cta"` via pirlygenes:

```python
ref = SelfProteome.from_ensembl(species="human")
```

**Non-human users** must either use `include="all"` or supply their own
CTA source, because pirlygenes is human-only today:

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
construction — silent unfiltered results would be a misleading
cross-reactivity signal.

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
string that captures the species + scope + filter identity. Two runs
produce interchangeable `self_nearest_peptide` values iff the strings
match:

```
ensembl-human-93+scope-non_cta+cta-pirlygenes-3.12.1
ensembl-mouse-102+scope-all
ensembl-human+scope-non_cta+cta-sha256:abc123...
```

Custom filters (user-supplied CTA sets or callables) hash into the
version string so reproducibility holds even when no stable label is
available.

## Algorithm

SIMD-vectorized Hamming distance against int8-encoded reference arrays,
substitutions only. For a query of length L, the search is restricted
to reference peptides of the same length.

### Performance notes

- Construction: one pass over the reference proteome extracts every
  L-mer for each configured length, dedupes per length, and encodes
  into a `(M, L) int8` array. For human non-CTA Ensembl × length 9,
  expect ~200k rows.
- Lookup: per query, the full reference array is compared in one SIMD
  operation. Chunked to bound peak memory. Typical throughput for
  ~200k reference × ~10k queries is seconds.

Seed-and-extend indexing and 1aa indel candidates are queued in
[#124](https://github.com/openvax/topiary/issues/124) — benchmark
decides the default algorithm.
