# API Reference

## TopiaryPredictor

::: topiary.predictor.TopiaryPredictor

| Parameter | Type | Description |
|-----------|------|-------------|
| `models` | class, instance, or list | Predictor model(s). Classes require `alleles`. |
| `alleles` | list of str | HLA alleles. Used to construct model classes. |
| `filter` | EpitopeFilter or RankingStrategy | Which peptide-allele groups to keep. |
| `rank_by` | Expr or list of Expr | How to sort surviving groups. |
| `padding_around_mutation` | int | Residues around mutation for candidate epitopes. |
| `only_novel_epitopes` | bool | Drop peptides without mutated residues. |

### Methods

| Method | Input | Output |
|--------|-------|--------|
| `predict_from_named_sequences(dict)` | `{name: sequence}` | DataFrame |
| `predict_from_sequences(list)` | `[sequence, ...]` | DataFrame |
| `predict_from_variants(variants)` | VariantCollection | DataFrame |
| `predict_from_mutation_effects(effects)` | EffectCollection | DataFrame |

## Filter expressions

| Expression | Meaning |
|------------|---------|
| `Affinity <= 500` | IC50 ≤ 500 nM |
| `Affinity.rank <= 2.0` | Affinity percentile rank ≤ 2% |
| `Affinity.score >= 0.5` | Affinity score ≥ 0.5 |
| `Presentation.rank <= 2.0` | EL percentile rank ≤ 2% |
| `Presentation.score >= 0.5` | EL score ≥ 0.5 |

Combine with `|` (OR) and `&` (AND). Chain with `.rank_by()`.

## Expr transforms

| Method | Description |
|--------|-------------|
| `.norm(mean, std)` | Gaussian CDF normalization → [0, 1] |
| `.clip(lo, hi)` | Clamp to range |
| `.log()` / `.log10()` | Logarithm |
| `.exp()` | Exponential |
| `.sqrt()` | Square root |
| `abs(expr)` | Absolute value |
| `expr ** n` | Power |
| `+`, `-`, `*`, `/` | Arithmetic between expressions and scalars |

## String parsing

| String | Equivalent |
|--------|-----------|
| `"affinity <= 500"` | `Affinity <= 500` |
| `"ic50 <= 500"` | `Affinity <= 500` |
| `"presentation.rank <= 2"` | `Presentation.rank <= 2` |
| `"el.score >= 0.5"` | `Presentation.score >= 0.5` |
| `"aff <= 500 \| el.rank <= 2"` | `(Affinity <= 500) \| (Presentation.rank <= 2)` |

## Input functions

| Function | Input | Returns |
|----------|-------|---------|
| `read_fasta(path)` | FASTA file | `{name: sequence}` |
| `read_peptide_fasta(path)` | FASTA of peptides | `{name: peptide}` |
| `read_peptide_csv(path)` | CSV with `peptide` col | `{name: peptide}` |
| `read_sequence_csv(path)` | CSV with `sequence` col | `{name: sequence}` |
| `slice_regions(seqs, regions)` | Sequences + intervals | `{name:start-end: subseq}` |
| `exclude_by(df, ref, mode)` | DataFrame + ref sequences | Filtered DataFrame |

## Source functions

| Function | Returns |
|----------|---------|
| `sequences_from_gene_names(names)` | `{GENE\|TRANSCRIPT: seq}` |
| `sequences_from_gene_ids(ids)` | `{GENE\|TRANSCRIPT: seq}` |
| `tissue_expressed_sequences(tissues)` | `{GENE\|TRANSCRIPT: seq}` |
| `tissue_expressed_gene_ids(tissues)` | `set` of Ensembl gene IDs |
| `cta_sequences()` | CTA protein sequences |
| `ensembl_proteome()` | All Ensembl proteins |
| `available_tissues()` | List of 50 tissue names |
