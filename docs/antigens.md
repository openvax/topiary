# Antigen Fragments

`AntigenFragment` is a universal record for a protein / peptide sequence with source-type, target-region, and comparator metadata. It's the substrate that lets Topiary handle antigens from any origin — somatic variants, structural variants, ERVs, CTAs, viral proteins, allergens, autoantigens, synthetic constructs — through one pipeline, and threads identity through predictions so downstream tools (vaxrank, vaccine-window selection) can group peptides back to their source.

## The dataclass

```python
from topiary import AntigenFragment

f = AntigenFragment.from_variant(
    sequence="MAAVTDVGMAVATGSWDSFLKIWN",
    reference_sequence="MAAVTDVGMAAATGSWDSFLKIWN",   # WT protein
    mutation_start=10, mutation_end=11, inframe=True,
    variant="chr7:140753336",
    effect="p.Val600Glu",
    gene="BRAF",
    annotations={"vaf": 0.42, "ccf": 0.9},
)
```

Every field except `fragment_id` and `sequence` is optional. `target_intervals` carries the half-open regions within `sequence` considered targetable — its meaning depends on source type (see vocabulary below).

## Running predictions

```python
from topiary import TopiaryPredictor

predictor = TopiaryPredictor(models=[...], alleles=[...])
df = predictor.predict_from_antigens(fragments)
```

Output DataFrame columns, beyond the standard prediction fields:

| Column | Meaning |
|---|---|
| `fragment_id` | Source identity — threads back to `AntigenFragment.fragment_id` |
| `source_type`, `variant`, `effect`, `effect_type`, `gene`, `gene_id`, `transcript_id` | Propagated from the fragment |
| `gene_expression`, `transcript_expression` | Propagated from the fragment |
| `overlaps_target` | `True` / `False` / NaN — whether the peptide overlaps any of the fragment's target intervals |
| `contains_mutant_residues` | Backwards-compat alias — `True` iff `source_type.startswith("variant")` AND `overlaps_target` is True |
| `wt_peptide`, `wt_peptide_length` | Derived by slicing `effective_baseline` at the peptide's offset; `None` when no baseline exists |
| *(each annotation key)* | Flattened from every fragment's `annotations` dict |

## source_type vocabulary (recommended, not enforced)

Free-form string. Topiary never interprets it; used for display and DSL filtering. Colon subtyping is convention.

| Category | Values |
|---|---|
| Variant, small | `variant:snv`, `variant:indel`, `variant:frameshift`, `variant:stop_gain`, `variant:silent` |
| Structural variant | `sv:fusion`, `sv:tandem_duplication`, `sv:inversion`, `sv:translocation`, `sv:cryptic_exon`, `sv:large_insertion`, `sv:large_deletion` |
| Aberrant expression | `erv`, `cta`, `tumor_overexpressed`, `intron_retention`, `utr`, `novel_orf` |
| Pathogen | `viral`, `viral:hpv16`, `viral:hiv`, `bacterial`, `parasitic` |
| Environmental | `allergen`, `allergen:plant`, `allergen:food`, `allergen:mold`, `allergen:dander` |
| Self / autoimmunity | `self`, `autoantigen`, `autoantigen:myelin` |
| Synthetic | `synthetic`, `designed` |

Producers are free to invent new subtypes.

## target_intervals — geometry per source type

The producer computes `target_intervals`; Topiary never interprets. Meaning varies by source type:

| source_type | `target_intervals` |
|---|---|
| `variant:snv` at position k | `[(k, k+1)]` |
| `variant:indel` (in-frame insertion) at k, length L | `[(k, k+L)]` |
| `variant:indel` (in-frame deletion) at k | `[(k, k)]` — the junction where formerly-distant residues now sit together |
| `variant:frameshift` at k | `[(k, len(sequence))]` — everything downstream is novel (sequence should be truncated at the new stop) |
| `sv:fusion` (in-frame, coding-coding) with junction at k | `[(k-1, k+1)]` — junction residues only; internal partner residues are self |
| `sv:fusion` onto non-coding partner with junction at k | `[(k, len(sequence))]` — readthrough translation is all novel |
| `sv:tandem_duplication` with breakpoints at k1, k2 | `[(k1, k1+1), (k2, k2+1)]` — breakpoints only; duplicated bulk is self |
| `sv:inversion` within coding region [a, b] | `[(a, b)]` — reversed translation is entirely novel |
| `sv:cryptic_exon` (in-frame inclusion) at [a, b] | `[(a, b)]` |
| `sv:cryptic_exon` (frameshift inclusion) at [a, b] | `[(a, len(sequence))]` |
| `erv`, `cta` | Producer-computed non-self regions (based on the producer's definition of "self" — healthy-tissue expression, homology to non-CTA proteins). `None` when the producer can't decide. |
| `viral`, `allergen` | Immunodominant / IgE-reactive hotspots if known; `None` otherwise |

`None` means unspecified — downstream tools decide whether to treat as "whole sequence." Empty list `[]` explicitly means "nothing targetable."

## Reference vs germline

```python
reference_sequence: str | None      # canonical (Ensembl, RefSeq, reference strain)
germline_sequence: str | None       # patient / strain-specific baseline
```

The DSL's `wt.*` scope reads `effective_baseline`:

```python
@property
def effective_baseline(self) -> str | None:
    return self.germline_sequence if self.germline_sequence is not None else self.reference_sequence
```

Germline takes precedence when populated; reference is the fallback. Both `None` → `wt.*` returns NaN.

| source_type | typical `reference_sequence` | typical `germline_sequence` |
|---|---|---|
| `variant:*` | Canonical WT from varcode/Ensembl | Patient's non-tumor protein if available |
| `sv:*` | Usually `None` | `None` |
| `viral[:strain]` | Reference-strain protein | `None` (patient has no germline virus) |
| `erv` | `None` | `None` |
| `cta` | Canonical protein (equals `sequence`) | Same as reference (CTAs are non-neoantigens) |
| `autoantigen` | Canonical (UniProt MBP etc.) | Patient-specific with SNPs — can matter for TCR specificity |
| `allergen` | Canonical isoform | `None` (patient doesn't have it) |
| `synthetic` | Natural parent if any, else `None` | `None` |

## Reserved DSL scope: `self_nearest`

For cross-reactivity filtering — "what's the closest peptide in essential healthy tissues, and does it also bind this MHC?"

Topiary **does not compute** these — producers populate externally (via BLAST / edit distance against a healthy-tissue proteome with their own definition of "self"). The DSL scope just reads `self_nearest_*` columns. When columns are absent, `self_nearest.*` returns NaN.

Reserved column namespace:

| Column | Meaning |
|---|---|
| `self_nearest_peptide` | Closest healthy-tissue peptide at the same length |
| `self_nearest_peptide_length` | (Trivially same as the mutant) |
| `self_nearest_edit_distance` | Producer-chosen distance metric (Hamming / Levenshtein / BLAST score) |
| `self_nearest_gene` | Source gene of the nearest-self hit |
| `self_nearest_gene_id`, `self_nearest_transcript_id` | Provenance |
| `self_nearest_tissues` | Which healthy tissues the source gene is expressed in |
| `self_nearest_value`, `self_nearest_score`, `self_nearest_percentile_rank` | MHC binding of the nearest-self peptide, paired to the same allele |

```python
from topiary import Affinity, Column, apply_filter, self_nearest

# Drop neoepitopes too similar to healthy-tissue self
df = apply_filter(
    df,
    (Affinity.score >= 0.5) & (Column("self_nearest_edit_distance") >= 3),
)

# Ranking that penalizes cross-reactivity
ranking = Affinity.score - 0.5 * self_nearest.Affinity.score
```

## IO

```python
from topiary import read_antigens, write_antigens

write_antigens(fragments, "my_antigens.tsv")
loaded = read_antigens("my_antigens.tsv")
```

TSV format: one row per fragment. Scalar fields map to same-named columns. `target_intervals` and `annotations` are JSON-encoded in their own columns. Missing columns on read fall back to field defaults; unknown columns raise.

For single-fragment / API use: `fragment.to_dict()`, `fragment.to_json()`, and the `from_dict` / `from_json` classmethods — stdlib only, no dependencies.

## Identity

`fragment_id` is canonical. Two fragments with the same id are equal and hash-equal, regardless of other content. Use `make_fragment_id(prefix, sequence, variant=...)` for a deterministic content-derived id with a readable prefix:

```
BRAF_p.Val600Glu__a1b2c3d4
EWSR1-FLI1_fusion__3c8e4b91
erv_Hsap38.chr7.64991215__7f2e89a1
HPV16_E6__5f6a1c23
__4f9c2a8e                  # no metadata → hash-only fallback
```

Prefix is sanitized to `[A-Za-z0-9._:-]`; runs of other characters collapse to `_`. Hash is 8 hex chars of SHA-1 over `sequence` + optional `variant`.

## What's not in this release

- **WT model predictions** (`wt_value`, `wt_score`, `wt_percentile_rank`) — `wt_peptide` is derived but not fed through the MHC predictor. `wt.Affinity.score` returns NaN until a future PR populates these.
- **Nearest-self compute** — the scope is reserved but no Topiary module produces the columns. Populate externally for now.
- **Varcode refactor** — the existing `predict_from_variants` pipeline doesn't yet emit `AntigenFragment`s; it still takes its own path. Planned for a follow-up PR.
- **Format-specific loaders** (`read_lens_fragments`, `read_pvacseq_fragments`, `read_isovar_fragments`) — each ~50-100 lines on top of the core abstraction; separate PRs.
