# Protein Fragments

`ProteinFragment` is a universal record for a protein / peptide sequence with source-type, target-region, and comparator metadata. It's the substrate that lets Topiary handle antigens from any origin — somatic variants, structural variants, ERVs, CTAs, viral proteins, allergens, autoantigens, synthetic constructs — through one pipeline, and threads identity through predictions so downstream tools (vaxrank, vaccine-window selection) can group peptides back to their source.

## The dataclass

```python
from topiary import ProteinFragment

f = ProteinFragment.from_variant(
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
df = predictor.predict_from_fragments(fragments)
```

Output DataFrame columns, beyond the standard prediction fields:

| Column | Meaning |
|---|---|
| `fragment_id` | Source identity — threads back to `ProteinFragment.fragment_id` |
| `source_type`, `variant`, `effect`, `effect_type`, `gene`, `gene_id`, `transcript_id`, `transcript_name` | Propagated from the fragment |
| `gene_expression`, `transcript_expression` | Propagated from the fragment |
| `overlaps_target` | `True` / `False` / NaN — whether the peptide overlaps any of the fragment's target intervals |
| `contains_mutant_residues` | Backwards-compat alias — `True` iff `source_type.startswith("variant")` AND `overlaps_target` is True |
| `wt_peptide`, `wt_peptide_length` | Derived by slicing `effective_baseline` at the peptide's offset. Only populated for substitution-compatible fragments (baseline and `sequence` the same length); `None` otherwise or when no baseline exists. |
| `wt_value`, `wt_score`, `wt_affinity`, `wt_percentile_rank`, `wt_prediction_method_name`, `wt_predictor_version` | Populated when `TopiaryPredictor(predict_wt=True)` scores non-null `wt_peptide` values with the configured MHC model(s). Rows without a length-compatible WT peptide keep NaN values. |
| *(each annotation key)* | Flattened from every fragment's `annotations` dict. Underscore-prefixed keys (e.g. `_subsequence_offset`) are reserved for internal plumbing and never surface as columns. |

## Building a fragment from a variant effect

`fragment_from_effect(effect, padding_around_mutation)` is the varcode arm of the multi-source story — the same `ProteinFragment` a LENS or pVACseq reader produces, built from a translated variant effect instead:

```python
from topiary import fragment_from_effect

fragment = fragment_from_effect(effect, padding_around_mutation=8)
```

Two rules worth knowing before you rely on it:

- The window is clipped at the protein's **first stop codon**, so a mutation near the end yields a shorter fragment rather than a padded one. If the stop falls before the reported mutation, there is nothing to present and the function returns `None`.
- `reference_sequence` is populated **only** when the pre- and post-mutation proteins are the same length. An indel or frameshift shifts every downstream residue, so the window sliced at the same offsets would be a different piece of protein presented as a comparator — the same restriction `wt_peptide` applies.

It returns `None` when the effect has no mutant protein sequence at all (silent, non-coding, untranslatable). That is an absence, not an error.

## Building fragments from variants, with or without RNA

```python
from topiary import fragments_from_variants

# RNA available: the context around each mutation is assembled from reads
fragments = fragments_from_variants(variants, alignment_file=bam)

# No RNA: the same variants translated from the reference
fragments = fragments_from_variants(variants)
```

**The two arms are interchangeable.** Both return `ProteinFragment`s with the
same core, so the rest of a pipeline does not change shape when the RNA does or
does not exist. What changes is what the fragments can tell you: an assembled
sequence carries the patient's other variants and whatever phasing the reads
support, and comes with counted read support; a translated one carries the
reference everywhere except the variant itself, and no read counts at all.
`annotations["sequence_source"]` says which.

`protein_sequence_length` is a *sequence* length, not a peptide length — a
fragment is scanned by a sliding window downstream, so the assembled context has
to be long enough to contain every peptide that could cover the mutation.
Default 25 (a 9-mer plus 8 either side). `padding_around_mutation` defaults to
half of it, so the reference arm produces a comparable window.

`allow_reference_fallback=True` translates variants isovar could not support
rather than dropping them. The fragments stay distinguishable by
`sequence_source`, which is the reason to record it rather than blend an
RNA-backed candidate with an inferred one.

isovar is needed **only** when `alignment_file` is given. A caller without RNA
never touches the optional dependency — there is a test asserting the reference
arm works with isovar unimportable.

## Every source reaches a fragment

Four paths, one shape. They differ only in which fields they can fill:

| Source | Builder | Sequence is | Read counts |
|---|---|---|---|
| isovar | `fragment_from_isovar_result(result)` | assembled from RNA reads | **counted** (`measured`) |
| varcode | `fragment_from_effect(effect, padding)` | translated from the reference | none |
| LENS | `fragments_from_dataframe(read_lens(p).df)` | the peptide's context | overlapping counted; support is CDS-overlap (`approximated`) |
| pVACseq | `fragments_from_dataframe(read_pvacseq(p).df)` | the peptide itself | depth × VAF (`approximated`) |

```python
def rna_support(fragment):
    if not fragment.is_usable_as_biology("n_alt_reads"):
        return None
    return fragment.n_alt_reads, fragment.is_approximate("n_alt_reads")
```

That function reads all four without knowing which it has: isovar returns
`(30, False)`, pVACseq returns a count with `True`, varcode returns `None`
because it has no RNA data at all. **No branching on `source_type`** — which is
the whole point of the abstraction, and why `source_type` stays biological.

`SEMANTIC_CORE` names the fields every source is expected to speak to, whether
or not it can populate them.

### isovar is optional in the strong sense

Not imported at module scope, not in `requirements.txt`, and topiary's shape is
identical whether or not it is installed — `import topiary` does not import it.
Only `fragment_from_isovar_result` needs it, and it says how to install it if
missing. A consumer that only reads LENS reports should not pay for a package
it never calls.

isovar is also the only source that *counts* the reads supporting an assembled
sequence. Everything else derives them or counts something adjacent, which is
what the derivation names record.

## RNA evidence, and knowing which fields are real

Beyond `gene_expression` / `transcript_expression`, a fragment can carry
read-level evidence:

| Field | Meaning |
|---|---|
| `n_overlapping_reads` | Reads spanning the variant position |
| `n_alt_reads` | Reads supporting the variant allele |
| `n_ref_reads` | Reads supporting the reference allele |
| `n_alt_reads_supporting_protein_sequence` | Reads supporting *this assembled protein sequence*, not merely the allele |

These are not derivable from a TPM, which is why they are fields rather than
annotations.

**`None` is unknown; it is not `0`.** A source with no read data leaves them
`None`. A source that looked and found no support sets `0`. A consumer must be
able to tell those apart, so ask `fragment.is_known("n_alt_reads")` rather than
testing truthiness — and the distinction survives writing to and reading from a
TSV.

Different sources populate different subsets, and some of them *estimate*. So a
fragment can also say how real each populated field is:

```python
from topiary import ProteinFragment, APPROXIMATED, SYNTHESIZED

ProteinFragment(
    fragment_id="...",
    sequence="...",
    variant="chr1:100:N>N",       # invented by the loader; not real alleles
    n_alt_reads=12,               # reconstructed as depth x VAF, not counted
    field_provenance={"variant": SYNTHESIZED, "n_alt_reads": APPROXIMATED},
)
```

| Provenance | Meaning |
|---|---|
| `"measured"` | Observed directly from data |
| `"approximated"` | Derived or estimated — e.g. read counts as depth × VAF |
| `"synthesized"` | A placeholder the loader invented because the source supplied none |

A field not named in the mapping is unqualified: it means what it says.

Read it through the accessors rather than the dict:

| Call | Answers |
|---|---|
| `is_known(name)` | Does the field carry a value at all? |
| `provenance_of(name)` | How real is it, or `None` if unqualified |
| `is_approximate(name)` | Was it estimated rather than observed? |
| `is_usable_as_biology(name)` | May it be interpreted as a fact about the sample? |

`is_usable_as_biology` is the one that matters for correctness: it is `False`
for an absent field *and* for a synthesized one. Anything that annotates variant
effects, reannotates transcripts, or otherwise treats a value as biology must
check it and refuse rather than compute on a placeholder.

The point of all this is that **a consumer never branches on `source_type`**.
Every source produces the same shape; they differ only in which fields are
populated and how real those fields are:

```python
def rna_support(fragment):
    if not fragment.is_usable_as_biology("n_alt_reads"):
        return None
    return fragment.n_alt_reads
```

## source_type vocabulary (recommended, not enforced)

Free-form string. Topiary never interprets it; used for display and DSL filtering. Colon subtyping is convention.

| Category | Values |
|---|---|
| Variant, small | `variant:snv`, `variant:indel`, `variant:frameshift`, `variant:stop_gain`, `variant:stop_loss`, `variant:start_loss`, `variant:exon_loss`, `variant:alternate_start` |
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

Topiary computes these when you give it a reference proteome: `TopiaryPredictor(self_proteome=..., predict_self_nearest=True)` fills the similarity columns from `SelfProteome.nearest()` and then scores each `self_nearest_peptide` at its row's own allele, filling `self_nearest_value` / `_score` / `_percentile_rank`. That second half is the one a cross-reactivity judgement turns on — a near-identical self peptide the patient's MHC never presents is not the same risk as one it does.

The self peptide is scored **without flanking context**: it comes from the reference proteome, and `nearest()` reports its gene, transcript and offset but not the residues either side. Kinds that read flanks — antigen processing, and presentation where its model uses them — are therefore scored on the peptide alone; affinity and stability are unaffected.

Without `predict_self_nearest`, or for producers populating externally (via BLAST / edit distance against a healthy-tissue proteome with their own definition of "self"). The DSL scope just reads `self_nearest_*` columns. When columns are absent, `self_nearest.*` returns NaN.

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
from topiary import read_fragments, write_fragments

write_fragments(fragments, "fragments.tsv")
loaded = read_fragments("fragments.tsv")
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

- **Coordinate remapping for indel / frameshift `wt_peptide`** — `wt_peptide` is only populated when the baseline is the same length as the mutant sequence (substitution-compatible). Length-changing edits yield `None` until remapping lands.
- **Nearest-self compute** — the scope is reserved but no Topiary module produces the columns. Populate externally for now.
- **Format-specific loaders** (`read_isovar_fragments`, `read_exacto_fragments`) — each ~50-100 lines on top of the core abstraction; separate PRs. `read_lens` is already shipped (5.1.0); pVACseq is loaded via `read_pvacseq` (5.16.0) at the *row* level — its output is already at peptide × allele granularity, so the fragment-window abstraction doesn't apply.
