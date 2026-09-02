# Consumer guide

What topiary offers a downstream consumer, as of **5.45.0**, and what changed
across 5.28.2–5.45.0.

Written for vaxrank, but nothing here is vaxrank-specific.

---

## The shortest useful version

```python
from topiary import read_lens, evaluate_scores, resolve_default_methods
from topiary.ranking import parse

df = read_lens(path).to_long().df

scores = evaluate_scores(
    df,
    parse("affinity.value.logistic_normalized(350, 150) * (n_rna_alt > 5)"),
    default_methods=resolve_default_methods(df),
)
```

Every reader produces the same column vocabulary, so that expression runs
unchanged against LENS, pVACseq, or a predictor's own output.

---

## RNA and DNA evidence

Columns come in three layers. **Filter against layer 1.**

1. **Canonical cross-source** — same meaning from every reader.
2. **Canonical unit-specific** — `n_rna_alt_reads` / `n_rna_alt_fragments`, present
   only where a source reports both units and they differ.
3. **Source-prefixed originals** — `lens_vaf`, `pvacseq_tumor_dna_vaf`:
   exactly the number the tool printed, never reinterpreted. Find them with
   `source_columns(df)`, or `source_columns(df, "lens")` for one tool.

**The same vocabulary across readers, not the same columns** — a reader emits
one only where its source can answer:

| Column | Meaning |
|---|---|
| `n_rna_alt` | Evidence supporting the variant allele |
| `n_rna_ref` | Evidence supporting the reference allele |
| `n_rna_other` | Evidence supporting neither — a third allele, error, or nearby indel |
| `n_rna_overlapping` | Evidence covering the position (total coverage) |
| `rna_vaf` | Variant allele fraction |
| `rna_evidence_subject` | `"reads"` or `"fragments"` — what those counts count |
| `rna_evidence_method` | How they were obtained |
| `rna_alt_expression` | Abundance attributed to the variant allele |
| `rna_alt_expression_method` | How that was obtained |
| `gene_expression` | Gene-level abundance |
| `transcript_expression` | Transcript-level abundance, where separately stated |
| `sequence_source` | How the protein sequence came to exist |

**DNA support is the same shape**, so a filter written against RNA depth
becomes the DNA one by changing three letters: `n_dna_alt`, `n_dna_ref`,
`n_dna_other`, `n_dna_overlapping`, `dna_vaf`, `dna_evidence_subject`,
`dna_evidence_method`. There is no DNA expression pair — abundance is a
transcript property.

`n_rna_other` / `n_dna_other` are **absent unless the source counted the
reference independently.** Where topiary derived `ref` as `depth - alt`, that
figure already absorbs the other alleles, so reporting a difference of zero
would assert a clean locus nobody checked.

**A column is omitted, not nulled, where the source cannot answer it.** A
pVACseq aggregated report states a DNA VAF but no DNA depth, so it gets
`dna_vaf` and no `n_dna_*` at all — a column full of nulls would make
`available_evidence_columns()` report a capability the source lacks.

This holds on every path as of 5.48.0. LENS emits no `rna_alt_expression`
(it has no DNA VAF to scale abundance by, and its `vaf` never names its
assay), and a prediction run where nobody supplied expression emits no
`gene_expression`. **This changes filter behaviour:** `gene_expression > 1`
against such a frame now raises instead of evaluating to NaN and silently
emptying your candidate list. Check with `available_evidence_columns(df)`
before naming a column in a config that runs against more than one source.

**Check which are present before naming one in a config that has to run against
more than one source:**

```python
from topiary import available_evidence_columns, EVIDENCE_COLUMNS

missing = set(EVIDENCE_COLUMNS) - set(available_evidence_columns(df))
```

Naming an absent column in an expression **raises** rather than evaluating to
NaN:

```
transcript_expression > 1  on a pVACseq aggregated frame
  ValueError: Column 'transcript_expression' not found in DataFrame.
              Did you mean: ['gene_expression', 'rna_alt_expression', ...]
```

That is the intended behaviour — a silent NaN would drop every row in a filter
and say nothing. The alternative, emitting all-null columns everywhere, is
worse: a column that is present and empty asserts the question was asked and
answered as nothing. Absent beats substituted here as everywhere else.

Concretely, a pVACseq *aggregated* report has gene-level `RNA Expr` but no
separately stated transcript-level abundance, so it has no
`transcript_expression`; a LENS file without a `tpm` column has no
`gene_expression` either.

**`n_rna_alt`, `rna_evidence_subject` and `rna_evidence_method` are present
wherever a source can determine variant support**, which is what makes a
threshold written against `n_rna_alt` portable — whatever unit that source
counts in. Coverage without a usable fraction can still populate
`n_rna_overlapping` and its subject while correctly omitting the unavailable
alt count and derivation.

**Read `rna_evidence_subject` before a number leaves the run.** Within one run
the unit is consistent and rankings are unaffected. It matters for things that
travel — a documented threshold, a config copied between projects, a number in
a paper. Five fragments and five reads are different bars.

### Where each number came from

`rna_evidence_method` is one of:

| Method | Meaning | Provenance |
|---|---|---|
| `rna_alignment` | Counted from an RNA alignment | `measured` |
| `rna_depth_x_vaf` | RNA depth × RNA VAF, rounded | `approximated` |
| `rna_depth_x_source_vaf` | RNA depth × a VAF whose assay the source did not state | `approximated` |
| `cds_overlap_reads` | Counted, but of reads overlapping the peptide's CDS | `approximated` |
| `tpm_x_dna_vaf` | Transcript abundance × DNA VAF | `approximated` |
| `source_reported` | The source supplied the number without saying how | `approximated` |

`topiary.provenance_for_method(method)` maps a method to its provenance, so
there is one definition of what counts as measured. It does not: only a direct
count does.

`describe_read_evidence(df)` summarises a whole frame without walking rows.

### Per source

| | `n_rna_alt` | subject | method | DNA columns |
|---|---|---|---|---|
| isovar | 30 | `fragments` | `rna_alignment` | none — isovar reads RNA |
| pVACseq (all_epitopes) | 429 | `reads` | `rna_depth_x_vaf` | yes, from DNA depth × DNA VAF |
| pVACseq (aggregated) | from `RNA VAF` | `reads` | `rna_depth_x_vaf` | `dna_vaf` only — no DNA depth stated |
| LENS | from `lens_vaf` | `reads` | `rna_depth_x_source_vaf` | none — see below |

**LENS's `vaf` carries no assay qualifier**, and lands as `lens_vaf`. LENS names
its read columns `rna_*` explicitly and leaves `vaf` bare, so topiary cannot tell
whether the fraction is from RNA or DNA. It is used to split the RNA depth —
under a method that says so — and *not* to scale expression or to populate any
`n_dna_*` column, either of which would assert an assay nobody stated.

### Multiple samples

Evidence is per row, and `sample_name` is a first-class column and a DSL group
key, so a stacked multi-sample frame already carries each sample's own counts
and each stays attributable:

```python
merged = stack_results([per_sample_result(name) for name in samples])
merged.df[["sample_name", "n_rna_alt", "n_rna_overlapping", "rna_vaf"]]
```

**There is no built-in cross-sample aggregate yet** — summing counts across
samples is currently a `groupby` you write yourself. Tracked in
[#247](https://github.com/openvax/topiary/issues/247).

---

## Fragments

Four sources, one shape:

```python
fragments_from_variants(variants, alignment_file=bam)   # assembled from RNA
fragments_from_variants(variants)                        # translated from reference
fragments_from_effects(effects, padding_around_mutation) # reference arm alone
fragments_from_dataframe(read_lens(path).df)             # from a reader frame
fragment_from_isovar_result(result)                      # an IsovarResult you hold
```

`SEMANTIC_CORE` names the fields every source speaks to. They differ only in
which they can fill, so one consumer function reads all of them:

```python
def support(fragment):
    return fragment.n_rna_alt          # None where the source has no RNA
```

**isovar is optional in the strong sense** — not in `requirements.txt`, and
`import topiary` does not import it. Only `fragments_from_variants` with an
`alignment_file` needs it.

### Reads and fragments

isovar reports both units. Both are carried, each under a name that says what it
holds:

```python
fragment.n_rna_alt_reads        # 58
fragment.n_rna_alt_fragments    # 30
fragment.n_rna_alt          # 30 — fragments preferred
fragment.rna_evidence_subject()      # "fragments"
```

Fragments are preferred where a source has them: a paired-end fragment is one
molecule read twice, so it is one piece of evidence and two reads. Reads are
used where that is all there is. There is no conversion between them — that
needs library information no source carries.

### Knowing which fields are real

```python
fragment.is_known("n_rna_alt_reads")             # populated at all?
fragment.provenance_of("n_rna_alt_reads")        # measured | approximated | synthesized
fragment.is_approximate("n_rna_alt_reads")
fragment.is_usable_as_biology("n_rna_alt_reads") # False for absent *and* synthesized
```

`None` means the source could not answer. It is not zero, and the distinction
survives a TSV round trip. `is_usable_as_biology` is the one that matters for
correctness: a placeholder the loader invented has a value that means nothing,
and anything doing variant-effect annotation on it must refuse rather than
compute.

---

## The ranking DSL

### Resolving ambiguity

A frame with two models producing one kind, or one model at two versions, raises
rather than guessing. Both have a configured answer:

```python
evaluate_scores(df, node,
                default_methods=resolve_default_methods(df),
                default_versions=resolve_default_versions(df))
```

`describe_default_versions(df)` returns the candidates each choice was made
between, so a run can report "netMHCpan at 4.1b and 4.2, scoring with 4.2"
without re-deriving anything. `validate_default_*` catches an entry naming a
model or version that never ran.

Version ordering is PEP 440 — `4.10` beats `4.9` — and a version that is not
PEP 440 sorts before everything that parses, so "newest" means a real release.

### Alleles

`alleles=` takes a sequence, a mapping, or a callable:

```python
EvalContext(df, alleles=["HLA-A*02:01", "HLA-B*07:02"])   # every peptide
EvalContext(df, alleles={"SIINFEKLA": ["HLA-A*02:01"]})    # per peptide
EvalContext(df, alleles=lambda keys: genotype_for(keys["peptide"]))
```

**Use a per-peptide form when peptides were not each reported against the whole
genotype** — a reader emitting one row per (peptide, allele) passing its own
threshold produces exactly that. Declaring the union invents a group for every
pairing that was never scored, and an expression reading only peptide-level
evidence gives each invented group a real number.

### Peptide-level evidence

A row whose kind is MHC-independent broadcasts across a peptide's allele groups
— **unless the row names an allele**, in which case it lands there and nowhere
else. That is the mechanism for allele attribution: writing a row onto chosen
alleles is how a policy says "credit this evidence here".

`KIND_MHC_DEPENDENCE` and `mhc_dependence()` say which kinds are peptide-level.
There are five, not one.

### Sharing a context

```python
ctx = EvalContext(df, group_keys=gk)
a = evaluate_scores(df, node_a, context=ctx)
b = evaluate_scores(df, node_b, context=ctx)
```

`apply_filter` and `apply_sort` return *new* frames, so a context cannot be
threaded down a pipeline — reuse applies to several operations on one unchanged
frame. Passing a stale one raises.

---

## Shared helpers, so you do not reimplement them

```python
is_stated(value)        # did the source say anything here?
stated_values(series)   # the same rule over a column
is_named_version(value) # the same rule, named for predictor_version
```

`None`, `NaN`, blank, and the literal strings `"nan"` / `"None"` / `"<NA>"` all
mean *not stated*. The obvious version of this test — `if str(v).strip()` —
excludes only the blank spellings, since `str(None)` is `"None"` and
`str(float("nan"))` is `"nan"`, both truthy. That mistake shipped in topiary and
independently in a consumer that had reimplemented it.

Also public rather than reimplemented: `fragment_from_effect`,
`resolve_default_methods`, `CANONICAL_METHOD_PREFERENCE`, `KIND_MHC_DEPENDENCE`,
`derive_mhc_class`.

---

## Migrating renamed columns

Every column renamed since 5.46.0 is in `RENAMED_COLUMNS`, and
`renamed_column(name)` looks one up:

```python
from topiary import RENAMED_COLUMNS, renamed_column

renamed_column("vaf")              # 'lens_vaf'
renamed_column("tumor_rna_depth")  # 'pvacseq_tumor_rna_depth'
renamed_column("gene_expression")  # None — not renamed
```

The DSL consults it before fuzzy matching, so a stale expression says what
to do:

```
Column 'vaf' not found in DataFrame. It was renamed to 'lens_vaf'.
```

**Do not fuzzy-match these yourself.** `vaf` is the trap: the closest
surviving name is `rna_vaf`, and that is the wrong answer — `rna_vaf` is the
canonical cross-source fraction, while `vaf` became `lens_vaf`, LENS's own
fraction whose assay the file never states.

**If you read reader-frame columns with `row.get(...)` or `df[...]` rather than
through the DSL, topiary cannot warn you** — a missed `.get()` returns `None`
and becomes a silent zero. Check your column names against `RENAMED_COLUMNS`
once at startup. Serialized `ProteinFragment` JSON and TSV are the exception:
their old unit-specific evidence names are migrated on load, including
`field_provenance` keys.

Reader frames have no compatibility aliases. Two output columns for one
quantity is the ambiguity the renames existed to remove. `ProteinFragment`
accepts its old unit-specific names when loading, constructing, and reading
attributes so the 5.x API remains compatible, but serialization emits only the
new names.

---

## Reader escape hatches

```python
read_lens(path, binding_metrics={("newtool", "ic50_nm"): ("affinity", "value")})
```

Merged over the built-in table, keyed on `(tool, metric)` — the pair the
unmapped-column warning prints, and version-free so one entry covers a tool
however a file spells its release. `None` as a value declares a column a
non-prediction, silencing the warning without remapping it.

---

## What changed, 5.28.2 → 5.45.0

Floors worth knowing:

| Need | Floor |
|---|---|
| Multi-version LENS tables keep both versions | 5.28.2 |
| `default_methods` on the predictor | 5.29.0 |
| `read_lens(binding_metrics=...)` | 5.30.0 |
| `default_versions` | 5.31.0 |
| `is_stated` / `is_named_version` | 5.35.0 |
| `allele_set` in the cache key | 5.36.0 |
| RNA evidence from the readers | 5.37.0 |
| Per-peptide `alleles=` | 5.33.0 |
| Named-allele rows not broadcast | 5.39.0 |
| `fragments_from_variants` (isovar) | 5.40.0 |
| RNA columns scoped and cut to nine | 5.45.0 |
| DNA evidence columns, mirroring the RNA ones | 5.47.0 |
| `n_rna_other` / `n_dna_other` for third-allele support | 5.47.0 |
| `rna_vaf` / `dna_vaf` canonical fractions | 5.47.0 |
| Source prefixes: `vaf` → `lens_vaf`, `tumor_dna_vaf` → `pvacseq_tumor_dna_vaf` | 5.47.0 |
| `PREDICTION_KEY_COLUMNS` public | 5.47.0 |
| `dna_evidence_subject` derived, not asserted | 5.48.0 |
| Unit columns assay-scoped: `n_alt_reads` → `n_rna_alt_reads` | 5.48.0 |
| All-null evidence columns omitted on every path | 5.48.0 |
| `topiary.rna_evidence` module renamed to `topiary.evidence` | 5.47.0 |

### Removed in 5.45.0, with no compatibility shim

`count_in`, `read_count_subject`, `count_column_for_subject`,
`subject_for_method`, the per-subject column renaming from 5.44.0, and the
supporting-count columns on reader frames.

All of it existed to work around one mistake: topiary carried isovar's
*fragment* counts in fields then named `n_alt_reads`, then built an API to
explain why reads were unavailable. isovar exposes `num_alt_reads` beside
`num_alt_fragments`; they were never unavailable. Carrying both under honest
names left nothing for those five to do.

### Caches

A `CachedPredictor` store written before 5.36.0 loads fine — `_normalize` runs
on every construction and repairs split allele buckets. What does not repair
itself is a store whose split buckets held *different* values for what is now
one key; that raises on load rather than silently answering, which is intended.

**Both doors agree about a repeated key as of 5.48.0** (topiary#231). An exact
duplicate is stored once; a key appearing twice is an error only when the rows disagree on a
`PREDICTION_VALUE_COLUMNS` entry:

| | before | now |
|---|---|---|
| shards sharing an identical row | `concat` raised, constructor accepted | both accept |
| rows differing only in `sample_name` | `concat` raised, constructor accepted | both accept |
| same key, different `affinity` | `concat` raised, **constructor accepted silently** | both raise |

The last row was the real hazard: a lookup returned whichever row came last.
`conflicting_predictions(df)` is the single check both use, and returns the
offending rows so you can see them.
`allele_set` joined the cache key in 5.36.0, so two genotypes deconvolving to
the same best allele are two entries rather than one silently picked.
