# Changelog

## 5.51.0

**One name for the frame nodes group: `EvalContext.df`.** 5.50.0 left `df` and
`evaluation_df` returning the identical object, which is one public name too
many for a context that already exposes three frame-shaped attributes.
`evaluation_df` is gone; read `ctx.df`.

`evaluation_df` was added in 5.49.1 and never documented or exported, so it is
removed outright rather than kept as an alias. Anything reading it becomes
`ctx.df` with no other change — same object, same contents.

The DSL docs now show how to write your own node, including the rule this
pair of releases existed to fix: group `ctx.df`, not the frame you handed to
`EvalContext`, and reindex onto `ctx.group_index`.

## 5.50.0

**`EvalContext.df` now returns the frame nodes are grouped against.** Custom
DSL nodes are documented to group `ctx.df` by `ctx.group_keys` and reindex onto
`ctx.group_index`. That only holds if the frame carries the same identity keys
the group index was built from, and it did not: `ctx.df` handed back the
caller's raw frame, where a missing allele spelled `"nan"` is a different
groupby key than one spelled `None`. A node outside Topiary keyed its results
to groups that do not exist and its values silently became `NaN`, or — when a
text spelling came first — the wrong row won. Built-in nodes were unaffected;
they already read the normalized frame.

`ctx.df` now returns that same normalized frame, so built-in and third-party
nodes see one set of keys. The caller's DataFrame is still never mutated:
normalization lands on a copy, and only when a spelling actually needs it.

Reading `ctx.df` is unchanged for everything that treats it as the context's
prediction rows. What does change is identity: `ctx.df` is now a different
object from the frame you passed whenever a key needed normalizing, so
`ctx.df is my_frame` is no longer the way to ask whether a context belongs to
a frame. New `ctx.is_built_on(my_frame)` answers that — the same check
`apply_filter`, `apply_sort` and `evaluate_scores` make before accepting a
`context=`.

## 5.49.1

**Hardened cross-sample evidence aggregation after review.** Equivalent
missing identity spellings now collapse through the same `EvalContext`
normalization used by filtering, sorting, and scoring, including pandas string
and categorical columns. Canonical counts are validated before duplicate rows
are collapsed, so booleans and malformed values cannot disappear based on row
order or an incomplete sample. Exact large numeric values and Arrow-backed
columns are supported up to the canonical nullable-Int64 limit; larger
individual or pooled counts raise a documented validation error.

## 5.49.0

**Added strict cross-sample aggregation for canonical evidence.**
`aggregate_evidence_across_samples` returns a separate pooled DataFrame while
leaving each sample's evidence intact. Repeated prediction rows count once,
complete canonical counts sum, and RNA/DNA VAFs are recomputed from pooled
alternate and overlapping counts rather than averaged.

Partial measurements stay absent instead of becoming zero. Samples must name
the same evidence subject before counts can be combined. Allele-support counts
must also share a derivation method, so measured counts cannot be flattened
together with estimates; coverage-only depths need no invented method.
Expression remains per-sample, and the pooled result reports `n_samples` for
each represented candidate identity.

The test runner now probes pytest plugins and launches pytest with the same
Python interpreter, avoiding false xdist detection across environments.

## 5.48.1

**Fixed lazy mhctools model-name discovery.** String model names now resolve
through mhctools' public predictor registry, so lazily exported predictors such
as `mhcflurry` work in a fresh process without first being imported by some
unrelated caller. The regression test no longer skips when MHCflurry has not
already been loaded, and the frameshift overlap grid now parameterizes only
cases that contain a peptide window rather than reporting inapplicable cases as
skips.

## 5.48.0

**Made evidence units and absence explicit across every path.** RNA
unit-specific fields now name their assay (`n_rna_alt_reads`,
`n_rna_alt_fragments`, and the ref / other / overlapping / supporting
families). Reader and prediction frames emit a canonical evidence column only
when at least one row states a value. A row that would mix reads and fragments
under one `rna_evidence_subject` is refused rather than mislabeled, and a DNA
read depth cannot be called fragments.

Old `ProteinFragment` names remain accepted throughout the 5.x API: direct
construction, attribute access, JSON, TSV, and `field_provenance`. New output
uses only the assay-scoped names. Every dataclass field now survives JSON and
TSV round-trips, and other-allele counts survive the isovar, DataFrame, and
prediction adapters.

**Made the cache's constructor and concat paths conform.** Exact duplicate
predictions are stored once, context-only differences remain, contradictory
values raise through either door, and malformed numeric strings are rejected
before coercion can hide them as null.

**Corrected pVACseq aggregated expression semantics.** pVACseq defines
`RNA Expr` as gene-level expression, so it now maps to `gene_expression`
rather than inventing transcript resolution. A missing source `Allele Expr` is
left absent instead of being reconstructed with DNA VAF; pVACseq defines its
reported quantity using RNA VAF.

`RENAMED_COLUMNS`, `renamed_column`, the three cache-column classifications,
and `conflicting_predictions` are public so consumers can share Topiary's
decisions rather than reimplementing them.

## 5.46.0

**Fixed: the consumer guide overpromised.** It said the nine evidence
columns were "identical across readers". They are the same
*vocabulary*, not the same *columns* — a reader emits one only where its
source can answer, and a pVACseq **aggregated** report has no gene-level
abundance, so no `gene_expression`.

That mattered for the exact use the guide recommends: naming an absent
column in an expression raises rather than evaluating to NaN, so a
config written against a LENS frame would break on an aggregated
pVACseq one. Corrected, with the failure shown.

Raising stays the behaviour, and the columns stay absent rather than
all-null. A present-but-empty column asserts the question was asked and
answered as nothing; absent beats substituted here as everywhere else.

**Added: `available_evidence_columns(df)` and `EVIDENCE_COLUMNS`**, so a
consumer writing a portable config can check rather than discover the
gap at runtime.

**Fixed: the guide was not in the distribution.** No `docs/` in the
sdist and no `.md` in the wheel, so anyone working from an installed
package could not find the document framed as "the thing you read
instead of asking". `MANIFEST.in` now ships `docs/`.

Both found by the downstream consumer checking the guide's claims
against the shipped package instead of reading it — which is what the
guide asks its readers to do, so it is fitting that it is how the guide
was found wrong. Every *code example* in it had been executed; the
sentence summarising them had not.

## 5.45.1

**Added: `docs/consumer-guide.md`** — what topiary offers a downstream
consumer as of 5.45.0, and what changed across the 5.28.2–5.45.0 series.

Written because a consumer adopting this stretch had to reconstruct it
from twenty-three changelog entries and a dozen cross-session messages.
It covers the nine RNA-evidence columns and what each method means, the
four paths to a `ProteinFragment` and the one shape they share, the DSL's
ambiguity resolution and per-peptide alleles, the shared helpers that
exist so consumers stop reimplementing them, and a floor-by-feature
table. Every code example in it was executed against the shipped
package.

## 5.45.0

**Replaced the read/fragment API. 5.43.0 and 5.44.0 were built on a
false premise and this supersedes both.**

isovar exposes `num_alt_reads` **and** `num_alt_fragments` — reads,
fragments, ref, total, supporting, both units for every count. topiary
carried only the fragment counts, stored them in fields named
`n_alt_reads`, and then added an API to explain that reads were
unavailable. They were never unavailable.

```python
# before — a fragment count in a field named for reads, and this:
fragment.count_in("n_alt_reads", "reads")   # None
```

Asking a field named `n_alt_reads` for reads and being told `None` was
not a subtle contract, it was a wrong one.

**Now both units are carried under names that say what they hold:**

| Reads | Fragments |
|---|---|
| `n_alt_reads` | `n_alt_fragments` |
| `n_ref_reads` | `n_ref_fragments` |
| `n_overlapping_reads` | `n_overlapping_fragments` |
| `n_alt_reads_supporting_protein_sequence` | `n_alt_fragments_supporting_protein_sequence` |

**And one accessor per quantity takes the better of the two:**

```python
fragment.n_rna_alt                  # 30  — fragments, since isovar has them
fragment.n_rna_ref
fragment.n_rna_overlapping
fragment.n_rna_supporting_protein_sequence
fragment.rna_evidence_subject()     # "fragments" | "reads" | None
```

Fragments are preferred where a source reports them, because a
paired-end fragment is one molecule read twice — one piece of evidence
and two reads. Reads are used where that is all there is. Frames carry
the same `n_rna_*` columns plus `rna_evidence_subject`, so **one
threshold spans every source** and a number that travels can still name
its unit.

**Every generated column is now scoped by assay.**
`variant_allele_expression` is `rna_alt_expression` (with
`rna_alt_expression_method`) — it is an RNA quantity and the name did
not say so. The counts and the two describing columns were already
`n_rna_*` / `rna_evidence_*`.

**Fixed: LENS's unqualified `vaf` was used as both an RNA and a DNA
fraction.** LENS carries one `vaf` column while naming its read columns
`rna_*` explicitly, so the fraction's assay is unstated — and topiary
was using it *both* to split the RNA depth *and* as a DNA VAF to scale
expression. One of those was necessarily wrong.

It now splits the depth under a method that says what actually happened,
`rna_depth_x_source_vaf`, rather than `rna_depth_x_vaf` asserting an
assay nobody stated. It is no longer used to scale expression, so LENS
frames carry no `rna_alt_expression` — they carried none anyway, since
the estimate was empty on every row. pVACseq's VAFs *are* qualified
(`Tumor RNA VAF`, `Tumor DNA VAF`), so it keeps `rna_depth_x_vaf` and
`tpm_x_dna_vaf`.

**Simplified the reader-observable columns.** A LENS frame carried 12
topiary-generated evidence columns and four of them were exact
duplicates: `n_rna_alt` equalled `n_alt_reads` on every reader, since no
reader produces fragments. Both readers now expose the same nine:

```
n_rna_alt   n_rna_ref   n_rna_overlapping
rna_evidence_subject    rna_evidence_method
variant_allele_expression   variant_allele_expression_method
sequence_source   gene_expression
```

One name per quantity, one column saying the unit, one saying the
origin. The unit-specific `n_*_reads` / `n_*_fragments` fields stay on
`ProteinFragment`, where a caller who needs a specific unit names it —
on a frame they duplicated `n_rna_*` for every single-unit source.

`read_count_method` is `rna_evidence_method`, matching
`rna_evidence_subject`.

**Dropped `n_alt_reads_supporting_protein_sequence` and
`supporting_read_count_method` from reader frames.** LENS counts reads
overlapping the peptide's *CDS*, which is not a count of reads
supporting the assembled protein sequence — emitting it under that name
overstated what the reader has. It passes through under LENS's own
column instead. The assembled count stays on `ProteinFragment`, where
isovar makes it real.

**Renamed: `rna_reads` → `rna_alignment`** (old name kept as an alias).
It is a *method*, naming where a number came from, and it explicitly
fixes no unit — an aligner counts reads and fragments alike. Calling it
`rna_reads` implied the unit it refused to fix, which only became
visibly wrong once the value it labels could be fragments.

**Where every number comes from, alongside what it counts:**

| Source | `n_alt_reads` | `n_alt_fragments` | `n_rna_alt` | method |
|---|---|---|---|---|
| isovar | 58 | 30 | 30 (fragments) | `rna_alignment` |
| pVACseq | 429 | — | 429 (reads) | `rna_depth_x_vaf` |
| LENS | from `vaf` | — | reads | `rna_depth_x_vaf` |

Where a source gives only depth and a variant allele fraction, the alt
count is depth × VAF — computed, and labelled as computed rather than
passed off as counted.

Removed: `count_in`, `read_count_subject` (the fragment method and the
frame column), `count_column_for_subject`, `subject_for_method`, and
5.44.0's renaming of count columns by subject. All of it existed to
work around carrying one unit under the other's name.

## 5.44.0

**Fixed: read counts did not survive fragment → prediction frame.** The
frame carried `read_count_method` and `read_count_subject` — which
arrive as annotations — describing a count that was not there. So it
said how a number was obtained and what it counted, while omitting the
number. The four count fields now propagate the way `gene_expression`
already did.

**Fixed: a threshold written for reads was answered by a fragment
count.** Once the counts reached the frame, `n_alt_reads > 5` on an
isovar-derived frame was satisfied by 8 *fragments*. Both are integers
and both are plausible, so nothing failed.

Count columns are now named for what they count. A fragment-subject
frame has `n_alt_fragments`; a read-subject frame keeps `n_alt_reads`:

```
n_alt_reads > 5        ValueError: Column 'n_alt_reads' not found.
                       Did you mean: ['n_alt_fragments', ...]
n_alt_fragments > 5    works
```

Naming rather than only tagging is what makes the wrong reference fail.
5.43.0 put the subject on the fragment and on the frame; a threshold is
written against a *column name*, and the DSL was never going to consult
a sibling column before answering.

Reader frames are unchanged — both state `reads`, so both keep the
`n_*_reads` spelling. A source that states no subject keeps it too,
since that is what a source which does not say counts.

**Cost, stated plainly:** on any other path this would break every
existing config. It is free here only because the isovar path shipped
days ago and nothing depends on it yet — which is the argument for doing
it now rather than later.

## 5.43.0

**Added: a read count now names what it counts.** isovar counts
*fragments* — `num_alt_fragments`, not `num_alt_reads`. A depth × VAF
estimate is inherently about *reads*, because depth is a read depth.
Both landed in `n_alt_reads`, so the field was honest about **how** a
number was obtained and silent about **what it counts** — the same shape
as the CDS-overlap column, a real count of an adjacent thing.

```python
fragment.read_count_subject()                    # "fragments" or "reads"
fragment.count_in("n_alt_reads", FRAGMENTS)      # 30
fragment.count_in("n_alt_reads", READS)          # None — says so
```

Frames carry `read_count_subject` beside `read_count_method`.

**Why it matters, and where it does not.** Within one run the unit is
internally consistent, so a ranking does not change. The harm is
entirely in what travels: a documented `n_alt_reads > 5`, a config
copied between projects, a number in a paper. Five fragments and five
reads are different bars, and nothing said which was cleared.

Fragments are the right subject for a `sqrt()` confidence weight: two
mates of one fragment are not independent evidence, so read counts
overstate paired-end support roughly twofold relative to single-end —
precisely the distortion a diminishing-returns transform should not
inherit.

**What this deliberately does not attempt.** Perfect cross-path
comparability is unattainable: converting a read estimate to fragments
needs library information no source carries. So there is no conversion
and no single comparable number. Every path names its subject, a score
names the subject it wants, and a source asked for the other one
**returns `None` rather than substituting** — the derivation rule one
level up.

`rna_reads` deliberately fixes no subject: it says a count came from an
alignment, not whether the aligner counted reads or fragments, so the
producer states it.

Design settled with the downstream consumer, whose framing this is.

## 5.42.0

**Fixed: `read_pvacseq` spoke two vocabularies depending on which
flavour of its own format it was given (#238).** The aggregated report
supplies pVACseq's own `Allele Expr` and `RNA Expr`, which were passed
through as `allele_expression` and `rna_transcript_expression` — names
the all_epitopes path never emits. **And that branch never called
`attach_read_evidence` at all**, so it had no method columns whatsoever:

```
all_epitopes   variant_allele_expression + 3 method columns
aggregated     allele_expression, rna_transcript_expression, no methods
```

So no single filter worked across two pVACseq files. Both branches now
emit the same columns.

**Added: `SOURCE_REPORTED`.** When pVACseq supplies the estimate itself,
neither answer was honest — passing it through unlabelled claims a
derivation nobody can check, and recomputing it discards the number the
source stands behind. `variant_allele_expression_method` is
`source_reported` on the aggregated path and `tpm_x_dna_vaf` where
topiary derived it. It maps to `approximated`, not `measured`: the
source stands behind the number but did not say how it got there.

This was the sharper half of #238 and I closed the issue without it,
having verified on the all_epitopes fixture and concluded about the
reader. The consumer who reported it made the mirror-image error,
grepping for `express` against headers abbreviated `Expr`. **A function
with two branches needs both exercised**; the tests are parameterized
over both flavours now.

## 5.41.0

**Fixed: gene-level expression had two names (#238).** `read_lens`
called it `gene_tpm`; `read_pvacseq` called it `gene_expression`. A
filter naming either matched nothing on the other frame rather than
failing, so a consumer had to know which reader produced the frame —
the thing a shared vocabulary exists to avoid.

LENS frames now also carry `gene_expression`, the name topiary already
uses for gene-level abundance in `ProteinFragment.gene_expression` and
in `read_pvacseq`. `gene_tpm` stays as the LENS-native spelling, and
`gene_tpm_raw` still holds the original string, since LENS writes fusion
rows as composites and the numeric column is NaN for them.

The other two problems in #238 were already fixed by 5.37.0, before the
issue was read: both readers emit `variant_allele_expression` (there is
no `allele_expression` on either frame, so no one quantity under two
names), and both label every derivation — on the expression axis and the
read axis. They are pinned by tests now rather than left to chance.

## 5.40.0

**Added: `fragments_from_variants` — isovar, actually run (#102).**
5.38.0 could adapt an `IsovarResult` a caller already had. This runs
isovar to produce them, so topiary can build the surrounding protein
context for a mutation from RNA rather than only consume someone else's:

```python
fragments = fragments_from_variants(variants, alignment_file=bam)  # assembled
fragments = fragments_from_variants(variants)                      # translated
```

**The two arms are interchangeable.** Both return `ProteinFragment`s
with the same core, so a pipeline does not change shape when the RNA
does or does not exist. What differs is what the fragments can tell
you — an assembled sequence carries the patient's other variants and
whatever phasing the reads support, plus counted read support; a
translated one carries the reference everywhere except the variant, and
no counts. `annotations["sequence_source"]` says which, so an RNA-backed
candidate and an inferred one never blend.

`protein_sequence_length` is a *sequence* length, not a peptide length:
a fragment is scanned by a sliding window downstream, so the assembled
context must contain every peptide that could cover the mutation.
Default 25. `padding_around_mutation` defaults to half of it so the
reference arm produces a comparable window.

`allow_reference_fallback=True` translates variants isovar could not
support instead of dropping them.

`filter_thresholds` now actually filters. **isovar records filter
outcomes and never drops anything**, so a caller's thresholds — and
isovar's own defaults — annotated results that then flowed on as
RNA-backed evidence. `require_passing_filters=True` drops them; pass
`False` for the old behaviour.

Assembly is turned **on**. isovar defaults `variant_sequence_assembly`
to off, which requires a single read or fragment to span the whole
window — so a longer context yields *fewer* variants rather than longer
sequences, and "carrying the phasing the reads support" would not be
true of the result. The default window is isovar's 21 rather than a
raised 25, for the same reason.

`fragments_from_effects` is public: the reference arm on its own, for a
caller with variants and no alignment file. It filters silent and
non-coding effects first (several varcode classes expose a
`mutant_protein_sequence` while leaving the amino-acid offsets `None`,
and `fragment_from_effect` raises on those — one such effect anywhere in
a batch discarded every fragment already built), and it uses
expression-aware transcript selection when transcript expression is
given, so the same variants pick the same transcript whichever entry
point built them.

Conflicting or inapplicable arguments are refused rather than silently
resolved: `protein_sequence_length` together with a
`protein_sequence_creator` (the creator's length used to win silently),
isovar-only arguments with no `alignment_file` (they were swallowed), a
non-positive window, and a `padding_around_mutation` too small to
contain an epitope — the last validated by the existing
`check_padding_around_mutation` rather than a fresh derivation.

isovar is needed **only** when `alignment_file` is given.
## 5.39.1

**Added: `tests/test_consumer_workflows.py`** — documented workflows
exercised end to end, and a standing rule in AGENTS.md to keep them
there.

This exists because of a specific failure. Asked whether a downstream
consumer was blocked, I checked that the four capabilities their design
needed were exported and said they were unblocked. They were exported.
They did not *compose*: writing a peptide-level row onto an allele to
mean "credit this evidence here" was silently discarded, so every
attribution policy produced identical scores (fixed in 5.39.0 as #232).
Separately, I said the DSL could reference a LENS annotation named
`tpm`, having read that the reader passes annotations through without
running an expression — the column is `gene_tpm`, since `tpm` gets
special handling for fusion rows and the raw string is kept in
`gene_tpm_raw`.

Checking that parts exist is not checking that the whole works. The new
tests walk each workflow from input to answer: a LENS report filtered
and sorted by a DSL expression; annotations addressable under their
actual names; the resolve-then-evaluate loop and the raise that guards
it; **narrowing attribution changing the answer** rather than the
pieces merely existing; one consumer function reading isovar, LENS and
pVACseq fragments; and a shared context serving several operations plus
the guard that makes sharing safe.

No behavior change.

## 5.39.0

**Fixed: a peptide-level row that names an allele was projected onto
every other allele (#232).** The explicit allele was silently discarded:

```
row: antigen_processing, allele=HLA-A*02:01, score=0.8

  HLA-A*02:01    0.8
  HLA-B*07:02    0.8      <- the row said A*02:01
```

Peptide-level evidence usually carries no allele, and then there is
exactly one thing a reference to it can mean — this peptide's value, for
every allele — so topiary projects it. But a producer that writes such a
row *onto one allele* is saying something narrower, and projecting it
anyway credits a score to alleles the row explicitly did not name. That
is a value attributed to something that did not state it, the same
family as the stringified-null group keys one level up.

Now: a blank-allele row broadcasts as before; a row that names an allele
lands in that allele's group and nowhere else; and the two compose — the
named row claims its allele, a blank row fills the rest. "Names an
allele" uses the same `is_stated` rule as everything else, so a frame
that went through `astype(str)` does not stop broadcasting.

**`haplotype` is deliberately exempt.** mhctools stamps a genotype-level
score with the allele it deconvolved as the best presenter, so that
allele is an artifact of reporting rather than a restriction — treating
it as one would strand a joint score on a single allele, which is the
failure projection exists to prevent.

**Two peptide-level rows at different alleles are no longer a
conflict.** They are two answers to two questions. Two *blank-allele*
rows that disagree still raise, since that is a peptide contradicting
itself.

**The warning now describes what happened.** It said "which carries no
allele" in every case — which becomes false the moment a row names one,
and that is exactly the user whose scores just narrowed. Three messages
now, naming the action taken rather than the counterfactual avoided:
rows all naming alleles, rows mixed, and rows carrying none.

This unblocks allele attribution downstream (openvax/vaxrank#349):
writing a row onto chosen alleles is the natural way to say "credit this
evidence here", and it was being discarded — so every attribution policy
produced identical per-allele scores, and a narrowing knob could compute
an answer and then silently fail to apply it.

## 5.38.0

**Added: isovar → `ProteinFragment`, and every other path with it
(#102).** 5.37.0 put `isovar_assembly` in the vocabulary with nothing
emitting it — a name for a thing that did not exist, which is the same
shape as a configured knob that does nothing.

```python
fragment_from_isovar_result(result)      # one result
fragments_from_isovar_results(results)   # drops those with no RNA support
fragments_from_dataframe(read_lens(path).df)      # the table readers
fragment_from_effect(effect, padding)             # varcode, since 5.35.0
```

**isovar is the only source that counts.** It assembles a protein
sequence from reads and counts the ones supporting it, so its numbers are
`measured`. pVACseq derives the split from depth × VAF and LENS counts
reads overlapping the peptide's CDS — both `approximated`. One mapping,
`provenance_for_method`, decides which is which, so a frame and a
fragment cannot disagree about whether depth × VAF counts as measured.
It does not.

**Optional in the strong sense**: isovar is not imported at module
scope, not in `requirements.txt`, and `import topiary` does not import
it. A consumer that only reads LENS reports should not pay for a package
it never calls. There is a test asserting it stays unimported.

**Every path now reaches a fragment with the same core.** `SEMANTIC_CORE`
names the fields every source speaks to, whether or not it can fill
them, so this reads all four without knowing which it has:

```python
def rna_support(fragment):
    if not fragment.is_usable_as_biology("n_alt_reads"):
        return None
    return fragment.n_alt_reads, fragment.is_approximate("n_alt_reads")
```

isovar returns `(30, False)`, pVACseq a count with `True`, varcode
`None` — no branching on `source_type`, which is why `source_type` stays
biological.

**Fixed: the supporting-count derivation was accepted and not
recorded.** `attach_read_evidence` took `supporting_method` and dropped
it, so a frame carried LENS's count of 45 CDS-overlapping reads with no
way to say it was not 45 reads supporting the variant. It is now a
column, and the fragment reads it.

## 5.37.0

**Added: RNA read-level evidence from the readers, with each number
naming its derivation (#102).** 5.32.0 added the fields to
`ProteinFragment`; nothing populated them. `read_lens` and
`read_pvacseq` do now.

```
n_overlapping_reads   n_alt_reads   n_ref_reads   read_count_method
               1233           429           804     rna_depth_x_vaf
```

The derivation is named because "429 reads counted" and "429 reads
implied by depth × VAF" are different claims:

| Method | Meaning |
|---|---|
| `rna_reads` | Counted directly from an RNA alignment |
| `rna_depth_x_vaf` | depth × VAF, rounded — not counted |
| `cds_overlap_reads` | Counted, but of reads overlapping the peptide's CDS rather than supporting the variant |
| `tpm_x_dna_vaf` | Transcript abundance × DNA VAF — an expression proxy, not a read count |

Per source: **pVACseq** reports `Tumor RNA Depth` (a real count) and
`Tumor RNA VAF`, so the depth is counted and the split is arithmetic.
**LENS** counts reads covering the genomic origin *and* reads covering it
with the peptide's CDS — both real counts, but the second is of
something adjacent to what was asked for, which is why it is
`cds_overlap_reads` rather than `rna_reads`. A LENS row with no `vaf`
gets no split and no method, rather than a zero.

**Added: `variant_allele_expression`**, the bulk-RNA fallback —
transcript abundance × DNA VAF, for when there is no alignment to count
alt reads from. It assumes both alleles are transcribed equally, so a
variant on a transcriptionally silenced allele looks expressed — the
error being exactly what allele-specific counting exists to detect,
which is why it is labelled rather than mixed in with measured values.

**Added: `sequence_source`.** `source_type` says what an antigen *is*
(`"variant:snv"`) and deliberately says nothing about method, so a frame
could not answer "was this sequence assembled from RNA or translated
from the reference?" — the first question you ask auditing a ranking.
One of `isovar_assembly`, `varcode_translation`, `lens_pep_context`,
`pvacseq_epitope`, `caller_supplied`.

`describe_read_evidence(df)` summarizes how a run's numbers were
obtained without walking the rows. `split_reads_by_vaf` and
`attach_read_evidence` are public, so a caller with its own source uses
the same implementation rather than writing the arithmetic again.

`None` throughout means the source could not answer, which is not zero —
and `attach_read_evidence` refuses a supporting count whose derivation
is unnamed, since a count that cannot be told from a measurement is
worse than no count.

## 5.36.0

**Fixed: the genotype is now part of the cache key (#229).**
`CachedPredictor` keyed on
`(peptide, allele, peptide_length, kind, n_flank, c_flank)`. MHCflurry
presentation in haplotype mode scores a peptide against a whole genotype
and reports the *deconvolved best allele*, so two different genotypes
that deconvolve to the same best allele collided on every key column —
and the lookup silently returned one of them.

`_CACHE_COLUMNS` already said why this mattered:

> The genotype a haplotype-mode prediction was scored against; blank for
> per-allele rows. Without it a cached presentation row reads as a
> prediction for its deconvolved best allele.

`allele_set` joins the key on exactly the argument already made there for
`n_flank` / `c_flank`: the same peptide in a different context produces a
different score, so the context is part of the identity. Blank genotypes
coexist with populated ones the way absent flanks do, and every spelling
of "no genotype" — `None`, `NaN`, blank, `"nan"` — is one key rather
than four.

**Deliberately not in the key:** `source_sequence_name`, `peptide_offset`
and `sample_name`. Those say where a peptide was *found*, not what was
predicted about it, and a prediction for one
(peptide, allele, length, kind, flanks, genotype) is the same prediction
whichever protein or sample it came from. Keying on them would defeat
the cache — the same peptide would be re-predicted once per source
protein.

Caches on disk keep loading; the key gains a dimension, so an entry
written without a genotype keys as blank, which is what it is.

## 5.35.1

**Fixed: `CachedPredictor` turned a missing allele into the allele named
`"None"`** — the cache axis of the defect 5.35.0 centralized.

`_normalize` rejects null identity columns for `prediction_method_name`,
`predictor_version` and `kind`, with a comment saying why, then applied
`astype(str)` to `allele` and `peptide` without the same guard. The
reasoning was written down and not applied one field over.

Two consequences. The cache keys on
`(peptide, allele, peptide_length, kind)`, so `None`, `NaN` and `""`
were three different keys, and one predictor's allele-free evidence sat
in three buckets while still looking present in the store. And
`.alleles` — the mhctools surface answering "what can this predict for"
— reported the string `"None"` as a queryable allele.

An allele-free row is legitimate (`proteasome_cleavage` and the other
peptide-level kinds have no allele), so `allele` collapses every
spelling onto `""`, the way `n_flank` / `c_flank` already did, and
`.alleles` excludes allele-free rows. A missing `peptide` *is* rejected
— there is no such thing as a peptide-free prediction.

Found by following a downstream report of the identical shape in their
own store, where the same spellings split one predictor's evidence
across four buckets.

## 5.35.0

**Changed: `_fragment_from_effect` is now public as `fragment_from_effect`.**
It builds a :class:`ProteinFragment` from a varcode variant effect — the
varcode arm of the multi-source fragment story, and something a caller
doing its own variant annotation would otherwise reimplement. Now
exported, with the two non-obvious rules written down: the window is
clipped at the protein's first stop codon, and `reference_sequence` is
populated only when the pre- and post-mutation proteins align 1:1,
because slicing the same offsets out of a frameshifted protein would
present a different piece of protein as the comparator.

**Added: `is_named_version`, `known_versions`, `NOT_STATED_VERSIONS`.**
Whether a value names a predictor version. `None`, `NaN`, whitespace,
and the literal strings `"nan"`, `"none"`, `"<na>"`, `"nat"`, `"null"`
all mean "not stated" — those being what a missing value becomes once
anything calls `str()` on it, which this repo does on reload, on CSV
round trip, and on cache export.

It is public because the obvious version of the rule is wrong in a way
that is easy to miss: `if str(v).strip()` excludes only the *blank*
spellings, since `str(None)` is `"None"` and `str(float("nan"))` is
`"nan"` — both truthy. So the naive rule admits three of the five ways a
version goes missing, not one. That mistake shipped in topiary and,
independently, twice in a downstream consumer that had reimplemented it.

`known_versions(series)` is the same rule over a column, built from the
same `NOT_STATED_VERSIONS` set rather than restating the test.
`is_named_version` now **raises** on a Series or list instead of
answering — returning True for a column of missing versions would have
delivered the exact phantom-version outcome the function exists to
prevent.

**Centralized: one definition of "did the source say anything here?"**
The question was being asked eight different ways — for versions,
alleles, kinds, method names, filter values and TSV cells — and **none
of the copies rejected `"nan"`**, so a frame that had been through
`astype(str)` anywhere carried stringified nulls that every check
accepted as real values. A stringified missing allele became a real
per-allele group.

There are now two public predicates and one set:

```python
is_stated(value)        # scalar:  did the source say anything here?
stated_values(series)   # the same rule over a column
NOT_STATED              # the spellings that mean "no": "", "nan", "none", "<na>", "nat", "null"
```

`is_named_version` / `known_versions` remain as the version-facing names
and delegate — versions were never special. Both scalar forms **raise**
on a container instead of answering; returning True for a column of
missing values is the outcome they exist to prevent.

`NULL_TEXT` is `NOT_STATED` minus the empty string, and the group key
collapses it into a real null. The distinction is load-bearing rather
than cosmetic: `str(None)` is `"None"` and `str(nan)` is `"nan"`, but
**never `""`** — so a blank cell is a stated-but-empty value, which
frames use as a group of its own for allele-free rows. Collapsing it
would merge groups a caller meant to keep apart.

**Fixed: three unmigrated copies of that rule.** `wide.py`'s
`_version_str`, `io.py`'s `_model_version_str` and `cached.py`'s
identity check each had their own version with different coverage —
`_version_str` had no `"nan"` test at all, so a stringified missing
version became the model key `netmhcpan_nan`. All three now delegate.

**Fixed: `fragment_from_effect` could emit a fragment that contradicted
itself.** A stop codon upstream of the reported mutation gave a
zero-length sequence carrying a target interval pointing outside it, so
`peptide_overlaps_target` answered True for a fragment with no residues.
The window is now clamped so every interval lies inside the sequence.
Also: the reference window is clipped at its own stop, so a comparator
cannot carry a `*` the fragment does not; a negative
`padding_around_mutation` is refused rather than producing negative
intervals; and an effect exposing a mutant protein with no
`aa_mutation_start_offset` (varcode's `HaplotypeEffect`,
`ExonicSpliceSite`) raises a message naming the effect and the attribute
instead of `TypeError: unsupported operand type(s) for -`.

AGENTS.md gains a section on this: **logic that does real work belongs in
one documented public function**, because a consumer that needs behavior
it cannot import will reimplement it, and the copy will drift.

## 5.34.0

Two more absent-vs-empty conflations (#223), found by auditing after #214,
#216 and #219 all turned out to be the same defect: two code paths
disagreeing about what "nothing" means. Both turned a missing value into
the literal text `"nan"` and then treated it as data.

**Fixed: a row with no `predictor_version` could be selected as version
`"nan"`.** 5.31.0 taught the ambiguity check and
`resolve_default_versions` that a missing version names no version, but
the *selection* path still compared stringified values — so the two
disagreed, and `Affinity["netmhcpan", "nan"]` returned the NaN-version
row. It was self-inconsistent too: the "no such version" error lists
available versions from `dropna()`, so `"nan"` was never offered, yet
passing it worked. Selection now goes through the same
"was a version named at all" test as everything else, and tolerates
surrounding whitespace the way the resolver does.

**Fixed: `read_pvacseq` fabricated variant ids containing `"nan"`.** The
coordinate fallback (used when a file has no `Index` column) concatenated
stringified `Chromosome`/`Start`/`Reference`/`Variant`, so a blank field
became text inside the identifier:

```
chr1-154590262-nan-A
```

Well-formed-looking, wrong, and *stable* — every row sharing that gap
grouped together under one fabricated id, silently. Such rows now get a
null `variant` and a warning naming how many and which columns were
incomplete. An absent identifier is honest; a fabricated one is not, and
nothing downstream can tell it from a real one. Files with complete
coordinates, and files with an `Index` column, are unchanged and silent.

## 5.33.0

**Added: per-peptide allele sets for `EvalContext(alleles=...)` (#219).**
`alleles=` declared **one** set for the whole frame. A reader that emits
one row per (peptide, allele) passing its own threshold — LENS does —
produces peptides that were each reported against a different subset of
the genotype, so there is no single set to declare. Every LENS fixture
vaxrank has carries 2 to 8 distinct allele sets per file.

It now takes the same three forms `from_predictions(allele_set=...)`
does:

```python
EvalContext(df, alleles=["HLA-A*02:01", "HLA-B*07:02"])   # every peptide, as before
EvalContext(df, alleles={"SIINFEKLA": ["HLA-A*02:01"]})    # per peptide
EvalContext(df, alleles=lambda keys: genotype_for(keys["peptide"]))
```

Declaring the union instead invents a group for every pairing that was
never scored. **This is easy to miss**: an expression containing an
allele-scoped term makes the invented groups read NaN, so the output
looks unchanged — vaxrank swapped in the union and every fixture came
out byte-identical. An expression reading only peptide-level evidence
gives each invented group a real number:

```
peptide_view(proteasome_cleavage.score), two peptides each scored at one allele
  union        6 groups scored, 2 of them pairings never predicted
  per-peptide  4 groups scored, 0 invented
```

A peptide the mapping or callable declares nothing for keeps only the
groups its own rows name; it does not inherit another peptide's
genotype. A mapping key matching no peptide in the frame raises — a key
that declares nothing is indistinguishable from a peptide left
undeclared on purpose. An empty set *for one peptide* is meaningful
("declare nothing here") even though an empty frame-wide sequence stays
an error.

**Added: `describe_default_versions` (#220).** `resolve_default_versions`
returns the winner but not what it won against, so a consumer telling a
user "netMHCpan reports 4.1b and 4.2, scoring with 4.2" had to re-derive
the candidates — and that re-derivation re-implements "was a version
named at all", the rule whose subtlety caused the phantom-`"nan"` bug in
both topiary and vaxrank.

```python
describe_default_versions(df)   # {("pMHC_affinity", "netmhcpan"): ["4.1b", "4.2"]}
resolve_default_versions(df)    # {("pMHC_affinity", "netmhcpan"): "4.2"}
```

Candidates are ordered oldest to newest by the same PEP 440 rule, so the
winner is the last entry for `prefer="newest"` and the first for
`prefer="oldest"`, and the keys match `resolve_default_versions` exactly
so the two zip. `resolve_default_versions` is now implemented *in terms
of* `describe_default_versions`, so there is one place deciding what
counts as a version rather than two that can drift.

## 5.32.0

**Added: read-level evidence and per-field knownness on `ProteinFragment`
(#102).** The first half of the multi-source fragment work — consumer
requirements from vaxrank, which would drop its own
`MutantProteinFragment` and read topiary's. isovar integration is not part
of this; see the issue.

Four RNA read-count fields, none derivable from the aggregate expression
the fragment already carried:

```python
n_overlapping_reads
n_alt_reads
n_ref_reads
n_alt_reads_supporting_protein_sequence   # this assembled sequence, not just the allele
```

**`None` means unknown, and is not `0`.** A source with no read data
leaves them `None`; a source that looked and found nothing sets `0`.
Collapsing the two would let a consumer read "no RNA support" out of "this
source cannot answer" — and the distinction survives a TSV round trip,
since a distinction that does not serialize is decorative.

`field_provenance` states how real a populated field is, mapping a field
name to `"measured"`, `"approximated"` or `"synthesized"`:

```python
ProteinFragment(
    ...,
    variant="chr1:100:N>N",
    n_alt_reads=12,
    field_provenance={"variant": SYNTHESIZED, "n_alt_reads": APPROXIMATED},
)
```

This covers what a bare `None` cannot say. A LENS or pVACseq read count is
real but *estimated* (CDS-overlapping reads; depth × VAF). A placeholder
ref/alt invented because the source supplied none has a value that means
nothing, and anything doing variant-effect annotation on it must refuse
rather than compute. Read it through `provenance_of`, `is_known`,
`is_approximate` and `is_usable_as_biology` rather than the dict. A
provenance entry naming a field that does not exist, or a label outside
the vocabulary, is refused — a typo would sit inert and quietly stop
protecting the field it was written for.

**Fixed: `ProteinFragment.from_dict` hardcoded its field list**, so it
rejected every field added after that list was written. It now derives the
set from the dataclass and cannot drift again.
## 5.31.1

**Fixed: `SelfProteome.nearest` crashed when a peptide-length bucket was
empty (#216).** A proteome whose sources are all shorter than the peptide
window raised `ValueError: attempt to get argmin of an empty sequence`
instead of reporting no match:

```python
SelfProteome.from_fasta("short.fasta").nearest(["SIINFEKLA"])
```

`_build_index` created an entry for every requested peptide length whether
or not any source was long enough to fill it, and `nearest` guarded on
*key presence* rather than emptiness — so an empty-but-present bucket
skipped the graceful path and reached `argmin` over a zero-width axis.

The same fact already had a correct answer by another route: a query whose
length the proteome has no bucket for returns a clean empty row. Absent and
empty mean the same thing here — "no self peptide of this length to compare
against" — and now give the same answer. Unfillable lengths are also no
longer recorded, so `peptide_lengths` stops claiming coverage the proteome
does not have.

This is reachable through `from_fasta` and `from_peptides`, the
caller-supplied paths.

**Fixed: `to_wide` silently emitted `_x` / `_y` columns (#217).** The
annotation columns carried through from the long frame were merged against
the generated `{model_key}_{kind}_{field}` columns with pandas' default
suffixes. A name present on both sides renamed *both* to `_x` / `_y`: the
canonical name then did not exist at all, `from_wide` found no value for
it, and the prediction was lost with nothing said.

```python
to_wide(long_df)          # long_df carries an annotation "netmhcpan_affinity_value"
# ['netmhcpan_affinity_value_x', ..., 'netmhcpan_affinity_value_y']
from_wide(wide)["value"]  # [nan] — the 75.0 that went in is gone
```

Now refused, naming the offending column. This is the same silent
round-trip loss as #208 and #211, one layer up at the point where the
column names are finally assembled — and `_x` / `_y` were pandas merge
artifacts leaking into a schema that never documented them. The realistic
`read_lens` → `to_long` → `to_wide` path was never affected, since
`to_long` consumes the prediction columns rather than leaving them behind
as annotations.

## 5.31.0

**Added: `default_versions` and `resolve_default_versions` (#214).**
`default_methods` answered "which model, when a kind has several".
Nothing answered "which version, when a model has several", so an
unqualified reference on a multi-version frame raised with no configured
way through:

```python
EvalContext(df, default_versions={("pMHC_affinity", "netmhcpan"): "4.2"})
resolve_default_versions(df)          # {("pMHC_affinity", "netmhcpan"): "4.2"}
validate_default_versions(df, mapping)
```

Keyed on `(kind, model)` because a version is only meaningful within a
model — `"4.2"` says nothing on its own. Forwarded by `apply_filter`,
`apply_sort` and `evaluate_scores` like the other context options.

`resolve_default_versions(df, prefer="newest")` orders by PEP 440, which
is the only ordering where `4.10` beats `4.9`; `prefer="oldest"` serves a
pipeline pinned to an older validated model. Versions that aren't PEP 440
(a build tag, a date, a git hash) sort *before* everything that parses,
so "newest" means a real release rather than whichever string sorted
last, with the string as a deterministic tiebreak.

Unconfigured behavior is unchanged — an ambiguous version still raises,
and the error now names `default_versions` alongside the bracket form.

**This was our own gap, made worse by our own fix.** 5.28.2 added the
version-ambiguity raise so two versions of one model could no longer be
silently resolved to whichever row came first. That was right, but it
shipped without the way through — in the same release series whose notes
described the identical problem one level up as "a working configuration
turned into a hard error". Keeping both versions of a multi-version LENS
table (#208) is what made it reachable: a consumer's default scoring
expression could not run on that input at all.

**An unknown version is not a version.** A missing `predictor_version` —
NaN, `None`, blank, or no column at all — is the absence of a version
claim, and both the ambiguity check and the resolver now treat it that
way. Previously a frame with one real version alongside rows recording
none raised `Ambiguous: ... (netmhcpan 4.2, netmhcpan nan)`, naming a
version no caller could pass, while `resolve_default_versions` dropped
the NaN and reported no choice to make — so feeding the resolver's own
answer back in still crashed. A genuine disagreement between two real
versions still raises, and the message no longer invents a third.

`packaging` is now a declared dependency. It was already present
transitively, but the version ordering depends on it, and a fallback that
quietly demotes every version to string order is not a fallback worth
having.

## 5.30.0

**Added: `read_lens(..., binding_metrics=...)` (#211).** The binding-column
map was private and `read_lens` took no mapping argument, so a consumer
hitting an unmapped or mis-mapped column had no supported way to correct
it locally — the only route was a topiary release. That is the real cost
behind #208 blocking a downstream consumer outright instead of being a
local workaround.

```python
read_lens(path, binding_metrics={
    ("newtool", "ic50_nm"): ("affinity", "value"),
    ("sometool", "noisy_metric"): None,   # not a prediction
})
```

Overrides **merge over** the built-in table rather than replacing it, so
one column can be patched without restating the other thirteen. They can
also correct a built-in mapping, not only fill a gap.

Keys are `(tool, metric)` — the same pair the unmapped-column warning
already names, so what topiary tells you is what you pass back — and
deliberately carry no version: a mapping keyed on the raw column name
would stop working the moment a file spelled the version differently,
which is the brittleness #206 removed. Tool and metric are matched
case- and whitespace-insensitively.

Values are `(kind, field)`, validated up front. An unknown kind or field
would emit a wide-form column name that `from_wide` cannot read back, so
the data would reach the frame and then vanish on `to_long()` — the
failure mode this area keeps producing, and not one to hand callers a
new way to cause.

An override cannot build a frame that cannot be read back. Two columns
of one tool mapping to the same `(kind, field)` would put a duplicate
column in the frame — one set of values unreachable, `to_long()` raising
"Expected a 1D array", which is exactly #208's shape. That is refused,
naming both source columns and the name they collided on. Keys that
could never match a column, and two keys that normalize to the same
pair, are refused for the same reason: a validation that lets a silent
no-op through is not doing its job.

`None` declares a column is not a prediction: it silences the
unmapped-column warning without remapping anything. The column itself is
left alone and still reachable as `Column("sometool_1.0.noisy_metric")`
— overriding a mapping does not delete data.

**Also: the unmapped-column warning now names the `(tool, metric)` key**,
not only the column name, so it can be pasted straight into
`binding_metrics` without splitting the column and stripping the version
by hand. An applied override is recorded in
`metadata.extra["lens_binding_metrics"]`, alongside `lens_version` and
`topiary_model_keys` — a frame whose binding columns came from a
corrected map should not be indistinguishable from one built with the
defaults.

## 5.29.0

**Added: `context=` on `apply_filter` / `apply_sort` / `evaluate_scores`
(#179).** Since #176 the three entry points took the same `group_keys` /
`default_methods` / `kind_support` / `alleles` kwargs, so callers could
*specify* one grouping — but each call still built its own `EvalContext`
and recomputed the grouping, a `drop_duplicates` over the frame plus a
row-to-group code array. Pass a prebuilt context instead and that work
happens once:

```python
ctx = EvalContext(df, group_keys=gk)
affinity = evaluate_scores(df, Affinity.value, context=ctx)
presented = evaluate_scores(df, Presentation.score, context=ctx)
ordered = apply_sort(df, [Affinity.value], context=ctx)
```

**What is shareable is narrower than the issue's example suggests, and
worth stating plainly: `apply_filter` and `apply_sort` both return new
frames, so a context cannot be threaded down a filter → sort → score
pipeline.** It is reusable across operations on one *unchanged* frame —
several `evaluate_scores` calls for different score columns, or a filter
and a sort keyed the same way. Passing a context built on a different
frame raises rather than silently mapping rows to another frame's
groups; the check is identity, not equality.

`apply_filter` needs `filter_context=True` and `apply_sort` deliberately
needs it `False`, so a shared context cannot simply be handed to both.
New `EvalContext.derive(**overrides)` returns a context on the same
frame with options changed, inheriting the frame-derived caches when
`df` / `group_keys` / `alleles` are untouched and dropping them when
they are. `apply_filter` uses it internally, so the caller's own context
is never flipped.

**Added: `default_methods` on `TopiaryPredictor` and
`TopiaryResult.filter_by` / `sort_by` (#178).** A predictor running two
models that produce the same kind made every unqualified reference in
`sort_by` ambiguous, and there was no way to resolve it through the
predictor:

```python
TopiaryPredictor(
    models=[NetMHCpan, MHCflurry], alleles=[...],
    sort_by=Affinity.value,                      # ValueError: Ambiguous
    default_methods={"pMHC_affinity": "mhcflurry"},   # now resolvable
)
```

Filtering hid this — `filter_context=True` auto-aggregates directional
comparisons across methods — so it surfaced on the sort rather than on
the filter that looks the same. `TopiaryResult.filter_by` / `sort_by`
also gained `group_keys`, for a frame that acquired a provenance column
after prediction.

There is deliberately no `group_keys` on `TopiaryPredictor`: it builds
its own frame, so the inferred grouping is right by construction, and a
knob that can only be set wrong is not worth the surface. `kind_support`
was already forwarded on both paths.

## 5.28.2

**Fixed: `read_lens` collapsed two versions of one tool into one column
(#208).** A regression introduced by 5.28.0.

Keying binding columns on `(tool, metric)` fixed the version-brittleness
in #206 but left the emitted name — `{tool}_{kind}_{field}` — with no
room for a version. A table carrying both `netmhcpan_4.1b.aff_nm` and
`netmhcpan_4.2.aff_nm` produced the column `netmhcpan_affinity_value`
twice, one version's values were dropped, and `Metadata.models` kept
only one. topiary said nothing: the sole signal was a pandas
duplicate-column warning later, which doesn't name the predictor that
lost its values, and `to_long()` raised `ValueError: Expected a 1D
array` — so a consumer could not route around it either.

Multi-version LENS tables are a real input shape, not a constructed
edge case.

When two versions of one tool would claim the same output column, that
tool's columns are now qualified with the version —
`netmhcpan_4.1b_affinity_value`, `netmhcpan_4.2_affinity_value` —
following the convention `to_wide` already uses for the same situation,
and topiary warns, naming the tool and the versions.

The test is a genuine name collision, not merely "two version strings
appeared for this tool". A file that spells one run's version
inconsistently across metrics — `netmhcpan_4.1b.aff_nm` beside
`netmhcpan_4.1.score_ba` — has no collision, and qualifying it would
split one predictor's affinity axis into two half-populated ones,
undoing what #206 fixed.

`Metadata.models` keeps its documented `{method: version}` shape, so
`models["netmhcpan"]` still answers for every file; where a method has
several versions it holds one of them. The full mapping is
`metadata.extra["topiary_model_keys"]`, which gives each emitted model
key the `[method, version]` it was built from. A file with one version
per tool is unchanged, down to the absence of a warning.

**Fixed: `from_wide` lost the version it was handed.** Independent of
LENS, and older. `to_wide` appends the version to the model key when one
method has several, but `from_wide` set
`prediction_method_name` to the whole key and left `predictor_version`
NaN — so a round trip produced a method named `netmhcpan_4.1b` with no
version, and a version-qualified reference like
`Affinity["netmhcpan", "4.2"]` matched nothing.

`to_wide` now records what each model key was built from in
`attrs["topiary_model_keys"]`, and `from_wide` reads it. It does not
guess: a method genuinely named `netmhcpan_4.1b` and an encoded
`(netmhcpan, 4.1b)` are the same string, so stripping a trailing
`_{version}` would rename the former. Only the writer knows which it
made, and now it says so.

**Added: the DSL refuses a silently arbitrary version.** An unqualified
`Affinity.value` on a frame holding one method at two versions raised
nothing and returned whichever row came first. It now raises, listing
the versions, the way it already did for two methods.

## 5.28.1

**Fixed: `derive_mhc_class` classified alleles by name prefix.**

It read `HLA-A/B/C` as class I and `HLA-D*` as class II, so every other
real allele came back `pd.NA`:

| allele | before | after |
|---|---|---|
| `HLA-E*01:01`, `-F`, `-G` | NA | I |
| `H2-Kb` (mouse) | NA | I |
| `H2-IAb` (mouse) | NA | II |
| `BoLA-N*01301`, `Mamu-A1*001:01`, `SLA-1*01:01` | NA | I |

`pd.NA` is not a harmless answer here: it drops a row from the
`class_i` **and** `class_ii` filters alike, so those peptides were in
neither view rather than in the wrong one. The non-classical human class
I genes are the reachable case for a human pipeline; the non-human ones
matter for anyone predicting outside *Homo sapiens*.

Alleles are now parsed with mhcgnomes, which places all of the above.
AGENTS.md has said so all along — *"Use mhcgnomes for MHC allele
parsing. Never `startswith("HLA-")` or other string hacks — alleles
aren't always human"* — and this was the one place in the codebase not
following it.

Distinct alleles are parsed once per call rather than once per row:
100k rows over two distinct alleles takes ~13 ms.


## 5.28.0

**Fixed: `read_lens` dropped a predictor whose version spelling it
didn't know (#206).**

A LENS binding column is named `<tool>_<version>.<metric>`, and the
mapping table was keyed on the whole name. `netmhcpan_4.1b.aff_nm`
mapped; `netmhcpan_4.1.aff_nm` passed through verbatim, so that
predictor's entire affinity axis was absent from the normalized frame —
with nothing raised. A consumer reading normalized names could not tell
"this tool emitted nothing" from "this tool emitted something under a
version I don't recognize".

`aff_nm` is an IC50 whichever NetMHCpan produced it, so the table is now
keyed on `(tool, metric)` and the version is *recorded* — in
`Metadata.models`, where it already was — rather than matched. A new
predictor release needs no change here. Version detection had the same
brittleness for its NetMHCstabPan marker and got the same treatment.

**A column topiary doesn't recognize is now reported.** One that looks
like predictor output but names an unknown tool or metric warns, naming
the column. Its values stay in the frame under the original name —
nothing is discarded — but the silence was the part that made this
expensive to find.

**Correction to the 5.27.0 notes.** They described the `allele_set`
callable as serving "attribution decided per peptide". It doesn't.
`allele_set` declares *the genotype a prediction was scored against*, and
a peptide-level score is then projected to every allele group present —
so naming one allele does not withhold the score from the others.
Attributing a peptide-level score to some alleles and not others is a
different operation, and one topiary does not provide. The callable is
still useful for its actual purpose: per-prediction genotype
declaration, where different predictions in one frame were scored
against different allele sets.

## 5.27.0

**`from_predictions()` can carry a consumer's own columns and per-peptide
allele sets (#203):**

```python
df = from_predictions(
    predictions,
    extra_columns={"prediction_id": ids, "peptide_offset": offsets},
    allele_set=lambda prediction: attribution_for(prediction),
)
```

5.26.0's version could not express two things a real consumer needs, so
adopting it would still have meant hand-writing the frame — which is
what the function exists to stop.

`extra_columns` carries identity a prediction doesn't itself have: a
provenance key the consumer groups by rather than letting topiary infer
from `source_sequence_name`, or an offset belonging to the candidate a
peptide came from rather than to the prediction object. A scalar fills
the column; a sequence is positional, one value per prediction, and a
length mismatch is an error rather than a silent misalignment.

`allele_set` now also takes a **callable**, receiving each prediction
(or each row, for a DataFrame input) and returning its alleles or
``None``. Per-kind was the wrong granularity for attribution decided per
peptide, where two peptides in one frame legitimately get different sets
for the same kind — and the alternative, one call per candidate, doesn't
scale to a report with ~100k of them.

The 1:1 input ordering is now **documented as part of the contract**,
since positional data depends on it. It was already true; it wasn't
promised.

## 5.26.1

**Correction to the 5.24.0 notes.** They claimed the shipped
`CANONICAL_METHOD_PREFERENCE` "matches the convention already in use
downstream, so adopting this changes no existing scores." That was
wrong, and the claim has been removed from the 5.24.0 entry.

The orders differ in where `netmhcstabpan` sits:

```
topiary   mhcflurry, netmhcpan, netmhcpan_ba, netmhcpan_el, netmhcstabpan
vaxrank   mhcflurry, netmhcpan, netmhcstabpan, netmhcpan_el, netmhcpan_ba
```

So a frame where two of those models produce the *same* kind resolves
differently depending on which table was consulted — reachable for
`pMHC_affinity`, where `netmhcpan_ba` predicts affinity directly while
NetMHCstabPan's affinity is a by-product of predicting stability.
Adopting topiary's table therefore does change that resolution for a
consumer that had the other order.

The order itself stands, on the rationale the docstring gives: a model
whose output for a kind is secondary to its main job sorts after ones
that predict it directly. What was wrong was asserting compatibility
without checking it — the divergence is exactly the disagreement
`resolve_default_methods` exists to end, and it was live between two
tables while the notes said otherwise.


## 5.26.0

**`predict_self_nearest` — paired MHC binding for the nearest self
peptide (#190):**

```python
TopiaryPredictor(
    models=[...], alleles=[...],
    self_proteome=SelfProteome.from_fasta(path),   # your reference
    predict_self_nearest=True,
)
```

`SelfProteome.nearest()` already said *which* healthy peptide a
candidate resembles. Whether that peptide is presented by the same
allele is a separate question, and it is the one a cross-reactivity
judgement turns on: a near-identical self peptide the patient's MHC
never presents is not the same risk as one it does.

The flag runs a second prediction pass, as `predict_wt` does, scoring
each `self_nearest_peptide` at its row's own allele and filling
`self_nearest_value` / `_score` / `_percentile_rank`. The columns reach
the DSL through the `self_nearest` scope, so an exclusion is expressible
directly — and since 5.23.0 scoped fields work in filters, which is what
an exclusion needs:

```python
apply_filter(df, (Affinity.value <= 500) & (self_nearest.Affinity.value >= 1000))
```

The reference proteome stays the caller's: `SelfProteome` takes a FASTA,
a peptide mapping, or Ensembl with your own `cta_source` and
`tissue_gene_ids`. Topiary computes the comparison, not the definition
of self.

**Known limitation.** The self peptide is scored without flanking
context — it comes from the reference proteome, and `nearest()` reports
its gene, transcript and offset but not the residues either side. Kinds
that read flanks (antigen processing, and presentation where its model
uses them) are scored on the peptide alone; affinity and stability are
unaffected.

## 5.25.0

**`from_predictions()` — build the long form without copying the schema
(#194):**

```python
from topiary import from_predictions

df = from_predictions(predictions)                        # Prediction objects
df = from_predictions(model.predict_dataframe(peptides))  # or mhctools' rows

df = from_predictions(
    predictions,
    allele_set={"pMHC_presentation": patient_alleles},    # genotype-level kinds
)
```

A caller holding `mhctools.Prediction` objects — a report reader, a
cache, anything that didn't run a `TopiaryPredictor` end to end — had to
write topiary's long form by hand: the column names, the `kind` strings,
the value / affinity / score / percentile_rank mapping, and the
provenance columns. That is a copy of the schema topiary cannot see and
cannot migrate, so a column added here never reaches it. `allele_set`
(5.21.0) is the live example.

The normalization is now one function, shared by `from_predictions` and
by `TopiaryPredictor`'s own output, so the two paths cannot diverge.
`allele_set` takes a sequence (applies to every row) or a
`{kind: alleles}` mapping, which is what a mixed list of per-allele and
genotype-level predictions needs.

An empty input returns an empty frame **in topiary's vocabulary** rather
than mhctools' — a caller that got `offset` and `predictor_name` back
from an empty result would break on the frame's shape, not on its
emptiness.

## 5.24.0

**Opt-in canonical method resolution (#193):**

```python
from topiary import resolve_default_methods, validate_default_methods

defaults = resolve_default_methods(df)   # {"pMHC_affinity": "mhcflurry"}
evaluate_scores(df, node, default_methods=defaults)
```

An unqualified reference to a kind produced by several models raises,
and that stays — silently choosing a model is not something topiary
should do behind a caller's back. What was missing is a supported way to
say *pick the canonical one*, so every consumer wrote its own preference
table, and two tools could disagree about what canonical means with
nothing surfacing the difference.

`resolve_default_methods` returns an entry only for kinds that actually
have a choice. It resolves by `CANONICAL_METHOD_PREFERENCE`, which is a
**tie-break convention, not a quality ranking**: general-purpose
predictors ahead of ones whose output for a kind is secondary to their
main job (NetMHCstabPan predicts stability; its affinity comes along
with it), mode variants after the model they vary, and anything unlisted
alphabetically after those so the answer is always deterministic. Pass
`preference=` to override it.

`validate_default_methods(df, default_methods)` reports an entry naming
a kind or a model the frame doesn't have. `EvalContext` only consults a
default when a kind is *actually* ambiguous, so such an entry is
otherwise inert — and stays inert until the day two models produce that
kind, when it starts deciding. Checking up front turns a typo in a
config file into an error where it was written.


## 5.23.0

**Scoped fields work in filters (#192):**

`wt.`, `self_nearest.`, `shuffled.` and `self.` raised `TypeError` inside
a comparison, directing callers to sorting expressions instead. That
blocked the standard analysis:

```python
# the mutant binds and the wildtype doesn't — differential agretopicity
apply_filter(df, (Affinity.value <= 500) & (wt.Affinity.value >= 1000))
```

Selecting neoepitopes that way is an exclusion, and a sort cannot
exclude. The same applies to a cross-reactivity rule on
`self_nearest.`, which is a filter by nature.

The ban never prevented the operation either — `column(wt_value) >= 1000`
reads the same values and was always allowed — so it only made the
scoped vocabulary unavailable for it, while leaving the identical
failure mode reachable by the other spelling.

**The hazard it was standing in for is now reported.** A comparator
column a producer never wrote makes the expression NaN for every group,
and NaN in a filter drops the frame. A filter reading a scope the frame
doesn't carry now warns, naming the missing column. Outside a filter
NaN is a sensible answer, so ranking and scoring are unaffected.


## 5.22.1

**Regression coverage for the 5.22.0 projection fix.**

The test shipped with #197 used a frame where some rows of the kind
carried a real allele. That case never reproduced the bug: the old row
scan saw the counter-example and already answered `single_allele`, so
the test passed before the fix as well as after it. Reproducing requires
*every* row of the kind to be blank-allele, which is the shape the
openvax/vaxrank#348 review found. No behavior change — 5.22.0's fix was
correct, its regression test simply did not pin it.


## 5.22.0

**`KIND_MHC_DEPENDENCE` — what a kind is about, before any rows (#195):**

```python
from topiary import KIND_MHC_DEPENDENCE

KIND_MHC_DEPENDENCE["pMHC_affinity"]       # "single_allele"
KIND_MHC_DEPENDENCE["antigen_processing"]  # "none"
```

A public, model-independent default for every kind topiary knows: does
it describe a peptide-MHC pair, or the peptide alone? The `pMHC_*` kinds
name a pair and are per-allele; the processing-pathway kinds (cleavage,
transport, trimming) describe the peptide. `immunogenicity` sits with
the per-allele kinds because every mhctools predictor emitting it scores
a peptide against an allele.

Consumers previously had to maintain their own copy of this table, which
catches completeness drift but not disagreement. It also could not be
answered at all on external-input runs, where there is no predictor and
therefore no `kind_support`.

**One public resolver, `mhc_dependence()`:**

```python
from topiary import mhc_dependence

mhc_dependence("antigen_processing")                       # "none"
mhc_dependence(kind, kind_support=predictor.kind_support)  # a model's own statement
mhc_dependence(kind, rows=df)                              # reads allele_set if present
```

Usable with nothing but a kind, which is the case on external-input
runs. Evidence is consulted in order of how specific it is — a model's
`kind_support`, then an `allele_set` in the rows, then the kind's
default, then the rows themselves and only for a kind topiary doesn't
know. The DSL's internal resolution is now a thin caller of this, so
there is one implementation rather than a public and a private one free
to diverge.

`MHC_DEPENDENCE_VALUES` is re-exported from mhctools rather than
restated — topiary had been carrying a hand-copy of the same three
values, which is the drift this release is about.

**Fixed: a malformed allele-scoped row was read as peptide-level.**

Dependence resolution fell back to scanning rows, and a peptide-level
record and an allele-scoped record that lost its allele scan the same
way. So a `pMHC_affinity` row with a blank allele was read as
allele-free and projected across the peptide's alleles — inventing
binding evidence for alleles no model scored. Only the kind separates
those two cases, and now it does:

1. a predictor's `kind_support`, if supplied
2. an `allele_set` in the rows
3. the kind's default from `KIND_MHC_DEPENDENCE`
4. row inspection, only for a kind this topiary doesn't know

A blank allele on an allele-scoped kind now warns and stays per-allele.
Genuinely peptide-level kinds project exactly as before.

One narrowing follows: a blank-allele `pMHC_presentation` row is no
longer projected on the strength of the blank alone. With
`kind_support` reporting `haplotype`, or an `allele_set` in the row, it
projects as before — but rows cannot distinguish a genotype-level
prediction from a per-allele one that lost its allele, so without that
evidence the conservative reading applies.


## 5.21.1

**Fixed: `apply_sort`'s ranking depended on the order rows arrived in
(#191).**

The comparator skipped a key when *either* side was missing. Skipping is
pairwise, so "equal" stopped being transitive — and `sorted` needs a
consistent comparator. Three groups, sorting on two keys, the same data
in three input orders:

```
input ['A', 'B', 'C'] -> ['A', 'B', 'C']   <- A (k0=1) above C (k0=2)
input ['C', 'B', 'A'] -> ['C', 'B', 'A']
input ['B', 'A', 'C'] -> ['B', 'C', 'A']
```

A missing sort key is the ordinary case — a peptide with no presentation
row has no presentation score — so this was reachable without anything
unusual, and it silently produced a different ranking depending on how
the frame had been concatenated.

**Keys are now ranked rather than compared pairwise**, which gives every
group a definite position while keeping the property the skip was
reaching for: a group with no value for a key takes the average rank of
the groups that do have one, so the key neither promotes nor penalizes
it and the remaining keys decide. Frames with no missing values sort
exactly as before.

**Also much faster.** The old comparator ran per pair in Python and was
the dominant cost of sorting:

| rows | groups | before | comparator's share |
|---|---|---|---|
| 100,000 | 85,000 | 2.05 s | ~100% |
| 400,000 | 340,000 | 11.42 s | 74% |

Ordering is now a single `np.lexsort` over the ranked keys.

## 5.21.0

**`allele_set` — storing what a genotype-level prediction was scored
against (#168):**

MHCflurry's presentation predictor in haplotype mode scores a peptide
against a sample's whole allele list and reports the allele it
deconvolved as the likeliest presenter. mhctools puts that one allele in
the row, so the prediction reads exactly like a per-allele one and the
frame loses the fact that the score is about the set. This is the
default path: `presentation_allele_mode="auto"` resolves to haplotype
for six alleles or fewer.

```
  peptide      allele              allele_set               kind    score
SIINFEKLA HLA-A*02:01                              pMHC_affinity 0.240074
SIINFEKLA HLA-B*07:02                              pMHC_affinity 0.044721
SIINFEKLA HLA-A*02:01 HLA-A*02:01,HLA-B*07:02  pMHC_presentation 0.027879
```

`allele` keeps the best allele — nothing that reads it breaks, and the
attribution isn't discarded. `TopiaryPredictor` writes `allele_set` for
kinds a model reports as `mhc_dependence='haplotype'`, and the cache,
CSV/TSV round-trip, `to_wide`/`from_wide`, and `combine_predictions`
identity all carry it.

**It joins the group keys when populated**, which is what keeps a
genotype-level row out of one allele's group — otherwise its score is
read as that allele's. Frames without genotype-level rows keep the
narrower key, the same way a blank `sample_name` is left out.

**It makes a frame self-describing.** `mhc_dependence` is now read from
the set, so a genotype-level row keeps its meaning through a file, where
`kind_support` cannot follow — the gap that motivated the column. An
allele-scoped filter also keeps such a row alive under the same
peptide-level rule as allele-free evidence (5.19.0).

**`Column.includes()` asks the set question; `eq()` stays equality:**

```python
Column("allele").eq("HLA-B*07:02")            # the row's allele is B*07:02
Column("allele_set").includes("HLA-B*07:02")  # the set scored includes B*07:02
```

`includes()` compares whole tokens, never substrings — allele names
prefix one another (`HLA-A*02:01` is a prefix of `HLA-A*02:010`, both
real alleles), so a substring test reports membership that isn't there.
Tokens compare as stored, so writers are responsible for canonical
names. Parses in string form as
`column(allele_set).includes('HLA-B*07:02')` and round-trips.

Deferred, as recorded on #168: per-allele attribution of genotype-level
scores. `includes()` is literal set membership — it does not match a
per-allele row whose own allele is the argument — and reaching the
peptide's allele groups stays a projection (`peptide_view`) rather than
a membership change.

## 5.20.1

**Fixed: a labeled haplotype kind read more quietly than an unlabeled one.**

The auto-projection added in 5.20.0 covered `mhc_dependence='none'` only,
so supplying correct metadata made the bare read fail *more* silently:

```
no kind_support          bare read = [0.9, 0.9, 0.9]   warns
kind_support=haplotype   bare read = [nan, nan, 0.9]   silent
```

A haplotype prediction scores a whole genotype, and mhctools stamps the
row with the allele it deconvolved as the best presenter. Reading it
plainly therefore hands that joint score to one allele and leaves the
rest of the genotype NaN — the same failure 5.20.0 fixed for
allele-free kinds, wearing an allele name. Both peptide-level modes
(`none` and `haplotype`) are now projected, with a warning naming
`peptide_view(...)`.

This matters on the default path: MHCflurry's
`presentation_allele_mode="auto"` resolves to haplotype for six alleles
or fewer, i.e. every ordinary patient genotype, and 5.18.1 made
`TopiaryPredictor` forward `kind_support` automatically — so a
`filter_by="presentation.score >= 0.5"` was reading NaN for every allele
but one.

`single_allele` kinds are unchanged: a plain read there returns a real
row, and choosing which row stays with the caller.

## 5.20.0

**A bare allele-free kind is projected instead of reading NaN (#186):**

Referencing an allele-free kind without `peptide_view()` in a grouping
keyed by allele returned NaN for every allele group, silently — the row
carries no allele, so it is in none of those groups and the plain read
can never find it:

```python
evaluate_scores(df, Processing.score)   # 5.19.0: [nan, nan, 0.77]
                                        # 5.20.0: [0.77, 0.77, 0.77] + UserWarning
```

That reading has no useful meaning, so the reference now means the one
thing it can: the peptide's value, projected across its groups, exactly
as `peptide_view()` does. A `UserWarning` names the explicit form, so
the implicit behavior is greppable and migration is optional rather than
urgent.

This matters most for user-facing config: `score_expr` / `filter_expr`
strings that read a processing kind used to work only because producers
duplicated the allele-free row across a patient's alleles before
evaluation. Removing that duplication — the point of #182 and #183 —
would otherwise have left the same string parsing, validating, and
scoring zero.

**Only `mhc_dependence='none'` is treated this way.** A per-allele kind
read plainly returns a real row, and choosing *which* row is a genuine
decision that stays with the caller and `best_*` / `peptide_view()`.
Explicit `peptide_view(...)` never warns, and the one-value-per-peptide
rule is unchanged: an allele-free kind carrying two different values for
one peptide still raises.

## 5.19.0

**Allele-free predictions reach a genotype (#182), and survive a filter
(#183):**

An antigen-processing prediction carries no allele, so it lands in a
group of its own rather than in any of the peptide's per-allele groups.
`peptide_view()` (5.18.0) broadcasts its value, but two things still
stopped it from replacing the row duplication producers do by hand.

**A filter on an allele-scoped kind no longer drops it.** That group
holds no rows of the kind being filtered on, so the predicate evaluated
to NaN — which pandas turns into False, which dropped the row and took
the evidence out of the frame before the score expression could read it:

```python
apply_filter(df, parse("affinity.value <= 500"), group_keys=keys)
# 5.18.0: the processing row is gone, and a later
#         peptide_view(processing.score) reads NaN
```

An allele-free group holding none of the kinds a filter reads is now
kept whenever the filter kept at least one of that peptide's allele
groups. A filter that *does* read that kind still decides it, and a
peptide excluded entirely takes its evidence with it.

**`alleles=` declares the alleles to evaluate against.** Groups come
from the rows, so a peptide whose only evidence is allele-free has no
per-allele group for a consumer keyed by patient allele to read — and
the genotype is not something the frame contains:

```python
evaluate_scores(df, peptide_view(Processing.score),
                group_keys=keys, alleles=["HLA-A*02:01", "HLA-B*07:02"])
```

`alleles` is the fourth shared context option, accepted by
`apply_filter`, `apply_sort`, `evaluate_scores` and `EvalContext`. It
adds one group per peptide per declared allele; those groups hold no
rows, so allele-scoped fields read NaN there — that allele has no
prediction of its own. The frame is untouched: only the group index
grows, so row counts out of every entry point are unchanged.

Together these let a producer keep one canonical allele-free row instead
of duplicating it across a patient's alleles.

## 5.18.1

**`peptide_view()` resolves the allele mode from the rows it reads (#181
follow-up):**

5.18.0 decided a kind's `mhc_dependence` from `kind_support` plus a scan
of the whole frame, which let three things go wrong silently:

- A `default_methods` entry naming a model absent from the frame — the
  normal case for a pipeline-wide default that covers another kind —
  filtered every model out of the metadata lookup, so the projection
  fell back to guessing from the rows. A haplotype frame with two rows
  for one peptide became a silent `max()` instead of an error.
- One model's allele-stamped rows reclassified another model's
  allele-free rows: adding an unrelated per-allele processing row made
  an explicitly method-qualified `peptide_view(processing['netchop'].score)`
  silently `max()` away a conflict it had correctly rejected before.
- Models that contributed no rows still triggered "models disagree", and
  the remedy the message suggested then failed with "no predictions from
  method matching ...".

The mode is now resolved from the already-selected rows — the ones the
expression will actually read, after kind, method and version filtering
— and `kind_support` is consulted only for models those rows came from.
That also removes the duplicate per-kind scan each evaluation did.

**The one-value-per-peptide check no longer depends on the grouping.**
With `allele` absent from `group_keys`, `peptide_view` returned a plain
field read, so the same node on the same data raised for one grouping
and silently returned `.first()` for another.

**`peptide_view(kind.best_X)` on a field with no defined best direction**
(e.g. `processing.best_value`) silently read the plain `value` column
instead. It now says what is wrong, as the equivalent mistake on a
peptide-level kind already did. The related error message no longer
claims `mhc_dependence='single_allele'` "means one row per peptide" when
the real cause is the missing direction.

**Unknown `mhc_dependence` values** are reported as version skew even
when another model reports a known value; previously the conflict check
ran first and offered a remedy that cannot resolve an uninterpretable
value.

**`kind_support` now reaches the DSL from the object API.**
`TopiaryPredictor(filter_by=..., sort_by=...)` and
`TopiaryResult.filter_by` / `.sort_by` forwarded no metadata, so every
guard that depends on `mhc_dependence` — telling `haplotype` from
`single_allele`, the inconsistent-value error, the version-skew error,
the `single_allele` warning — was inert on exactly the path the docs
advertise for `--filter-by` / `--sort-by`. `TopiaryPredictor.kind_support`
now also skips models that don't report it (older mhctools predictors,
test doubles) instead of raising `AttributeError`.

The `single_allele` warning names the expression that was written —
`peptide_view(affinity.value)` rather than a `best_score` the user never
typed — and points at the caller's frame.

**Breaking (unannounced in 5.18.0):** the scoped-field filter guard now
covers `BestAlleleField`, so `wt.Affinity.best_value <= 500` raises
`TypeError` in a filter like the bare `wt.Affinity.value` always did.
Use scoped fields in sort or score expressions instead.

## 5.18.0

**`peptide_view()` — per-`mhc_dependence` peptide-level projection (#169):**

The DSL reads one value per (peptide, allele) group, but which reduction
is *correct* depends on the predictor's allele mode: best-across-alleles
for `single_allele` kinds, a direct read for `haplotype` and allele-free
(`none`) ones. Callers had to know which kind needed `best_*` and which
did not, and getting it wrong read an arbitrary row.

```python
from topiary import peptide_view, Affinity, Processing

0.5 * peptide_view(Processing.score) + 0.5 * peptide_view(Affinity.score)
```

The allele-free case could not be expressed at all before. An antigen
processing row carries no allele, so it forms its own group and every
per-allele group reads `NaN`:

```python
evaluate_scores(df, Processing.score)                # [nan, nan, 0.77]
evaluate_scores(df, peptide_view(Processing.score))  # [0.77, 0.77, 0.77]
```

Producers worked around this by duplicating each processing row across
the patient's alleles before handing topiary a frame. With `peptide_view`
the frame keeps one canonical row and the value is broadcast at
evaluation time, so `affinity <= 500 & peptide_view(processing.score) >=
0.5` works on an unduplicated frame.

The mode comes from `EvalContext(kind_support=...)`; without it, a kind
whose rows carry no allele is treated as allele-free and everything else
as per-allele. Inconsistent input raises rather than picking silently:
a peptide-level kind carrying two different values for one peptide, and
models that disagree about `mhc_dependence`, are both errors.

Available in string form (`peptide_view(processing.score)`), so it works
in `--filter-by` / `--sort-by` and in consumer config files, and it
round-trips through `to_expr_string()`.

The `single_allele` caveat is unchanged and still warns when
`kind_support` is present: best-of-per-allele is not a joint
multi-allele aggregate, because the predictor never saw the alleles
together.

**Fixed: `best_*` fields sorted backwards under `sort_direction="auto"`.**
`apply_sort` inferred a sort direction only from a bare `Field`, so
`apply_sort(df, [Affinity.best_value])` ranked the *worst* binders
first — 5000 nM above 50 nM — while `apply_sort(df, [Affinity.value])`
ranked them correctly. Direction is now read through the wrappers that
reduce a field to one value per peptide (`BestAlleleField` and
`peptide_view`), which change which row is read, never which end is
better. Sorts that relied on the inverted order will flip; pass
`sort_direction="desc"` explicitly to keep it.

`peptide_view` also refuses input it cannot honor rather than reading
something else: a `best_*_allele` field (an allele name is not a value
per peptide), a scoped `wt.` / `shuffled.` / `self.` field inside a
filter (the wrapper was a hole in that guard), an `mhc_dependence`
value this topiary doesn't know, and a grouping with no peptide
dimension. Peptide-level values are compared with a float tolerance, so
a row round-tripped through a CSV and one computed in-process no longer
disagree over the last bit.

## 5.17.1

**Fixed: tissue expression lookups against current pirlygenes (#177):**

`topiary.sources` built its per-tissue column names as `nTPM_<tissue>`,
but pirlygenes now emits the suffix form (`<tissue>_nTPM`), matching the
convention its own FPKM/TPM helpers accept. `available_tissues()`
returned `[]` and `tissue_expressed_gene_ids(["heart_muscle"])` raised
`ValueError: Unknown tissue column(s)`, taking
`tissue_expressed_sequences` and the `--tissue` CLI paths with them.

Tissue names are now read under either spelling, so topiary works
across pirlygenes releases rather than pinning to one of them. Unknown
tissue names are also reported as tissue names rather than as column
names.

## 5.17.0

**Explicit group keys everywhere (#175):**

`apply_filter`, `apply_sort`, and `evaluate_scores` now accept the same
three keyword-only context options — `group_keys`, `default_methods`,
and `kind_support` — and forward all of them to `EvalContext`. Filtering,
sorting, and scoring can therefore share one grouping and one method
resolution instead of each entry point supporting a different subset.

```python
group_keys = ["prediction_id", "source_sequence_name", "peptide",
              "peptide_offset", "allele"]

kept = apply_filter(df, Affinity <= 500, group_keys=group_keys)
scores = evaluate_scores(kept, Affinity.score, group_keys=group_keys)
```

This matters when a frame carries a stable provenance identity: the
inferred group keys are sequence-oriented, so two rows sharing peptide,
source sequence, and offset are one group even when they came from
different variants, transcripts, or genes — and one row's filter decision
then applies to the other. Callers that previously reimplemented
`apply_filter` on top of `EvalContext` just to supply `group_keys` (e.g.
vaxrank's LENS/pVACseq path) can now call `apply_filter` directly.

Explicit `group_keys` are validated before anything is evaluated: a bare
string instead of a sequence, an empty sequence, duplicate entries, and
names that aren't columns all raise immediately, with a near-match
suggestion, instead of failing deep inside a node. Validation also runs
on the paths that return early (empty frame, `node=None`, no sort
nodes), so a typo can't pass silently just because a pipeline has
degenerated to zero rows.

Frames that group key *inference* can't handle — missing one of the
identity columns it expects — now raise a `ValueError` naming the
missing columns and pointing at `group_keys=`, rather than a bare
`KeyError` from inside pandas.

**Fixed: null group keys silently dropped rows and scores.** `None`,
`NaN` and `pd.NA` in an identity column are one group under
`groupby(dropna=False)` — which every node evaluates through — but the
group index was built from raw values, which keeps them apart, and rows
were matched to groups by key lookup, which never matches a null key
(`NaN != NaN`, and since Python 3.10 it hashes by identity). A frame
mixing `None` and `NaN` in `source_sequence_name` — which
`TopiaryPredictor` produces, since it writes `source_sequence_name =
None` — could lose rows from `apply_filter` that its own scores said
should pass, while `apply_sort` kept them. Null spellings are now
collapsed once, and rows map to groups by position via the new
`EvalContext.row_group_codes()`.

**Fixed: a single group key produced empty or all-NaN results.**
`EvalContext.group_index` built a 1-level `MultiIndex` for a single
group key, while `DataFrame.groupby` on one key produces a flat `Index`.
Every node result therefore reindexed to all-NaN: `apply_filter` dropped
every row, `evaluate_scores` returned all-NaN, and `apply_sort` silently
no-opped. `group_index` is now a flat `Index` (and `row_group_tuples`
yields bare values) when there is one group key, matching pandas.
Multi-key grouping is unchanged. This was reachable before via
`EvalContext(df, group_keys=["peptide"])`; the new kwarg makes it
reachable from all three entry points.

**Blank `sample_name` is no longer an inferred group key:**

`mhctools` stamps `sample_name=""` on every row of a single-sample run,
which made group key inference prepend a constant `sample_name` level to
every group tuple of ordinary predictor output. A `sample_name` column
that is entirely null or blank is now ignored by inference (null-only
columns already were). Frames that mix real names with blanks keep the
key — there the blank is a distinguishing value — and `group_keys=` can
still name `sample_name` explicitly.

**Breaking:**

- The context options are now keyword-only. Calls that passed them
  positionally — `apply_filter(df, node, default_methods)`,
  `evaluate_scores(df, node, group_keys, fill)` — must use keywords.
  Positional `df`, `node` / `sort_nodes`, and `sort_direction` are
  unchanged.
- An empty `group_keys` sequence now raises instead of falling back to
  inferred keys. Pass `group_keys=None` to infer.
- `EvalContext.group_index` is a flat `Index` rather than a 1-level
  `MultiIndex` when there is a single group key (see the fix above).
  Code that indexed single-key results with 1-tuples must use bare
  values.
- Inferred group keys drop a blank-only `sample_name`, so group tuples
  for single-sample `mhctools` output are 4-wide rather than 5-wide.
  Consumers that materialize a group tuple and index a per-group Series
  with it must drop the leading `sample_name` element (or pass
  `group_keys=` explicitly to keep it).


## 5.16.2

**Combine separate predictor runs (#170):**

`topiary.combine_predictions([a, b, ...])` combines separate
predictor outputs into the same long-form shape produced by running
those predictors together. It accepts `TopiaryResult` or fresh
`TopiaryPredictor` DataFrame outputs, supports both split-by-predictor
and split-by-allele/peptide-length runs, rejects duplicate
`(prediction_method_name, kind, identity)` predictions, and by default
requires every emitted `(prediction_method_name, kind)` group to cover
the same identity grid. Use `coverage="partial"` only for deliberate
sparse unions.

Fresh `TopiaryPredictor` DataFrames now carry lightweight
`DataFrame.attrs` model-version metadata (`topiary_models`) so this
helper can preserve model provenance without changing the public return
type. The emitted rows remain the source of truth for which predictor
produced which quantities: `prediction_method_name`, `predictor_version`,
`kind`, and the value/rank columns are not duplicated into separate
`kind_support` metadata.

`TopiaryPredictor(name=...)` now optionally records per-run provenance
in a `prediction_run_name` column. This is intended for split predictor
grids such as one NetMHCpan run per allele/peptide length: the logical
method remains `prediction_method_name="netmhcpan"`, while
`prediction_run_name` records the shard. `combine_predictions`
and `to_wide()` treat the run name as provenance, not as a separate
prediction identity, so disjoint shards combine cleanly and overlapping
shards still fail as duplicate predictions.

`combine_predictions` also treats `sample_name` as part of the
implicit row identity when present, matching `to_wide()` grouping for
multi-sample predictor outputs.

The combine docs now spell out the recommended allele-grid strategy:
split NetMHCpan-style per-allele predictors can be combined under
`coverage="complete"`, while intentionally sparse grids such as
MHCflurry haplotype-mode presentation should use `coverage="partial"`
and the ranking DSL's `best_*_allele` accessors for allele attribution.

`TopiaryResult` now treats long/wide representation as an internal,
cached view concern. Results expose `long_df` and `wide_df` on demand,
`to_long()` / `to_wide()` return results with that active `df` view,
and `topiary.stack_results()` normalizes mixed-form TopiaryResults
internally rather than requiring callers to pre-convert them.

Result merging now has user-facing names for the two distinct operations:
use `stack_results` / `result.stack_with(...)` when inputs are independent
result sets (files, samples, cohorts), and use `combine_predictions` /
`result.combine_predictions(...)` when inputs are complementary predictor
outputs for the same logical identity grid.

## 5.16.1

**pirlygenes 5.1.0 integration:**

`tissue_expressed_gene_ids` and `available_tissues` in
`topiary.sources` now call the typed
`pirlygenes.pan_cancer_expression()` accessor introduced in
pirlygenes 5.1.0 instead of indexing
`load_all_dataframes_dict()["pan-cancer-expression.csv"]`. pirlygenes
5.0.x had stripped the expression CSVs entirely, which broke the
tissue-exclusion path with a `KeyError` for anyone on that release
line; the 5.1.0 restore puts the data back next to the curated gene
lists, and the new accessor is the canonical handle.

`_check_pirlygenes` now also enforces `pirlygenes>=5.1.0` so a stale
install surfaces a clear upgrade message rather than a downstream
`AttributeError`. pirlygenes remains an optional dependency — only
the CTA / tissue-exclusion paths require it.

## 5.16.0

**pVACseq report loader (#94):**

`topiary.read_pvacseq(path)` parses both pVACtools output flavors into
long form: aggregated (`*.all_epitopes.aggregated.tsv`, one row per
variant) and the unaggregated `*.all_epitopes.tsv` (one row per
candidate peptide × allele × length). Format and MHC class are
auto-detected; the Median MT IC50 / percentile populate
`value` / `percentile_rank` with `prediction_method_name="pvacseq"`,
and WT IC50 / percentile populate the `wt_*` schema so DSL expressions
like `wt.Affinity.value` and `Affinity.value - wt.Affinity.value` work
without further setup.

For missense aggregated rows the WT peptide sequence is reconstructed
from `Best Peptide` + `Pos` + `AA Change` (the aggregated TSV doesn't
ship the WT sequence). Indel / frameshift / multi-residue rows leave
`wt_peptide` NaN; users wanting full WT context for those should load
the unaggregated `all_epitopes.tsv` flavor, which carries
`WT Epitope Seq` directly.

Per-algorithm score columns in the all_epitopes flavor (e.g.
"NetMHCpan MT IC50 Score", "MHCflurry WT Percentile") pass through as
snake_cased `pvacseq_<algo>_{ic50,pct}_{mt,wt}` annotation columns,
reachable via `Column("...")`. They are not melted into separate
`prediction_method_name` rows, so the DSL's `Affinity['netmhcpan']`
selector won't find them — callers wanting per-algorithm DSL access
should melt them out themselves or re-predict via `TopiaryPredictor`.

Multiple files (MHC-I + MHC-II, or a mix of flavors) compose through
`topiary.stack_results([read_pvacseq(p1), read_pvacseq(p2)])`; no dedicated
multi-file entry point is exposed.

Loader-derived columns aligned with `TopiaryPredictor` output so
downstream consumers (vaxrank, etc.) don't have to special-case the
loader source:

- `mhc_class` (`"I"` / `"II"`) — derived from the allele string;
  lets stacked MHC-I + MHC-II results be filtered or split by class.
- `contains_mutant_residues` (boolean) — true iff the row's mutation
  position falls inside the candidate peptide; false for flanking-only
  peptides that pVACseq scored but where the mutation lies outside.
- `mutation_start_in_peptide` / `mutation_end_in_peptide` (Int64,
  0-based half-open) — derived from pVACseq's 1-based Pos / Mutation
  Position.
- `source` — per-row provenance label, matching `read_tsv`
  convention; keeps multi-file stacks distinguishable without rooting
  through `Metadata.sources`.

`Metadata.extra["kind_support"]` mirrors `TopiaryPredictor.kind_support`
shape (`model_key -> {kind -> {mhc_dependence, mhc_class}}`) so the
loaded result can be passed straight to `apply_filter(... ,
kind_support=r.extra["kind_support"])` / `evaluate_scores(...)` without
constructing a parallel metadata dict.

`melt_pvacseq_algorithms(result)` expands the all_epitopes flavor's
per-algorithm `pvacseq_<algo>_<field>_<mtwt>` columns into separate
`prediction_method_name=<algo>` rows so the DSL's
`Affinity['mhcflurry'].value` / `Affinity['netmhcpan'].score` selectors
reach individual scoring algorithms natively.  The Median rows
(`prediction_method_name="pvacseq"`) are preserved; melt is a no-op on
aggregated input.

**DSL: categorical equality / membership for string columns:**

The filter DSL gains `Column.eq(value)`, `Column.ne(value)`, and
`Column.isin(values)` methods, plus a new `IsIn` node, so `mhc_class`,
`source`, `gene`, and other non-numeric columns are filterable
natively without pandas-side pre-masking:

```python
apply_filter(df, Column("mhc_class").eq("I"))
apply_filter(df, Column("mhc_class").isin(["I", "II"]))
apply_filter(df, (Affinity.value <= 500) & Column("mhc_class").eq("I"))
```

The string parser accepts string literals on the right-hand side of
`==` and `!=` (still rejected with `<` / `<=` / `>` / `>=`):

```python
parse('mhc_class == "I"')
parse('affinity.value <= 500 & mhc_class != "II"')
```

`IsIn` reads its column raw (bypasses `Column`'s float cast), so any
dtype works.  `DSLNode.__eq__` is intentionally *not* overridden —
nodes stay hashable for sets/dicts — these methods are the supported
path for categorical equality.

Why: vaxrank's typical filter shape combines numeric clauses with
class-I/class-II / provenance discriminators in one expression.
Pre-5.16.0 the categorical clause had to be applied as a pandas mask
before `apply_filter`; now both clauses compose in one DSL expression.

**Pre-built MHC-class filters:**

`topiary.class_i` and `topiary.class_ii` are pre-built `IsIn` nodes
referencing the `mhc_class` column.  Compose like any other DSL node:

```python
apply_filter(df, class_i & (Affinity.value <= 500))
apply_filter(df, class_i | class_ii)
```

Both require the `mhc_class` column (present after `read_pvacseq`,
absent from fresh `TopiaryPredictor` output where class lives in
`kind_support` at the model level).  For freshly predicted DataFrames,
`topiary.derive_mhc_class(allele_series)` returns a Series of `"I"` /
`"II"` / `pd.NA` derived from allele strings — assign it to
`df["mhc_class"]` and the shortcuts work.

## 5.15.0

**Backfill `value` from `score` for [0, 1]-score predictor kinds (#165):**

`TopiaryPredictor.predict_from_named_sequences` (and the rest of the
predict pipeline) used to leave `value` as `NaN` for prediction kinds
whose primary output is the [0, 1] score itself — most visibly
`pMHC_presentation`, but also `antigen_processing` and the other kinds
mhctools models without a distinct unit. Downstream consumers reading
`value` uniformly across kinds tripped on the resulting NaNs: strict
`simplejson.dumps` rejects them, `sorted()` is undefined on them, and
arithmetic on `value` silently propagated NaN. Vaxrank just hit this in
3.0.1 and worked around it in 3.0.2.

A new `_backfill_value_from_score` helper populates `value` from `score`
for any row whose `kind` is *not* in `mhctools.VALUE_BEST_DIRECTIONS`
(i.e. whose `value` has no distinct unit). Unit-bearing kinds —
`pMHC_affinity` (IC50 nM) and `pMHC_stability` (half-life) — are
explicitly skipped, so a NaN `value` on those kinds stays NaN rather
than being silently misrepresented as a [0, 1] score. The helper is
applied uniformly across the producer surface:

- `TopiaryPredictor._format_prediction_df` (the main predict path)
- `CachedPredictor._bindings_to_dataframe` and
  `_predictions_to_dataframe` (cached-predictor outputs, including
  mhcflurry's class1_presentation pipeline and NetMHCpan `-BA`)

`affinity` continues to be populated only for `pMHC_affinity` rows, so
that column's semantics are unchanged.

After this change the long-format schema is uniform: `value` is the
predictor's primary numeric output for every row (IC50 for affinity,
probability for presentation, ...) and `score` is the [0, 1] ranking
score (equal to `value` for presentation, derived from IC50 for
affinity).

## 5.14.1

Raise the varcode floor to `>=4.18.0`, the first varcode release that
drops PyVCF3 from the runtime import path. Together with topiary's
existing lazy varcode imports (5.10.7) this closes #122: a fresh
install no longer surfaces rpy2 / embedded-R noise when importing
varcode or running varcode-dependent topiary workflows.

## 5.14.0

**Peptide properties as DSL nodes (#95):**

The peptide-property registry (`Charge`, `Aromaticity`,
`Hydrophobicity`, `MolecularWeight`, plus the manufacturability and
immunogenicity entries) now exposes each property as a DSL node, so
ranking expressions can mix peptide-intrinsic properties with the
existing kind accessors:

    score = (
        Affinity.value.norm(mean=500, std=200)
        + 0.1 * Aromaticity.clip(lo=0, hi=3)
        - 0.1 * abs(Charge)
    )

The string parser recognizes property names as atoms in both bare and
scoped positions: `parse("charge >= 0 & aromaticity <= 3")`,
`parse("wt.hydrophobicity")`. Property nodes always recompute from
the `peptide` column, so they work on any predictions frame without
calling `add_peptide_properties` first.

## 5.10.8

**Restore Varcode CLI parser delegation:**

- Reverted Topiary's local copy of Varcode variant CLI arguments so
  `topiary.cli.args` again delegates `add_variant_args` and
  `variant_collection_from_args` to `varcode.cli`.
- Kept the narrower lazy imports for non-CLI Varcode helpers, so plain
  `import topiary` still avoids loading Varcode.

## 5.10.7

**Lazy Varcode imports (#122):**

- Topiary's CLI now registers Varcode-compatible variant arguments locally
  and imports Varcode only when the variant-loading pipeline runs.
- Moved Varcode imports in filtering and protein-change helpers into the
  functions that actually need Varcode objects.
- Added import-guard tests so `import topiary` and direct CLI parser setup do
  not load Varcode or its VCF dependency stack.

## 5.10.6

**CachedPredictor index memory footprint (#134):**

- `CachedPredictor` now stores row positions in its internal key index
  instead of duplicating each cached row as a Python dictionary.
- Added a prefix index for `(peptide, allele, peptide_length)` lookups
  so cache hits no longer scan every full row key.

## 5.10.5

**Deploy script interpreter selection (#149):**

- `deploy.sh` now uses one configurable Python interpreter throughout
  the release flow (`PYTHON=${PYTHON:-python3}`), including version
  detection, the PyPI version check, build, and twine upload.
- The build step removes both `dist/` and `build/` before packaging so
  stale local build outputs cannot shadow the PyPA `build` module.

## 5.10.4

**DSL version syntax cleanup:**

- Removed the experimental `model:version:kind` string form, e.g.
  `mhcflurry:release-2.2.0:ba.score`. Use bracketed model versions
  instead: `mhcflurry[release-2.2.0]:ba.score`.
- Kept the compact dash form for numeric-leading versions, e.g.
  `mhcflurry-4.1b:affinity.score`.
- Documented the recommended string DSL syntax separately from accepted
  compatibility aliases.

## 5.10.3

**Quote-free model-qualified kind syntax (#150):**

- The string DSL now accepts `mhcflurry:affinity`,
  `affinity:mhcflurry`, and `mhcflurry.affinity` as aliases for
  `affinity[mhcflurry]` / `affinity['mhcflurry']`.
- Model-first syntax can attach versions directly to the model:
  `mhcflurry[2.1.5]:ba.score` parses like
  `affinity[mhcflurry, 2.1.5].score`. Version slots accept common
  unquoted labels such as `4.1b`, `v2.1`, `release-2.2.0`, and
  `2.2.1+release-2.2.0`.
- Colon-separated version forms are also accepted:
  `mhcflurry:release-2.2.0:ba.score` and
  `mhcflurry-4.1b:affinity.score`.
- These aliases compose with fields, transforms, and scopes, e.g.
  `netmhcpan:ba.score`, `mhcflurry[2.1.5]:processing.score`, and
  `wt.mhcflurry[2.1.5]:ba.score`.
- Existing bracket and underscore forms remain supported and canonical
  string output still emits bracket form for deterministic round trips.

## 5.10.2

**Wildtype MHC scoring (#123):**

- `TopiaryPredictor(predict_wt=True)` now scores populated
  `wt_peptide` values with the configured MHC model(s) and attaches
  `wt_value`, `wt_score`, `wt_affinity`, `wt_percentile_rank`,
  `wt_prediction_method_name`, and `wt_predictor_version`.
- WT predictions are joined back by allele, peptide length, prediction
  kind, method, and version so affinity and presentation rows stay
  aligned.
- Rows without a length-compatible WT peptide keep NaN `wt_*`
  prediction values.
- The CLI now exposes the same behavior with `--predict-wt`, enabling
  `wt.*` sort expressions on variant-derived outputs.
- WT scoring uses the baseline protein context, not isolated peptide
  scoring, so context-sensitive predictors keep the correct flanks.

## 5.10.1

**CLI validation errors:**

- Bare `topiary` invocations and other missing-argument validation
  failures now render as normal argparse errors with usage text instead
  of printing the full parsed namespace followed by a traceback.
- Missing prediction requests report both required parts: an MHC source
  (`--mhc-predictor` or cached predictions) and an input source.

## 5.10.0

**`evaluate_scores(df, node)` — row-aligned DSL helper (#126):**

- New `topiary.evaluate_scores(df, node, group_keys=None, fill=nan)`
  evaluates a DSL node against a DataFrame and returns a Series
  aligned 1:1 with `df.index` (one value per row, broadcast from the
  peptide-allele group).
- Replaces the four-line pattern every DSL consumer wrote by hand:
  build `EvalContext`, call `node.eval(ctx)` (indexed by
  `ctx.group_index`), map via `ctx.row_group_tuples()` to row
  alignment, attach `df.index`.
- `fill` controls NaN behavior for rows whose group wasn't scored —
  default `NaN`, override with `0.0` for additive scoring, `-inf`
  for ranking, etc.
- Exported from `topiary.ranking` and as `topiary.evaluate_scores`.

**Bare identifier inside kind-qualified brackets (#119):**

- The string DSL now accepts `affinity[netmhcpan]` and
  `affinity[netmhcpan, "4.1b"]` in addition to the previously required
  `affinity['netmhcpan']` / `affinity['netmhcpan', '4.1b']`.  Bare
  identifiers read more cleanly in YAML configs, where
  `"affinity[netmhcpan] <= 500"` drops the nested-quote gymnastics of
  `"affinity['netmhcpan'] <= 500"`.
- Non-IDENT values (e.g. a version string `"4.1b"` that starts with a
  digit or contains a dot) still require quotes.
- Fully backwards compatible — every previously-valid expression
  still parses.  `to_expr_string()` continues to emit the
  single-quoted canonical form, so round-trips stay deterministic.

**Filter-context auto-aggregation across methods (#118):**

- Inside `apply_filter` (and `TopiaryPredictor(filter_by=...)`), an
  unqualified kind reference (`Affinity <= 500`, `presentation.rank <=
  2.0`, ...) no longer raises `Ambiguous: multiple models produce ...`
  when the DataFrame has multiple `prediction_method_name` values for
  that kind. Instead, the comparison is evaluated per method and
  combined via `nanmin` (for `<`/`<=`) or `nanmax` (for `>`/`>=`) —
  the "any method passes" interpretation.
- Scope is narrow on purpose: only directional `Comparison` nodes
  (`<`, `<=`, `>`, `>=`), only when evaluated under a filter context
  (`apply_filter` sets `EvalContext.filter_context=True`), only when
  all unqualified refs in the comparison are the same kind. `==` /
  `!=`, `apply_sort`, scalar score expressions, and cross-kind
  comparisons (`affinity.rank <= processing.rank`) keep the strict
  ambiguity error.
- `EvalContext(df, filter_context=True)` is now public — callers who
  hand-roll eval outside `apply_filter` can opt in explicitly.
- Single-method frames take the strict path (no behavior change).
- Complements `EvalContext(default_methods=...)` (#140): if a default
  resolves the unqualified ref, `Field.eval` returns before this
  branch runs, so `default_methods` takes precedence.

**EvalContext `default_methods` for multi-predictor frames (#140):**

- `EvalContext(df, default_methods={...})` — resolve unqualified
  `Affinity` / `Presentation` / etc. references when a DataFrame has
  multiple `prediction_method_name` values for the same kind. Without
  it, the ambiguity check still raises (behavior unchanged).
- Keys accept canonical kind names (`"pMHC_affinity"`), DSL short names
  (`"affinity"`, `"ba"`, `"el"`, ...), or mhctools `Kind` constants.
- `apply_filter(df, node, default_methods=...)` and
  `apply_sort(df, nodes, default_methods=...)` forward the kwarg.
- Error message on ambiguous unqualified access now points users at
  `default_methods` as the opt-in escape hatch.

Context: multi-predictor pipelines (e.g. LENS emitting MHCflurry
+ netMHCpan + netMHCstabpan) previously had to either qualify every
DSL expression with `['modelname']` or pre-subset the DataFrame to one
method per kind. `default_methods` is the declarative escape hatch;
filter-context auto-agg is the pragmatic default for filter
top-levels, while sort and score stay strict to avoid silent
semantics on compound arithmetic like `0.5*ba.score + 0.5*el.score`.

## 5.9.0

**SelfProteome part B (#138):**

- Renamed `scope=` → `include=` on `from_ensembl` / `from_fasta`.
- `include="protected_tissues"` — filters to genes expressed in named
  tissues. Human defaults via pirlygenes/HPA; any species via explicit
  `tissue_gene_ids=` set.
- BLOSUM62 distance metric (default). Conservative substitutions
  (I↔L) produce lower distances than non-conservative (I↔W). Loaded
  lazily from Biopython. `metric="hamming"` kept as opt-in.
- 1aa indel candidates (`include_indels=True`, default). Checks
  deletion (L-1) and insertion (L+1) neighbors via hash-set lookup.
  Indel at edit_distance=1 beats substitution at edit_distance≥2.
- 39 self_proteome tests (up from 25).

## 5.8.0

**New feature — `SelfProteome` for cross-reactivity analysis (#135, part A of #124):**

- `SelfProteome` class holds a species-tagged, scope-filtered reference
  protein corpus indexed by peptide length.  Answers per-query
  nearest-neighbor lookups: "given this mutant peptide, what's the most
  similar peptide in healthy human self?"
- Constructors: `from_ensembl(species, release, scope=...)`,
  `from_fasta(path)`, `from_peptides(dict)` (test/programmatic use).
- `scope="all"` (whole proteome) and `scope="non_cta"` (default for
  human — strips cancer-testis-antigen genes via pirlygenes).
  Non-human species must supply `cta_source=` explicitly; unsupported
  combos raise with actionable messages.
- SIMD-vectorized Hamming-distance search against int8-encoded
  reference arrays, chunked for memory bound.  Substitutions only
  in this release.
- `TopiaryPredictor(self_proteome=ref)` integration — `self_nearest_*`
  columns (`self_nearest_peptide`, `_peptide_length`, `_edit_distance`,
  `_gene_id`, `_transcript_id`, `_reference_offset`,
  `_reference_version`) attached before filter/sort so DSL expressions
  can reference them.
- Composite `reference_version` property captures species + release +
  scope identity; custom CTA filters hash into the string for
  reproducibility.
- New `docs/self_proteome.md` page + nav entry.

**Deferred to parts B/C (#124):**

- `scope="protected_tissues"` (HPA / GTEx tissue filtering).
- 1aa insertion / deletion candidates.
- BLOSUM62-weighted distance metric.
- `self_mimic_*` / `self_strongest_nearby_*` binding-aware axes.
- `self_nearest_candidates` structured column.
- Seed-and-extend indexing algorithm (benchmark-driven).
- Bundled self-proteome × common-HLA prediction artifacts.

**Tests:** 25 new in `tests/test_self_proteome.py`.  Full suite 1166
passed (up from 1141).

## 5.7.0

**CachedPredictor — CLI, multi-kind, flanks, NetMHC fixtures (#136).**

**CLI support for cached predictions:**

- New `--mhc-cache-file` / `--mhc-cache-directory` CLI arguments let
  users run topiary entirely from pre-computed prediction files without
  invoking a live MHC predictor.
- `--mhc-cache-format` is optional — topiary sniffs the format from
  file content (NetMHC-family preamble lines, mhcflurry column names,
  topiary-output schema, Parquet magic bytes).  Only the generic `tsv`
  format requires an explicit flag.
- `--mhc-predictor` and `--mhc-alleles` become optional when a cache
  supplies predictions.

**Multi-kind cache (closes #137):**

- Cache index expanded from `(peptide, allele, peptide_length)` to a
  6-tuple `(peptide, allele, peptide_length, kind, n_flank, c_flank)`.
  A single cache holds every kind a predictor emits — mhcflurry's
  class1_presentation pipeline (affinity + presentation + processing),
  NetMHCpan `-BA` (affinity + presentation), etc.  No more silent
  data loss from single-kind heuristics.
- `from_mhcflurry` explodes wide-format CSVs into one row per
  `(peptide, allele, kind)`, preserving `n_flank` / `c_flank` /
  `source_sequence_name` / `peptide_offset` / `sample_name` per kind.
- `from_netmhcpan_stdout` switches to `parse_netmhcpan_to_preds`
  (mhctools' multi-kind API), returning all kinds instead of collapsing
  by mode.  Dropped `mode=` kwarg (no longer meaningful).
- Generic TSV loader (`from_tsv`) now requires a `kind` column per row.
  Multi-kind TSVs work natively — add a kind column and list one value
  per row.

**Multi-allele NetMHC parsing fixed:**

- `parse_netmhcpan_to_preds` handles multi-allele stdout correctly
  (per-allele header lines that crashed the old
  `parse_netmhcpan_stdout` are no longer an issue).  Multi-allele
  fixtures promoted from xfail to happy-path tests.

**Flank sensitivity in cache key:**

- `n_flank` / `c_flank` are now part of the composite key.  mhcflurry's
  processing and presentation predictions depend on flanking residues;
  the same peptide at different protein positions can produce different
  scores.  Absent flanks normalize to empty string `""` (no None/NaN
  handling quirks).

**Real NetMHC-family fixtures:**

- `tests/data/netmhc_fixtures/` — captured from netmhc-bundle binaries
  for peptide SLLQHLIGL at HLA-A*02:01 / A*24:02 / B*07:02.
  NetMHCpan 4.0 + 4.1, NetMHC 4.0, NetMHCstabpan, single-allele +
  multi-allele variants.  6 real-fixture tests pin actual numeric
  predictions through the loaders.

**README reorder:**

- "MHC prediction models" moved near the top (after "Predicting MHC
  binding"); "Cached predictions" moved to the end.

**Tests:**

- 1141 tests pass (up from 1111 in v5.6.0).  9 new CLI integration
  tests, 6 real-fixture tests, 3 promoted multi-allele happy-path
  tests, multi-kind + multi-flank regression tests.

## 5.6.0

**Closes #128 — `CachedPredictor` reaches feature-complete.**

**New loaders for the DTU NetMHC suite (#132):**

- `CachedPredictor.from_netmhcpan_stdout(path, mode=…)` — auto-detects
  NetMHCpan 2.8 / 3 / 4 / 4.1. `mode` selects `"binding_affinity"` or
  `"elution_score"` for 4+.
- `CachedPredictor.from_netmhc_stdout(path, version=…)` — classic
  NetMHC 3 / 4 / 4.1.
- `CachedPredictor.from_netmhcpan_cons_stdout(path)` — NetMHCcons.
- `CachedPredictor.from_netmhciipan_stdout(path, version=…)` —
  NetMHCIIpan legacy / 4 / 4.3.
- `CachedPredictor.from_netmhcstabpan_stdout(path)` — NetMHCstabpan
  pMHC-stability predictor.

Each loader wraps an existing `mhctools.parsing.*_stdout` function
(zero new parsing code) and parses the tool version out of the
stdout preamble onto `predictor_version`. Parses stdout text, not
the `-xlsfile` tab-delimited variant — flagged in `docs/cached.md`.

**Sharding — `concat` + `from_directory`:**

- `CachedPredictor.concat([caches], on_overlap=…)` — merge several
  caches into one. All shards must share `(name, version)` per the
  core invariant.
- `CachedPredictor.from_directory(path, pattern="*", on_overlap=…)` —
  glob a directory and concat every matching file through
  `from_topiary_output`.
- Overlap resolution policies (`on_overlap`): `"raise"` (default — fail
  if any `(peptide, allele, peptide_length)` appears in more than one
  shard), `"last"` (later shard wins), `"first"` (earlier wins), or a
  user-supplied `callable(row_a, row_b) -> row` resolver.

**Polish from vaxrank-consumer review on #130 (#131):**

- `_fallback_resolve` filters fallback output to keys not already in
  the index before merging, so a partial-allele cache (peptide P
  present for allele A, missing for B) doesn't see its `(P, A)` row
  silently overwritten by the fallback's all-alleles response.
- Class docstring now flags silent peptide-length lock-in and
  non-thread-safety.
- `save()` raises on an empty never-queried cache with no identity,
  so users don't write schema-only files that can't be round-tripped.

**Tests:**

- 59 tests in `tests/test_cached_predictor.py` (up from 41): 6 NetMHC
  loader tests, 12 sharding tests. Full suite 1111 passed (up from
  1093).

## 5.5.0

**New feature — `CachedPredictor`:**

- Pluggable prediction source (part 1 of #128) that loads MHC binding
  predictions from a pre-computed table and plugs into
  `TopiaryPredictor(models=…)` alongside live mhctools predictors.
  Use cases: reproducibility, iterating on filters/ranking without
  rerunning the predictor, per-allele / per-sample parallel
  predictions, ingesting output from tools topiary doesn't natively
  run.
- Loaders shipped: `CachedPredictor.from_dataframe`,
  `from_topiary_output` (Parquet / TSV), `from_tsv` (generic with
  column mapping), `from_mhcflurry` (maps `mhcflurry_*` columns onto
  canonical names).
- NetMHCpan / NetMHC / NetMHCstabpan / NetMHCIIpan / NetMHCcons
  loaders are queued for a follow-up PR.
- Sharding (`concat` / `from_directory`) is queued for a separate
  follow-up.

**Version invariant:**

- A single `CachedPredictor` holds exactly one
  `(prediction_method_name, predictor_version)` pair; `None` / `NaN`
  / empty-string values are rejected at construction. Mixing versions
  would produce outputs that pass downstream filters invisibly, so
  the invariant is enforced everywhere (load, fallback attach, concat).
- Explicit opt-in equivalence: pass `also_accept_versions={"…", …}`
  when two labels really are interchangeable (rc → final, timestamp-
  only model-data reflashes).

**mhcflurry-specific version composition:**

- New `topiary.mhcflurry_composite_version()` helper discovers the
  locally-installed mhcflurry package version plus its active model
  release and returns a composite string like `"2.2.1+release-2.2.0"`.
  `CachedPredictor.from_mhcflurry(path)` uses it automatically when
  no explicit `predictor_version` is passed — users never enumerate
  model bundles manually.

**Fallback mode:**

- Pass `fallback=<live_predictor>` to delegate cache misses; results
  are merged back into the cache so subsequent queries serve locally.
  No separate flag — caching fallback hits is always right for the
  batch-prediction workload.
- Pure read-through: `CachedPredictor(fallback=p)` with no df starts
  empty; identity is discovered from the fallback's first output.

**Documentation:**

- New `docs/cached.md` covering the full surface.
- `CachedPredictor` section added to `docs/api.md`.
- Subsection in `docs/quickstart.md`.
- README has a top-level "Cached predictions" section.
- Feature list in `docs/index.md` updated.

**Tests:**

- 38 new tests in `tests/test_cached_predictor.py` (up from 0),
  covering construction, version invariant (mixed rows, null rejection,
  name/version round-trip as string), predict_peptides +
  predict_proteins sliding-window, fallback hit + miss + version
  mismatch + empty-cache identity discovery, `also_accept_versions`,
  all four loaders, `mhcflurry_composite_version` via stubbed
  mhcflurry module (no tensorflow/libomp collisions), and
  integration with `TopiaryPredictor(models=cache)`.
- Full suite: 1090 passed (up from 1052), 3 skipped.

**Related upstream issue:**

- Filed `openvax/mhctools#193` — `predict_peptides_dataframe` misses
  `predictor_version` / `kind` / `value` columns returned by
  `predict_proteins_dataframe`. `CachedPredictor` currently backfills
  the gap internally; can simplify once the mhctools asymmetry is
  resolved.

## 5.4.0

**Breaking rename (no back-compat alias):**

- `AntigenFragment` → `ProteinFragment`. Describes what the object is
  (a slice of some protein — natural, chimeric, foreign, or designed)
  rather than what it's used for. Matches Isovar's convention.
- `topiary/antigen.py` → `topiary/protein_fragment.py`;
  `topiary/io_antigen.py` → `topiary/io_protein_fragment.py`;
  `docs/antigens.md` → `docs/fragments.md`.
- `TopiaryPredictor.predict_from_antigens(fragments)` →
  `predict_from_fragments(fragments)`.
- `read_antigens` / `write_antigens` / `iter_antigens` →
  `read_fragments` / `write_fragments` / `iter_fragments`.

**Downstream migration checklist:**

- `from topiary import AntigenFragment` → `from topiary import ProteinFragment`.
- `from topiary.antigen import …` / `from topiary.io_antigen import …` →
  `from topiary.protein_fragment import …` /
  `from topiary.io_protein_fragment import …`.
- `predictor.predict_from_antigens(fragments)` →
  `predictor.predict_from_fragments(fragments)`.
- `topiary.read_antigens(path)` / `write_antigens(fragments, path)` /
  `iter_antigens(path)` → `read_fragments` / `write_fragments` /
  `iter_fragments`.
- TSV files written by 5.2.x `write_antigens` remain readable by
  5.4.0 `read_fragments`: the new `transcript_name` column is
  optional and defaults to `None` when missing.  TSVs written by
  5.4.0 are **not** readable by ≤5.2.x (the old reader rejects
  unknown columns).
- Unaffected surface: `TopiaryPredictor`, `EvalContext`, `apply_filter`,
  `predict_from_variants` / `predict_from_mutation_effects` / the
  legacy column contract.

**Refactor (predict_from_variants now builds on ProteinFragment):**

- `predict_from_mutation_effects` builds a list of `ProteinFragment`s
  from varcode effects (via the new `_fragment_from_effect` adapter)
  and delegates to a shared `_build_fragment_rows` step — one prediction
  pipeline instead of two. The ~60-line row-by-row metadata loop is gone.
- New fragment-derived columns (`fragment_id`, `source_type`,
  `overlaps_target`, `wt_peptide` / `wt_peptide_length`) now flow
  through the variant path alongside the legacy columns.
- Legacy column contract preserved: absolute `peptide_offset`,
  `mutation_start_in_peptide` / `mutation_end_in_peptide`,
  `transcript_name`, `contains_mutant_residues`, `only_novel_epitopes`,
  and legacy `gene_expression_dict` / `transcript_expression_dict`
  plumbing all behave identically to 5.2.0.
- `source_type` classification aligned with `docs/fragments.md`
  vocabulary: `PrematureStop` → `variant:stop_gain`, multi-residue
  `Substitution` → `variant:indel`, unlisted effect classes fall back
  to `variant:<classname_lowered>`.
- Filter / sort now run after `peptide_offset` rebasing on the variant
  path, so filter expressions referencing `peptide_offset` see absolute
  protein coordinates (matches 5.1.x behavior).

**New field:**

- `ProteinFragment.transcript_name` — human-readable transcript label
  alongside `transcript_id`. Threaded through `from_dict`, `from_variant`,
  `from_junction`, and the TSV IO schema.

**Internal:**

- New `TopiaryPredictor._build_fragment_rows(fragments)` — fragment
  scanning + metadata overlay without filter / sort.  Public entry
  points layer filter / sort / `only_novel_epitopes` on top.
  Underscore-prefixed annotation keys are reserved for internal
  plumbing and never surface as DataFrame columns.
- 18 new regression tests covering legacy column contract, expression-
  dict plumbing, and the effect→fragment source_type classifier —
  including a parametrized grid pinning every entry of the documented
  `source_type` vocabulary (`variant:snv`, `variant:indel`,
  `variant:frameshift`, `variant:stop_gain`, `variant:stop_loss`,
  `variant:start_loss`, `variant:exon_loss`, `variant:alternate_start`,
  plus the `variant:<classname_lowered>` fallback).
- `tests/test_frameshift_fragments.py` — new regression suite (75
  cases) pinning `_fragment_from_effect` behavior on varcode
  `FrameShift` / `FrameShiftTruncation` effects: target_intervals
  span the full downstream novel tail, per-peptide `overlaps_target`
  agrees with ground truth across peptide lengths 8–11,
  `inframe=True`/`False` produce identical intervals for frameshift
  shapes, and `only_novel_epitopes=True` preserves every downstream
  9-mer.

## 5.2.0

**New features (core abstraction for antigens from any origin):**

- `AntigenFragment` — a universal record for a protein/peptide sequence
  with source-type, target-region, and comparator metadata. Carries
  variants, structural variants, ERVs, CTAs, viral proteins, allergens,
  autoantigens, and synthetic constructs through one pipeline. Free-form
  `source_type` tag (recommended vocabulary documented, not enforced);
  `target_intervals: list[tuple[int, int]]` for disjoint regions
  (breakpoints of tandem duplications, non-self regions of ERVs, etc.);
  `reference_sequence` + `germline_sequence` with germline-precedence
  `effective_baseline`. Equality/hash keyed on `fragment_id` (stable
  human-readable prefix + SHA-1 hash). Convenience constructors
  `from_variant`, `from_junction`. Stdlib-only serialization:
  `to_dict` / `from_dict` / `to_json` / `from_json`.
- `topiary.read_antigens(path)` / `write_antigens(fragments, path)` /
  `iter_antigens(path)` — TSV IO with JSON-serialized list/dict columns.
- `TopiaryPredictor.predict_from_antigens(fragments)` — new entry point
  that scans each fragment's sequence, propagates every fragment field
  (including arbitrary annotations) onto prediction rows, threads
  `fragment_id` through for downstream grouping (vaxrank vaccine-window
  selection), and emits an `overlaps_target` column computed from each
  peptide's position vs. the fragment's target intervals. Backwards-compat
  `contains_mutant_residues` alias for `source_type` prefixed with
  `variant`. `wt_peptide` derived by slicing `effective_baseline`;
  model-side WT predictions deferred to a follow-up PR.
- `self_nearest` — reserved DSL scope for cross-reactivity filtering
  ("closest peptide in essential healthy tissues"). Topiary does not
  compute these columns — producers populate via BLAST / edit distance
  against a healthy-tissue proteome with their own "self" definition.
  The scope reads `self_nearest_*` columns when present, returns NaN
  otherwise. See `docs/antigens.md` for the reserved column namespace.
- `fragment_id` is now preferred over `variant` as the group key in the
  DSL's group-by logic (falls back to `variant`, then
  `source_sequence_name`).

**Internal:**

- New module `topiary/antigen.py` (dataclass + helpers) and
  `topiary/io_antigen.py` (TSV IO).
- 63 new tests covering identity, serialization, geometry,
  `predict_from_antigens` propagation, `self_nearest` scope reads.

## 5.1.0

**New features:**

- `topiary.read_lens(path)` — load LENS (Landscape of Effective
  Neoantigens Software) reports into Topiary's wide-form schema.
  Handles the three observed schema variants (v1.4, v1.5.1, v1.9-dev)
  with column-based version detection. Binding columns are remapped to
  `{model}_{kind}_{field}`; per-model versions populate
  `Metadata.models`. LENS-specific columns (`erv_*`, `priority_score_*`,
  `b2m_*`, `hla_allele_*`, etc.) pass through as annotations and remain
  accessible via `Column("…")` in the DSL. See
  [#110](https://github.com/openvax/topiary/issues/110). Known losses:
  `peptide_offset` set to 0 (LENS doesn't record it);
  `contains_mutant_residues` / `mutation_start_in_peptide` left NaN
  (LENS's `mut_aa_pos` semantics are ambiguous); `n_flank` / `c_flank`
  derived from `pep_context` only for SNV / SPLICE / FUSION.
- `DSLNode.logistic_normalized(midpoint, width)` — logistic sigmoid
  rescaled to reach 1 as `x → -∞`, so the output is a proper
  `[0, 1]` score.  `.logistic(...)` is unchanged.
  ([#116](https://github.com/openvax/topiary/issues/116))
- Allele normalization uses `mhcgnomes` unconditionally (Class I,
  Class II, mouse all supported).

## 5.0.1

Polish pass on the v5.0.0 DSL refactor — no user-visible behavior
changes, just internal cleanup.

- `DSLNode.child_nodes()` — new abstract method on every node type.
  Generic tree walkers (column validation, future AST rewriters) no
  longer need a per-node `isinstance` ladder.  `_collect_column_names`
  now uses it.
- The scoped-field filter guard moves from four per-operator overrides
  on `Field` (`__le__` / `__ge__` / `__lt__` / `__gt__`) into a single
  check in `Comparison.__init__`.  Same error, less surface area.
- `apply_filter` now reindexes the evaluated Series to
  `ctx.group_index` before masking, so an index mismatch surfaces as
  NaN → False rather than as misaligned row selection.

## 5.0.0

**Breaking changes (DSL refactor,
[#111](https://github.com/openvax/topiary/issues/111)):**

- Filter leaves (`EpitopeFilter`, `ColumnFilter`, `ExprFilter`) and the
  composite (`RankingStrategy`, `SortSpec`) are removed. Every DSL
  expression is now a single `DSLNode` tree whose `.eval(ctx)` returns
  a `pandas.Series` indexed by peptide-allele group tuples.
- `Affinity <= 500` (and friends) now returns a `Comparison` node;
  `A | B` / `A & B` returns a `BoolOp`. Both classes inherit the full
  arithmetic operator set, so boolean-as-number composition
  (`(Affinity <= 500) * Affinity.score`) is allowed.
- `apply_ranking_strategy` is split into `apply_filter(df, node)` and
  `apply_sort(df, sort_nodes, sort_direction="auto")`.
- `parse_ranking`, `parse_filter`, `parse_expr` are collapsed into a
  single `parse()` that returns a `DSLNode`. The parser uses standard
  precedence for `&` / `|` (`&` binds tighter); mixed-operator strings
  are now accepted.
- `TopiaryPredictor` kwargs: `ranking_strategy`, `ranking`, `filter`,
  `rank_by`, `ic50_cutoff`, and `percentile_cutoff` are removed. Use
  `filter_by=` (a `DSLNode` or string) and `sort_by=` (a `DSLNode` or
  list).  The `TopiaryPredictor.ranking_strategy` property is replaced
  by the separate `.filter_by` / `.sort_by` attributes.
- `Field` gains an optional `version` parameter;
  `Affinity["netmhcpan", "4.1b"]` filters on both
  `prediction_method_name` and `predictor_version`.
- Ambiguity semantics tightened — unqualified `Affinity.value` on a
  DataFrame that contains multiple `prediction_method_name` values
  raises `ValueError` pointing at `Affinity["modelname"]`.
  Previously the old filter silently passed if *any* row satisfied the
  threshold.
- `apply_filter` now errors when the evaluated Series contains values
  outside `{True, False, 0, 1, 0.0, 1.0, NaN}`, pointing the user at
  `<=` / `>=`. NaN still maps to `False`.

**New:**

- `EvalContext`, `DSLNode`, `Const`, `Column`, `Field`, `BinOp`,
  `UnaryOp`, `NormExpr`, `SurvivalExpr`, `LogisticExpr`, `ClipExpr`,
  `AggExpr`, `Comparison`, `BoolOp` exported from `topiary.ranking`.
- `apply_filter`, `apply_sort`, `parse` exported as the top-level DSL
  entry points.
- Every `DSLNode` has a `to_expr_string()` that round-trips through
  `parse()`.

## 4.12.0

**Breaking changes:**

- `topiary.read_tsv` and `topiary.read_csv` now return a `TopiaryResult`
  instead of an `(DataFrame, Metadata)` tuple. Callers using tuple
  unpacking must migrate: `df, meta = read_tsv(path)` →
  `result = read_tsv(path); df, meta = result.df, result.metadata`.

**New features:**

- `TopiaryResult` class bundling a predictions DataFrame with provenance
  (model versions, source files, form, filter/sort history).  Delegates
  common DataFrame operations (`len`, `iter`, `columns`, `shape`, `head`,
  `iterrows`, etc.) so most existing DataFrame-style code continues to
  work.  Provides `to_wide()`, `to_long()`, `to_tsv()`, `to_csv()`,
  `filter_by()`, `sort_by()`.
- `topiary.stack_results([r1, r2, ...])` merges `TopiaryResult`s, unioning
  models (warns on version conflicts), concatenating sources, and
  preserving filter/sort history only if all inputs agree.
- `read_tsv` / `read_csv` accept a `tag=` kwarg to label the source of
  the loaded rows; defaults to the filename.  Auto-populates a `source`
  column on the DataFrame.
- `Metadata` gains a `sources: list[str]` field; the comment block
  supports multiple `#source=...` lines.

**Deprecations (removed in 5.0 alongside the DSL refactor,
[#111](https://github.com/openvax/topiary/issues/111)):**

- `EpitopeFilter`, `ColumnFilter`, `ExprFilter`, `RankingStrategy`
  replaced by a unified `Comparison` / `BoolOp` DSL tree. See the 5.0.0
  entry above for migration details.

## 4.9.0

- Require `mhctools>=3.7.0`.
- Rename CLI sorting flags to `--sort-by` and `--sort-direction`.
- Add Python API `sort_by=` and `.sort_by(...)`, while keeping `rank_by` as a compatibility alias.
- Treat comma-separated `--sort-by` keys as lexicographic tie breakers, with fallthrough on missing values.
- Document upstream `mhctools 3.7.0+` support for multi-predictor CLI invocations, the simplified `Kind` API, and the updated NetChop/Pepsickle behavior.
