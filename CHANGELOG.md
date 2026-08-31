# Changelog

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
