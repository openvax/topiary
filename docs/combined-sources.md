# Combine source tables and re-score on demand

Topiary combines normalized source tables, preserves their original values,
and uses the existing DSL to filter and rank reported candidates. Models run
only when explicitly requested. Raw reads and protein reconstruction are not
prerequisites for combining tables.

```python
from topiary import (
    combine_sources, read_lens, read_pvacseq, melt_pvacseq_algorithms,
    rank_candidates,
)

combined = combine_sources({
    "lens": read_lens("lens.tsv"),
    "pvacseq": melt_pvacseq_algorithms(read_pvacseq("all_epitopes.tsv")),
}, sample_name="patient-01")
ranked = rank_candidates(
    combined, "affinity['netmhcpan'].value", ascending=True,
    duplicates="best",
)
per_source = rank_candidates(
    combined, "affinity['netmhcpan'].value", ascending=True,
    duplicates="best", strata=["source_label", "candidate_mhc_class"],
)
combined.to_tsv("all-evidence.tsv")
ranked.to_csv("ranked-candidates.tsv", sep="\t", index=False)
```

Inputs may be `TopiaryResult` objects or normalized DataFrames. Existing readers
normalize LENS and pVACseq; [native Exacto tables](exacto.md) use `read_exacto`.
Already normalized ORF/RNA DataFrames are also supported. Aggregated reports contribute only the
candidates they actually report.

A single input needs no model name, version or per-row source metadata. For
example, this table can be ranked directly after combination:

```python
import pandas as pd
from topiary import combine_sources, rank_candidates

simple = combine_sources({"input": pd.DataFrame({
    "peptide": ["SIINFEKL"], "allele": ["HLA-A*02:01"],
    "kind": ["pMHC_affinity"], "value": [50.0],
})}, sample_name="patient-01")
ranked = rank_candidates(simple, "affinity.value", ascending=True)
```

Unknown predictor names and versions remain unknown. The ordinary expression
`affinity.value` continues to work; adding provenance does not require a
version-qualified expression for this simple case.

## Named selection policies

`SelectionPolicy` saves the exact candidate filter, score, model/version
selections, ranking direction, duplicate policy and strata under a stable name.
It has no implicit scientific recipe. **Vaxrank owns the published
`builtin:openvax-v1` bundle**, shipped with experimental overlays in
[Vaxrank 3.36.0 / PR #565](https://github.com/openvax/vaxrank/pull/565).
Topiary evaluates its policy subtree and tests against that released bundle;
it does not maintain a second definition. Give changed policies new names.

```python
from topiary import (
    SelectionPolicy, read_selection_policy, write_selection_policy,
    rank_with_policy, read_tsv,
)

# Illustrative settings, not a calibrated biological model.
policy = SelectionPolicy(
    name="example-v1", score_by="1 / affinity.value",
    filter_by="n_rna_alt >= 5", duplicates="best",
)
write_selection_policy(policy, "example-v1.json")  # refuses to overwrite
combined.to_tsv("all-evidence.tsv")
ranked = rank_with_policy(combined, policy)
ranked.to_tsv("ranked.tsv")

replayed = rank_with_policy(
    read_tsv("all-evidence.tsv"), read_selection_policy("example-v1.json"),
)
```

The immutable policy has `to_dict()` / `from_dict()` for complete, strict,
schema-versioned mappings and `sha256` for the canonical definition, including
its name. Every default is saved explicitly. Changing an expression or model
selection changes the digest even if the name remains `openvax-v1`. Runtime
versions and input facts live separately in
`ranked.extra['selection_policy']['execution']`; authoring history can be passed
as `provenance=` and is stored alongside the definition, outside its digest.
Record actual base digests and ordered config-file hashes there when available.

`rank_with_policy` delegates to `rank_candidates`: it filters first, then scores
surviving evidence and chooses one representative per candidate/stratum. Unknown
filter results exclude groups under the existing DSL's evidence-retention
rules. Missing scores remain unranked; zero scores remain zero. It performs no
predictor calls, automatic method/version preference selection, or score filling.
Explicit model selections follow the same DSL rules as direct calls. To choose
input-dependent defaults, call `resolve_default_methods` and
`resolve_default_versions` explicitly and include their results in the effective
policy before saving. Unstated historical versions remain unknown. Distinct
source-local selections need separate effective policies/invocations; do not
apply one source's choice globally.

CSV/TSV exports retain the full definition, digest, derivation and execution
record in long and wide form. Numeric measurements and annotations round-trip
exactly as binary floats. Preserve the **full evidence** as well: a filtered
ranking cannot restore excluded observations or discarded alternatives. Replay
also needs the recorded Topiary version when exact execution semantics matter.

Proteasome cleavage and whole-peptide half-life can already be named explicitly,
for example `peptide_view(proteasome_cleavage.score)` and
`peptide_view(serum_half_life.value)`. Saving a policy does not enable these
predictors or add processing weights to existing defaults. Typed extracellular
site-level evidence remains [#288](https://github.com/openvax/topiary/issues/288).

### Composed consumer configuration

Vaxrank owns YAML composition and window/construct configuration. Its repeated
`--config builtin:openvax-v1 --config overrides.yaml` workflow merges mappings
left to right and replaces lists/scalars. The Topiary integration boundary is
the **already-composed policy subtree**:

```python
from topiary import resolve_selection_policy

# `merged` comes from the consumer's config loader, after all overrides.
policy = resolve_selection_policy(merged["selection_policy"])
saved_settings = {
    "selection_policy": policy.to_dict(),
    "selection_policy_sha256": policy.sha256,
    "selection_policy_provenance": config_provenance,
    "vaccine_settings": effective_vaccine_settings,
}
# Reload complete definitions strictly; never apply newer authoring defaults.
policy = SelectionPolicy.from_dict(saved_settings["selection_policy"])
```

`resolve_selection_policy` accepts a JSON- or YAML-decoded mapping with required
`name` and `score_by`. It fills omitted optional settings only after composition.
An omitted filter in an override inherits the base filter; an explicit YAML
`filter_by: null` removes it. Replacing a filter expression does not AND it with
the old expression. Model selections are mappings; version selections are lists
of `{kind, method, version}` records, so an override replaces that entire list.
Topiary does not merge files or parse unrelated consumer settings.

Vaxrank 3.36.0 consumes this subtree in its shipped configuration bundle.
Its post-score minimum gate, missing-score fill, window rules, RNA weighting
and construct settings belong to that bundle too. The consumer fixture in
`scripts/check_vaxrank_candidates.py` loads `builtin:openvax-v1` through
Vaxrank's public config loader and compares its numeric scores with
`evaluate_selection_policy`. A cutoff inside a score expression must not become
a destructive pre-filter when replaying that baseline.

The representative ranking above is one view. `evaluate_selection_policy`
retains every input row and evaluates explicit occurrence identities before
representative selection or vaccine-window construction:

```python
from topiary import (
    SelectionPolicy, evaluate_selection_policy, replay_selection_policy,
    select_policy_representatives, read_tsv,
)

policy = SelectionPolicy(
    "example-baseline", "(affinity.value < 5000) * affinity.value.logistic_normalized(350, 150)",
    score_fill=0.0, min_score=1e-5, duplicates="best",
)
evaluated = evaluate_selection_policy(combined, policy)
occurrences = evaluated.occurrences  # includes excluded and unscorable alternatives
eligible = evaluated.selected      # no candidate collapse
representatives = select_policy_representatives(evaluated)
evaluated.evidence.to_tsv("complete-evaluation.tsv")
replayed = replay_selection_policy(read_tsv("complete-evaluation.tsv"))
```

This example illustrates a scoring convention; use Vaxrank's published bundle
for `openvax-v1`. Neither expression is a calibrated probability of immunogenicity. The cutoff
remains inside the score, with a separate inclusive minimum-score gate.
`raw_score` preserves missingness even when `score_fill=0` supplies an effective
zero. Pre-filtered groups are not scored and cannot be restored by filling.
With no minimum gate, missing scores remain eligible but unranked, matching the
existing candidate-ranker convention.

For direct Vaxrank frames, pass
`group_keys=["prediction_id", "peptide", "peptide_offset", "allele"]` and the
consumer's per-occurrence `alleles` mapping/callback. The callback is evaluated
once per peptide identity and saved as concrete declarations, never Python code.
Projected allele groups carry `supporting_rows` links, not duplicated RNA counts.
`evidence_rows` links the group's original observations. These positions refer
to `evaluated.evidence.long_df`; retain that complete long-form result when
saving an evaluation. Input columns and metadata remain available there.

Use `source_contexts={label: {...}}` when sources have independent model defaults.
Every source label must be named; each mapping may replace `default_methods`,
`default_versions`, `kind_support`, and `alleles`. Filtering and scoring share
those choices. Definitions remain separate from runtime contexts, which are
stored alongside decisions in `extra["policy_evaluation"]`. Replay uses the
complete evidence and stored contexts and never invokes prediction. Model
defaults resolve ambiguity in the existing DSL; they are not a requirement that
all input measurements come from the selected model.

`select_policy_representatives` chooses an actual eligible occurrence, without
summing support, and records all alternative occurrence IDs, including excluded
alternatives. It records the choice and its runtime keys in the evaluation
evidence metadata; replay repeats that explicit choice and `audit` links
criteria to the selected occurrence and rationale. Direct consumers supply
their `candidate_keys` and `strata` if
combined-source candidate columns are absent. Missing values sort last; stable
input order breaks exact ties. Topiary owns these generic decisions; Vaxrank
still owns window geometry, source admission and construct assembly.

### Named criteria and audit decisions

Criteria reuse existing DSL expressions through an explicit namespace. This
example is synthetic policy content, not a recommended processing weight:

```python
from topiary import SelectionCriterion, RankingTerm

binding = SelectionCriterion("binding", "affinity.value < 500", "eligibility")
processing = SelectionCriterion(
    "processing", "peptide_view(proteasome_cleavage.score)", "score",
)
policy = SelectionPolicy(
    "example-processing", 'criterion("processing")',
    filter_by='criterion("binding")', criteria=(binding, processing),
    ranking_by=(RankingTerm("n_rna_alt", ascending=False),),
    unknown="exclude", duplicates="best",
)
evaluated = evaluate_selection_policy(combined, policy)
audit = evaluated.audit
```

Eligibility criteria compose with explicit `&`, `|`, and `~`; score terms
combine through explicit arithmetic; `ranking_by` lists ordered expressions
and directions after the primary score. YAML overrides do none of this
implicitly. Numeric references can be compared to form predicates, and predicates can
participate in explicit score arithmetic. A bare numeric reference is not an
eligibility predicate; a bare predicate is not a named numeric score term.
An input column called `binding` remains a column; only `criterion("binding")`
means the named criterion. Unknown/cyclic references, duplicate names, wrong
roles and non-boolean eligibility outputs raise. Optional `applies_to` is an
eligibility expression; false applicability is recorded as `not_applicable`,
and its reference remains unknown rather than becoming an observed failure.

Audit rows retain occurrence identity, raw row links, criterion name and role,
value, status and reason:

| Status | Meaning |
| --- | --- |
| `pass` / `fail` | An applicable predicate evaluated true / false |
| `unknown` | Missing column/input, missing/ambiguous/conflicting model evidence, unknown applicability, or an out-of-domain calculation |
| `value` | An observed numeric term, including zero |
| `not_applicable` | The applicability predicate was observed false |
| `not_evaluated` | Unreferenced, or scoring skipped after a pre-filter |

Named predicates use three-valued boolean logic. `unknown="exclude"`,
`"include"`, or `"error"` controls the decision on an unknown final eligibility
result, while audit values remain unknown. Direct-expression policies without
criteria keep historical DSL comparison behavior. A vector expression with an
ambiguous/conflicting model is unknown for that evaluation context; separate
source contexts prevent one source's model selection from being imposed on
another. Unexpected programming/DSL errors still raise.

Schema 2 persists criteria, their complete expansions, references, ordered
terms, fill/gate settings and unknown handling in the policy digest. Original
schema-1 definitions retain their original serialization and digest. Definitions
are self-contained; a mutable registry cannot change a saved recipe. Export
`evaluated.evidence`, not a filtered ranking, to reproduce rejected alternatives.
The Vaxrank consumer test carries these records through native dataset save/load
and verifies frozen scores, selected windows, peptide constructs and mRNA
constructs, plus a changed criterion that changes the selected window.

## Identities and source evidence

| Column | Meaning |
| --- | --- |
| `source_label`, `source_row` | Source label and zero-based row in its normalized long table |
| `source_observation_id` | One source's sequence/context/annotation observation; differing abundance stays distinct |
| `candidate_id` | Same sample, peptide and canonical HLA allele across sources |
| `candidate_sample`, `candidate_allele` | Resolved identity, without replacing original sample/allele cells |
| `protein_sequence_id` | Exact full amino-acid sequence explicitly supplied in `protein_sequence` |
| `source_prediction_kind`, `source_prediction_method`, `source_predictor_version`, `source_prediction_run_name` | Original prediction identity, including for sparse wide-file round trips |
| `source_prediction_mhc_dependence` | Original allele-free, per-allele or joint-genotype scope |

Rows lacking sample identity require `sample_name=`. Already named samples
retain their identities. Never assign different patients the same label.
Original source metadata is retained under `extra['combined_sources']`.

Observations are retained without summing counts, averaging predictions or
choosing a source silently. Identical predictor/version names in different
pipelines can therefore have distinct attributable measurements. The DSL groups
combined frames by source observation, peptide and allele, with its usual
sample/genotype context. Existing frames keep their existing grouping. Source
annotations and new features work through `Column(...)` and string expressions;
`peptide_view(...)` still reads allele-independent evidence in pMHC scoring.

Incompatible allele scopes require filtering to compatible sources before a
kind expression can score them together. Contradictory predictions within the
same source/observation/model/version slot require distinct source labels or
`prediction_run_name` values, so the DSL cannot silently select the first row.

`rank_candidates` chooses one representative observation per candidate per
stratum. Default `duplicates="error"` requires scores to agree, including
whether they are missing. `"best"` and `"worst"` explicitly select an observation
without combining its abundance with another source's. `candidate_observations`
links all contributing observations in that selected view. MHC classes rank
separately by default. Missing scores have `ranking_status="missing_score"`
and no rank. Filters narrow the returned view, preserving original evidence.

A shared table does not calibrate incompatible scores. Select compatible
measurements or an explicit composite expression; use stratified lists where a
pooled policy cannot score every candidate. Tumor specificity requires its own
evidence and eligibility decision.

RNA measurements need their own units and biological subject. For example,
[LENS reports](https://pmc.ncbi.nlm.nih.gov/articles/PMC10246587/) transcript TPM,
fusion fragments per million, splice-expression and viral read-count measures
in different workflows. A common column name does not make them interchangeable.
[pVACseq reports](https://pvactools.readthedocs.io/en/stable/pvacseq/output_files.html)
also distinguish transcript expression from tumor RNA depth/VAF and report
predictions per algorithm. Preserve these distinctions when defining a policy.

## Isovar hypotheses for comparison

`read_isovar_hypotheses` reads `isovar.protein_hypotheses.v1` and `v2`
JSON exports or the mapping returned by `export_protein_hypotheses`. Version 2
is produced by Isovar 1.37+. The companion
Isovar TSV lacks the evidence sets and full provenance; supply JSON here.

```python
from topiary import read_isovar_hypotheses, combine_sources, rank_candidates

hypotheses = read_isovar_hypotheses("tumor-1.hypotheses.json")
combined = combine_sources({"reported": reported_candidates, "isovar": hypotheses})
alternatives = combined.filter_by("isovar_rank > 1")
ranked = rank_candidates(combined, "affinity.value", ascending=True)
```

Each imported row is one translation, including synonymous nucleotide sequences
and lower-ranked alternatives. A hypothesis with no translations still has a
protein-only observation. The reader creates no peptides, alleles or scores;
these rows have no `candidate_id` after combination. Adding them leaves the
reported candidates, their scores/ranks, and exact-peptide re-scoring calls
unchanged. Existing top-protein fragment selection, reconstruction settings and
filter defaults also remain unchanged. Using alternative ORFs to generate new
candidates requires a separate, explicit choice.

`isovar_rank`, `representative` and `passes_all_filters` describe the producer's
results; they do not establish default eligibility. Filtered and uncertain
outcomes stay visible for comparison. `protein_hypotheses_complete` and
`protein_sequence_limit` report whether the upstream export may be truncated.
RNA support and reconstructed sequence alone do not establish tumor specificity.

`protein_hypothesis_sequence` includes partial translated windows.
`protein_sequence` is populated only for translations explicitly starting at the
annotated start codon whose protein ends at a stop codon. Only these full
sequences enter `protein_evidence_view`; a partial window is never promoted to a
full ORF. The producer's exact sequence identifier is kept as
`isovar_protein_sequence_id`, while the combined table reserves
`protein_sequence_id` for its full-protein grouping.

The complete original export is retained in
`hypotheses.extra['isovar_hypotheses']`, including events without proteins,
reference contexts, observed edits, filters and RNA evidence sets. After
combination it is under
`combined.extra['combined_sources']['isovar']['extra']['isovar_hypotheses']`.
Topiary CSV/TSV saving retains that metadata in both long and wide form.
Literal sample IDs and sequences also survive reload: for example, `001` stays
a string and the amino-acid window `NA` stays a sequence, not a missing value.

Read and fragment counts have separate `protein_*` and `translation_*` columns.
Both export versions populate `protein_reads` and `translation_reads`; the
older `*_segments` columns remain aliases for compatibility. Optional `*_umis`
and `*_cells` are label counts with separate `*_umis_complete` and
`*_cells_complete` flags. Null means unassessed, false means the count is not
exact, and true means the producer assessed complete labels. `*_unlabeled_reads`
and `*_unknown_library_reads` explain incomplete measurements; detailed
`label_statuses` and allele-level support stay in the original metadata.
These are not TPM or independent molecule counts.

For example, the existing filter DSL can restrict comparisons to complete
UMI measurements with `combined.filter_by("protein_umis_complete & protein_umis >= 2")`.
This filters comparison observations; it does not admit alternatives as candidates.
To combine read support explicitly with Isovar 1.37+:

```python
from isovar import union_rna_support
from topiary import normalize_isovar_rna_support

export = hypotheses.extra["isovar_hypotheses"]
keys = hypotheses.df.loc[hypotheses.df.isovar_rank <= 2, "protein_evidence_set_id"]
support = union_rna_support([
    normalize_isovar_rna_support(export["evidence_sets"][key]) for key in keys])
```

This counts shared reads once, including support repeated across synonymous
translation rows. Normalization translates legacy `segments`/`segment_ids`
into `reads`/`read_ids` and validates counts and membership without changing the
saved export. The union refuses incompatible evidence scopes. It does not union
UMI/cell counts: exports do not include the label identities needed for that.
Missing evidence-set
IDs cannot be resolved or safely combined from counts alone. The comparison
reader itself does not import Isovar or run any reconstruction or predictor.

## Full ORFs and RNA-only tables

```python
import pandas as pd
from topiary import protein_evidence_view

orf_table = pd.DataFrame({
    "event_id": ["GRCh38:event-123", "GRCh38:event-123"],
    "protein_sequence": ["MAAASIINFEKL", "MQQQSIINFEKL"],
    "transcript_expression": [11.0, 22.0],
    "expression_unit": ["TPM", "TPM"],
})
combined_orfs = combine_sources({"exacto_normalized": orf_table}, sample_name="patient-01")
proteins = protein_evidence_view(combined_orfs)
```

No peptide, HLA or prediction columns are required here. The protein view links
equal full sequences within a shared sample and explicit `event_id`, retaining
different sequences as alternative ORFs. Its observation links lead back to each
tool's RNA measurements; abundance is never automatically averaged or summed.

Equal protein products do not establish equal nucleotide ORFs. Distinct
`orf_id`, `coding_sequence`, transcript, frame or completeness annotations
remain independent source observations even when the translated sequence is
identical. Use `reconcile_evidence` below to establish explicit ORF relationships
while retaining those observations.

`event_id` must be explicitly harmonized, including assembly/variant identity
where appropriate. Native labels from different tools are not assumed equal.
Without a common event ID, observations stay separate while sequence IDs still
show exact sequence agreement. Local `pep_context` or `sequence` is not promoted
to a full ORF. A peptide-only table does not establish its generating full ORF.

To use another source's RNA measurement for an explicitly matching ORF, first
reconcile the input and join the selected measurement as a named feature. `join_annotations` refuses ambiguous annotation keys:

```python
from topiary import join_annotations, reconcile_evidence

combined = reconcile_evidence(combined)
# Require an explicit ORF match; equal proteins alone are insufficient.
keys = ["candidate_sample", "orf_hypothesis_id"]
rna = combined.df.loc[
    combined.df.source_label.eq("exacto_normalized"),
    [*keys, "transcript_expression"],
]
enriched = join_annotations(
    combined, rna, on=keys, prefix="exacto",
    provenance={"source": "exacto_normalized", "unit": "TPM",
                "policy": "same sample and reconciled ORF; retain transcript-level unit"},
)
ranked = rank_candidates(
    enriched, "exacto_transcript_expression / affinity['netmhcpan'].value",
    duplicates="best",
)
```

This expression illustrates a policy, not a calibrated biological model. The
alternative ORF's abundance cannot attach just because its variant or gene
agrees. Missing or ambiguous identity requires explicit resolution.

## Reconcile ORFs and RNA observations (5.87.0+)

`reconcile_evidence(combined)` returns a copy with stable links between biological
entities. It neither chooses a caller nor changes the candidate universe.
`evidence_views(combined)` exposes `events`, `orfs`, `proteins`, `occurrences`,
`candidates`, `rna_observations`, `links` and `candidate_occurrences` as DataFrames; supply
`source_labels=["lens"]` for a source-stratified view.

| Input assertion | Identity rule |
| --- | --- |
| `event_id` or list-valued `event_ids` | Same sample, explicit `reference_name` and normalized event name; absent reference remains source-local |
| `orf_id` | Caller-local ID; contradictory descriptors raise, absent descriptors can be supplied by another row with the same local ID |
| `coding_sequence` and `transcript_path`, or `transcript_id` plus `orf_start`/`orf_end` | Cross-caller ORF agreement requires the same sample, reference and all supplied descriptors |
| `reading_frame`, `orf_completeness`, start/stop flags, `linked_variants` | Retained in ORF identity; differing assertions remain alternative hypotheses |
| `protein_sequence` | Exact full product, separate from nucleotide ORF identity |
| `protein_hypothesis_sequence` | May be partial; never promoted to a full protein |
| `peptide_start`/`peptide_end` | Zero-based half-open occurrence in the supplied protein; validated against its sequence |
| Missing peptide coordinates | Source-local occurrence; no guessed equivalence from a shared peptide |

ORF bounds are also zero-based half-open. `transcript_path` is an explicit ordered
JSON path in the caller's normalized reference convention. Topiary does not lift
over assemblies, normalize native genomic variant strings or infer a path from
gene names. Incomplete records remain useful without establishing equivalence.
Reference scope follows the distinction between sequence and annotation identity
in [NCBI's feature documentation](https://www.ncbi.nlm.nih.gov/genbank/genomes_gff/).

`rank_candidates` retains the chosen row's `orf_hypothesis_id` and
`peptide_occurrence_id`. `candidate_observations` and the relational `links`
retain alternative support. Rediscovery does not multiply candidate scores.
Use an explicit duplicate policy or source filter when measurements disagree.
`candidate_occurrences` links all same-sample peptide occurrences to existing
pMHC queries, including sequence reports with no HLA assignment. Its
`reported_candidate` flag distinguishes a reported peptide-HLA pair from a
sequence match. These links transfer no scores, expression, specificity or
eligibility between observations, and create no new candidates.

RNA observations can be supplied as a list of records in `rna_observations`:

```python
measurement = {
    "sample_name": "patient-01", "entity_type": "transcript",
    "entity_id": "ENST-example", "quantity": "count", "unit": "reads",
    "value": 2, "library_id": "rna-library-1", "read_set_id": "alignment-1",
    "evidence_unit_ids": ["read-a", "read-b"], "method": "caller", "version": "1",
}
```

`normalize_rna_observation` validates this shape. `entity_type` may be `gene`,
`transcript`, `orf` or `variant`. A TPM measurement uses `quantity="abundance"`,
`unit="TPM"` and no evidence-unit list. Unknown values remain null. Original
quantifier metadata and extra fields survive; gene/transcript expression never
becomes ORF expression automatically. Each caller's observations remain distinct.

`union_rna_observations([measurement, other_measurement])` unions identified
read, fragment, UMI or cell memberships only within one sample, library, read-set
namespace, measured entity and unit. Shared members count once. Missing membership,
different namespaces and TPM quantities raise: neither different file names nor
different callers establish independent evidence. An explicitly empty membership
is measured zero; an absent membership is unknown. Scalar expression/count
columns already present in source tables stay untouched and are never implicitly
converted into identified evidence sets.

All identity columns, descriptors, alternative observations and metadata survive
Topiary CSV/TSV save/reload in long and wide forms. Reconciliation can be repeated
after reloading. The composed workflow test covers original and additive scoring,
retained hypothesis selection, and count union without running a predictor during
assembly.

## Predict exact peptide occurrences

Topiary 5.91.0 adds `predict_peptide_occurrences` and
`TopiaryPredictor.predict_from_peptide_occurrences` for a selected peptide
universe. Supply one record per `(prediction_id, peptide, peptide_offset)` within
each sample (when `sample_name` or `candidate_sample` is supplied). An ID may name
several peptide windows in the same source. Use the same occurrence
identity Vaxrank and the policy evaluator consume. Different occurrences of one
peptide remain separate even when they share a gene, coordinate or HLA candidate.

```python
import pandas as pd
from topiary import (
    TopiaryPredictor, TopiaryResult, SelectionPolicy, evaluate_selection_policy,
)

# model is an explicitly configured mhctools predictor with patient alleles.
occurrences = pd.DataFrame([
    dict(prediction_id="gene-a:12", peptide="SIINFEKL", peptide_offset=12,
         n_flank="AAA", c_flank="GGG", gene="gene-a"),
    dict(prediction_id="gene-b:20", peptide="SIINFEKL", peptide_offset=20,
         n_flank="TTT", c_flank="CCC", gene="gene-b"),
])
predictor = TopiaryPredictor(models=model)
predictions = predictor.predict_from_peptide_occurrences(occurrences)
evaluation = evaluate_selection_policy(
    TopiaryResult(predictions), SelectionPolicy("binding", "1 / affinity.value"),
    group_keys=["prediction_id", "peptide", "peptide_offset", "allele"],
    kind_support=predictor.kind_support,
)
evaluation.evidence.to_tsv("occurrence-evidence.tsv")
```

The model scores only the supplied peptides against its configured alleles.
Within a shared sample/flank/genotype context, distinct peptides are batched and
identical inference inputs are scored once, then linked to each original
occurrence. Source annotations and read support are copied, never summed.
`prediction_mhc_dependence` records each kind's scope; haplotype predictions carry
the configured `allele_set`. An explicitly supplied haplotype set must match the
configured model. Missing kind/allele coverage and ambiguous outputs raise.

Flank-dependent models require both `n_flank` and `c_flank`. An empty string is a
known terminus; a missing value is unknown. `use_flanks=False` explicitly requests
peptide-only inference while preserving the original context. The output's
`prediction_flanks_supplied` records whether flanks were supplied in the model call; it does not imply that every returned prediction kind depends on them.
Missing source offsets remain missing. Model peptide-length/domain checks still
apply; no padding, comparator sequence, or additional peptide window is inferred.

With `TopiaryPredictor(predict_wt=True)` (or `predict_wt=True` on the standalone
function), supplied `wt_peptide` values are scored using their own
`wt_n_flank` / `wt_c_flank`. Flank-dependent comparator prediction requires those
flanks or explicit `use_flanks=False`; primary-peptide context is not borrowed.
Absent comparators keep missing scores. Explicit filter/sort settings operate on
occurrences; `only_novel_epitopes` does not reinterpret this already selected
peptide universe. Cache failures are strict by default; an explicit
`cache_miss_handler` / `on_miss` retains the existing reported-partial-batch
contract, with occurrence IDs in the failure records.

Prediction inputs contain context and annotations, not historical measurement
columns. Keep imported measurements in their original table and use the additive
operation below. Context-aware live prediction requires mhctools' flank-aware
API; sparse historical tables with unknown model versions remain the separate
prediction-source work in [#368](https://github.com/openvax/topiary/issues/368).

## Add prediction features on demand

```python
from mhctools import MHCflurry
from topiary import rescore_candidates

model = MHCflurry(
    alleles=["HLA-A*02:01"], presentation_allele_mode="per_allele",
)
enriched = rescore_candidates(
    combined, model, prefix="fresh",
    select="candidate_allele == 'HLA-A*02:01'",
    use_flanks=False,  # explicitly request peptide-only predictions
)
original_policy = rank_candidates(
    enriched, "affinity['netmhcpan'].value", ascending=True, duplicates="best",
)
new_policy = rank_candidates(
    enriched, "fresh__mhcflurry__pMHC_presentation__score", duplicates="best",
)
```

Features such as `fresh__mhcflurry__pMHC_affinity__value` are appended. Original
values and predictor names keep their meaning. Adding features does not switch
the ranking policy. They work in `filter_by`, `sort_by`, `evaluate_scores`, and
Vaxrank DSL expressions. `extra['candidate_rescoring']` records the new producer,
models, versions, measurement semantics, UTC creation time, selection and discovery-source labels.
Discovery source and prediction producer are separate facts.

Configured mhctools models supply `predict_dataframe` and `kind_support()`.
This path reuses `predict_peptide_occurrences` to batch compatible requests and
preserve source identity; it appends primary-peptide features and leaves all
historical comparator measurements intact.
Supplied flanks are forwarded by default; flank-dependent models require both
flanks or explicit `use_flanks=False`. Haplotype models require a matching
stated `allele_set`. Existing candidate identities are preserved; ORF-only rows
remain unscored. Scanning new windows or adding HLA candidates is a separate
operation, with novelty/context handling tracked in
[#364](https://github.com/openvax/topiary/issues/364).

Every declared allele-dependent prediction kind must cover the selected
candidate. An allele-free processing output cannot substitute for missing
affinity or presentation; processing-only models remain supported.

### Save and reload flanking context

An empty `n_flank` or `c_flank` string states a known protein terminus; a missing
value states that the context is unknown. Use Topiary's `to_tsv`/`to_csv` and
`read_tsv`/`read_csv` together to preserve that distinction in both long and wide
tables. Context-dependent re-scoring after reload then receives the same flanks
as before saving; unknown context still requires an explicit resolution.

Files with canonical flank columns include
`#topiary_flank_encoding=escaped-v1`. In those columns, missing values are
written as `<NA>` and known-empty strings remain empty cells. Ordinary sequence
text, including `NA`, remains literal text. A literal `<NA>` or a string starting
with a backslash gains one leading backslash, which the reader removes.
Other columns containing only strings and missing values use the same cell
encoding. The `#topiary_text_encoding` metadata records `escaped-v1` and the
affected column names, preserving literal identifiers and sequence text.
Columns whose stated cells are all dicts or lists, such as
`measurement_context`, are written as one JSON document per cell, with missing
cells as `<NA>`. The `#topiary_json_encoding` metadata records `json-v1` and the
affected column names; the reader decodes the cells back into dicts and lists.
Values follow JSON's data model, so tuples come back as lists and non-string
keys as strings. A cell JSON cannot represent raises `TypeError` before the
output file is opened.
Numeric measurements retain their normal numeric representation and inference.

Legacy files without this flag retain the earlier missing-value interpretation.
Their blank flank cells cannot distinguish termini from unknown context. Recover
that distinction from the original source; do not replace all missing flanks
with empty strings. Ordinary `pandas.read_csv` also cannot recover the distinction
without decoding this format.

## Vaxrank

Pass the full enriched frame to Vaxrank's
`attach_per_allele_scores(..., topiary_df=...)`. Rebuilding it from prediction
objects alone loses annotation features. Its current grouping key is
`prediction_id`: map `source_observation_id` to that name in a separate consumer
view, preserving original IDs in the combined evidence. Explicitly select
representative candidates before construction so repeated discovery does not
multiply vaccine targets. Construct generation still requires suitable sequence
context, targetability and tumor-specificity admission evidence.

`scripts/check_vaxrank_candidates.py` tests actual DSL scoring and
`VaccinePeptide` construction against released Vaxrank for mutation, fusion,
splice, CTA, ERV and viral synthetic examples. Original, re-scored and
RNA-enriched policies select different construct orderings. Generalized
Vaxrank file/CLI ingestion and version adoption are tracked in
[Vaxrank #497](https://github.com/openvax/vaxrank/issues/497).
