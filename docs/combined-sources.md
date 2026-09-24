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
normalize LENS and pVACseq. Native Exacto parsing is separate work in
[#365](https://github.com/openvax/topiary/issues/365); an already normalized
ORF/RNA DataFrame is supported now. Aggregated reports contribute only the
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
identical. Full ORF reconciliation is tracked in
[#370](https://github.com/openvax/topiary/issues/370).

`event_id` must be explicitly harmonized, including assembly/variant identity
where appropriate. Native labels from different tools are not assumed equal.
Without a common event ID, observations stay separate while sequence IDs still
show exact sequence agreement. Local `pep_context` or `sequence` is not promoted
to a full ORF. A peptide-only table does not establish its generating full ORF.

To use another source's RNA measurement for an explicitly matching ORF, join it
as a named feature. `join_annotations` refuses ambiguous annotation keys:

```python
from topiary import join_annotations

# combined contains both ORF/RNA rows and peptide rows with explicit
# protein_sequence and event_id identifying their generating protein.
keys = ["candidate_sample", "event_id", "protein_sequence_id"]
rna = combined.df.loc[
    combined.df.source_label.eq("exacto_normalized"),
    [*keys, "transcript_expression"],
]
enriched = join_annotations(
    combined, rna, on=keys, prefix="exacto",
    provenance={"source": "exacto_normalized", "unit": "TPM",
                "policy": "same sample, event and exact full sequence"},
)
ranked = rank_candidates(
    enriched, "exacto_transcript_expression / affinity['netmhcpan'].value",
    duplicates="best",
)
```

This expression illustrates a policy, not a calibrated biological model. The
alternative ORF's abundance cannot attach just because its variant or gene
agrees. Missing or ambiguous identity requires explicit resolution.

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
Other columns keep their existing formatting and type inference.

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
