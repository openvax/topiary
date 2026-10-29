# Per-bond cleavage evidence

`from_cleavage_results` turns MHCTools' native `CleavageResult` objects or
their serialized mappings into one pandas table. It is included in the normal
Topiary installation. Compute evidence explicitly with the upstream model,
then inspect or save it:

```python
from mhctools import DPP4, PeptideInput
from topiary import from_cleavage_results, read_tsv, to_tsv

peptide = PeptideInput("APAA", occurrence_id="example",
                       source_sequence_name="source", source_start=10)
prediction = DPP4().predict(peptide)
sites = from_cleavage_results(prediction.cleavage_results)
to_tsv(sites, "cleavage.tsv")
restored = read_tsv("cleavage.tsv").df
print(restored[["enzyme", "evidence_type", "bond", "status", "score"]])
```

This short sequence is a software example. Consult each model's retained
`model_metadata` for its assay, domain, references, score interpretation and
limitations before interpreting an output.

| Field | Meaning |
|---|---|
| `bond` | Internal bond `b` splitting `sequence[:b]` and `sequence[b:]` |
| `source_bond` | `source_start + bond`, in the supplied source coordinate frame |
| `occurrence_id`, `source_sequence_name` | Original occurrence labels, including unknown labels |
| `status`, `reason` | Native site outcome, unsupported reason, whole-substrate observation or unassessed state |
| `match` | Nullable binary motif decision; never a calibrated probability |
| `score`, `scored_endpoint`, `score_name`, `score_units` | Native quantitative evidence and its declared meaning |
| `peptide_input` | Complete chemical form, occurrence and descriptive input context |
| `model_metadata`, `conditions` | Model identity, enzyme, assay, references, limitations and reported conditions |
| `result_index` | Position of the input result in this call, grouping its site rows |

A result with no sites still contributes a row. Null bond coordinates do not
mean bond zero or a terminus. Unsupported inputs retain their reason and null
measurements; a reported whole-substrate non-cleavage observation remains
distinct from an unassessed substrate. Omitted bonds are unassessed. Repeated
records are retained and do not establish independent biological evidence.

For a candidate interval `[start, end)` in the supplied source frame, internal
bonds satisfy `start < source_bond < end`; equality identifies a boundary.
First select the matching occurrence. The table does not decide whether a cut
destroys all copies of a desired ligand or whether a different window is useful.

Site evidence, whole-peptide half-life, and peptide-MHC stability answer
different questions. For example, the primary FGF21 study distinguishes
N-terminal DPP4 processing from C-terminal FAP processing using product
analysis and enzyme-specific experiments; it does not establish generic
serum-survival scores for arbitrary peptides.
[Zhen et al., 2016](https://pubmed.ncbi.nlm.nih.gov/26635356/).

Use ordinary DataFrame operations for this table. It is not a peptide-MHC
prediction table for `CachedPredictor` or the affinity DSL. Full chemical and
assay identity, unavailable states and native units survive Topiary TSV/CSV
round trips without a sidecar file. No model is run by conversion or reload.

This addresses the site-transport portion of [#288](https://github.com/openvax/topiary/issues/288).
Chemistry-aware queries for peptide-level caches and further processing-policy
integration remain separate work; this adapter does not validate a new model
or change default ranking.
