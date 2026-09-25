# Reports for every SV nomination

`build_sv_interest_report` retains the complete shared candidate set, including
DNA-only, RNA-only, unresolved, unproductive and unassessed entries. It accepts
the mapping returned by Osteosarc's SV interest catalogue and Isovar SV ORF
v1-v4 exports. Source observations and their original warnings remain intact.

```python
from topiary import build_sv_interest_report, write_sv_interest_report

report = build_sv_interest_report(catalogue, orf_exports,
                                 comparisons=rna_prediction_comparisons)
paths = write_sv_interest_report(report, "output/all-svs")
```

The optional comparisons are Isovar `compare_sv_rna_predictions` results
(`sv_rna_prediction_comparison.v1` through `v3`).
They add annotated-frame translations separately from ATG hypotheses, including
partial frame-anchored sequence. Their full-interval RNA counts remain unknown;
junction counts never substitute for support of a complete ORF.

For saved inputs, the offline CLI runs the same policy:

```sh
python -m topiary.cli.sv_interest \
  --catalogue catalogue.json \
  --orf-export sample.orfs.json \
  --comparison sample.protein-comparison.json \
  --output-prefix output/all-svs
```

Repeat the RNA input options for additional samples and reconstructions.
`--event-aliases aliases.json` accepts an explicit map from historical export
names to catalogue target IDs. Original names remain in the provenance; there
is no fuzzy event matching. Unknown event IDs fail rather than disappearing.

## What the ranks mean

The default ordinal evidence policy orders a transferred annotated frame that
reaches the alteration, then complete RNA-supported ORFs from an annotated
start, a 5′-UTR start, a splice-supported intronic start, and other/ambiguous
starts. Partial products and products lacking full-interval witnesses remain
visible. Ambiguous splice-compatible event linkage has its own tier: a sequence
observed near multiple SV nominations is not automatically attributed to each
DNA event. Candidate rows without any reconstructed protein follow with junction
RNA, same-sample DNA/gene expression, cross-sample expression, or unknown evidence.

These are inspectable evidence priorities, not calibrated translation
probabilities or vaccine recommendations. Biological initiation, protein
abundance and antigen presentation are not inferred. Read orientation does
not alter the rank. Conflicting isoforms retain an ambiguous start prior.

Support ranks use deduplicated full-ORF template IDs **within one sample and
source**. Repeat reconstructions union membership rather than adding counts;
source aliases, synonymous sequences and geometry aliases are not independent
discoveries. No across-library abundance rank is calculated. Gene TPM is shown
with its actual sample and remains separate from mutant-transcript abundance.
Protein abundance is null. Missing acquisitions are not zero support.

Isovar 1.37's ORF v4 records carry `reads`/`read_ids`, optional `umis` and
`cells`, their completeness flags, label-resolution details, signal lineage,
and read orientations. Each `source_observations` entry includes the original
`candidate` and a normalized `rna_support` record. Legacy `segments` and
`segment_ids` normalize to `reads` and `read_ids`. These measurements, including
null and false completeness flags, survive JSON and the nested TSV columns.
UMI/cell counts remain per observation and never contribute to the template
rank: their label memberships are not exported, so adding them would be unsafe.
An assessed empty set remains zero with false completeness; unassessed labels
remain null. UMIs are label counts, not independent molecules.

For a different explicit ordering, use Topiary's existing `apply_sort` DSL on
the candidate or protein DataFrame. Keep source-specific template counts out
of a cross-library abundance comparison.

## Outputs

The JSON includes the complete catalogue and source evidence. Candidate and
protein TSVs retain nested provenance as JSON columns; missing numeric values
are blank. The searchable HTML displays every nomination. FASTA contains each
distinct amino-acid hypothesis once, including partial sequences labelled in
the protein table. Inclusion in FASTA is not a claim of mutant specificity.

RNA abundance and protein abundance are distinct quantities; see
[Schwanhäusser et al.](https://www.nature.com/articles/nature10098).
Gene TPM and expected counts follow the
[RSEM definitions](https://deweylab.github.io/RSEM/rsem-calculate-expression.html).

For annotated-frame products, event linkage refers to the translated interval's
own departure evidence. An exact SV elsewhere in the surrounding RNA path
cannot promote a translation whose departure is ordinary-splice compatible or
unknown. Such products remain visible with ambiguous event linkage.
