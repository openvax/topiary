# Isovar 1.37 export fixtures

These synthetic exports were generated with the published Isovar 1.37.0 wheel
and producer test fixtures at tag `v1.37.0`, commit
`20a9682f8e1e5cc9811b3992ce0193034d17d3c9`. No patient data or reference download
is used. Regenerate from the Topiary root with:

```sh
python scripts/generate_isovar_export_fixtures.py /path/to/isovar-at-v1.37.0
```

- `protein-v2.json`: the producer's `tests/test_protein_hypotheses.py`
  `synthetic_result`, all hypotheses retained. It has overlapping support,
  synonymous translations and a partial window; cell/UMI labels are unassessed.
- `protein-v2-labelled.json`: the same result exported with a synthetic BAM
  containing matching read identities and CB/UB tags. One read lacks a UMI,
  and one read group lacks library metadata. Some measurements are complete,
  others are lower bounds. The generator records the exact BAM construction.
- `orfs-v4.json`: the producer's `tests/test_sv_rna_orf_export.py`
  `reconstruction`, with a resolved cell/UMI label. The one junction-spanning
  ATG ORF remains exploratory. Signal lineage remains unresolved.
- `comparison-v3.json`: the producer's `tests/test_sv_rna_comparison.py`
  `inputs`; an annotated partial translation matches part of a supplied protein.
- `orfs-input.json` and `comparison-input.json`: native synthetic producer
  inputs for the two SV exports. Integration tests call the installed producer,
  then run Topiary's API and offline CLI and verify the serialized report.

All export files are unmodified producer output. The v1 protein fixture remains
at `../isovar_hypotheses.json` to verify compatibility. Tests drive both RNA
readers through the same support-validation battery and exercise protein
comparison import, filtering, CSV/TSV long/wide reload and unchanged candidate
scoring. UMI/cell counts are never summed across hypotheses or interpreted as
molecules.
