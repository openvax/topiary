# Repeated translations from original MTND5 RNA

This is a small original-read subset of the public Sid T2 January-2025 UCLA
STAR corpus at MT 12994 G>A. Osteosarc 0.7.0 extracted 711 unchanged SAM records
from 419 query names. `recipe.json` pins the source BAM, region and query names;
`manifest.json` records the generator version and output checksums. The source
corpus and Ensembl 87 reference remain in `../osteosarc_all_variants/`.

The names were taken from the RNA supporting protein rank 561 in the full
Isovar 1.37.0 reconstruction with all hypotheses retained, context peptide
length 25, assembly enabled and the read policy in
`scripts/osteosarc_rna_overlay.py`. They are a regression subset, not a new
biological selection or an estimate of the event's total RNA support.

Unmodified Isovar 1.37.0 reconstruction/export of this subset yields 132
protein hypotheses. Rank 58 has two translation IDs each repeated three times;
rank 97 has another ID repeated three times. Different assembled extensions
produce identical exported translations after reference trimming. The export
itself is generated in the consumer test, without inserting repeated records
or altering its hypotheses, ranks or RNA evidence.

Regenerate from the Topiary checkout using the separate Osteosarc environment
described in `../README.md`:

```sh
/tmp/osteosarc-fixtures/bin/python tests/data/isovar_repeats/regenerate.py /tmp/new-isovar-repeats
```

The generator verifies that every selected SAM record, including multiplicity,
matches the original source. The integration test reconstructs the event,
exports it through Isovar, imports through both Topiary reader entry points,
then checks comparison filtering, support union and long/wide CSV/TSV reload.
Original repeated records remain in provenance; they must not multiply rows
or support. Synthetic v1/v2 tests also cover conflicting records and unchanged
default candidate scoring.
