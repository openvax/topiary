# Shared fixture mechanics

Move acquisition, original-record selection, transport, and integrity mechanics
to Osteosarc 0.7.0. Keep the reviewed recipe format, source/correction/reference
pins, data bytes and scientific expectations unchanged. Thin compatibility entry
points retain the existing commands; no sibling checkout is required.

Validate historical membership/multiplicity and source/header provenance, then
run the reconstruction/prediction/ranking regressions offline. This consumer
uses the published Osteosarc 0.7.0 release. New recipes can use the
versioned panel/bundle APIs; historical adapters state their SAM-text fidelity.

Tracked by [Osteosarc #15](https://github.com/iskandr/osteosarc/issues/15).
The regeneration command fetches the recipe's pinned sources into CACHE:

```sh
python -m scripts.generate_sid_fixtures --cache-root CACHE --output NEW_DIRECTORY
```

Add `--offline` to repeat it from a populated cache. Always use a new output
destination. `--source-directory tests/data` works only while every local source
still matches its recipe pin; `osteosarc_all_variants/manifest.json` no longer
does, so omit it. The Osteosarc 0.7.0 regeneration reproduced every read file
byte for byte; only the recorded `osteosarc_version` and the manifest hashes
covering it changed. The existing checked-in scientific expectations remain the oracle.
For a common versioned recipe, the builder also accepts `--panel-recipe
recipe.json --panel-source SOURCE_ID=original.bam --output NEW_DIRECTORY
--offline`. Repeat `--panel-source` for each local original input (Vaxrank also
requires `--cache CACHE`). Panel output contains indexed BAMs, checksums, retained
record multiplicities, source/header identities and selection reasons. See the
[shared workflow](https://github.com/iskandr/osteosarc/blob/v0.7.0/docs/fixture-migration.md).
