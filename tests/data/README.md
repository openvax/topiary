# Sid regression data

The Sid fixtures are an offline, checksum-verified Osteosarc export described by
`manifest.json`. Only reads overlapping the loci exercised by the tests are
bundled; unused vaccine read sets and whole alignments are excluded.

The corpus was regenerated with published Osteosarc 0.7.0. Its original read
and reference bytes remain unchanged. From the repository root, generate the
complete corpus into a **new** directory using a separate generation environment:

```sh
python -m venv /tmp/osteosarc-fixtures
/tmp/osteosarc-fixtures/bin/python -m pip install osteosarc==0.7.0
/tmp/osteosarc-fixtures/bin/python -m scripts.generate_sid_fixtures --output /tmp/new-sid-fixtures
```

This uses Osteosarc's cache and indexed extractor with the pinned sources and
regions in `sid-fixtures.json`. After the first acquisition, pass `--offline`
to regenerate from cache. `--live-alignments` additionally compares the selected
records against the original public indexed BAMs. Neither mode downloads an
entire BAM. Extraction needs samtools. Generation is separate from Topiary's
runtime environment while its Isovar integration pins an older Osteosarc
([Topiary #399](https://github.com/openvax/topiary/issues/399),
[Isovar #386](https://github.com/openvax/isovar/issues/386)). Before publishing
changed fixture bytes, set the recipe's `export_url` to their new release tag;
the generated manifest's download URLs must serve its recorded checksums.

Verify the checked-in bundle without downloading or changing anything:

```sh
python -m scripts.osteosarc_test_data --verify tests/data
```

Fixtures, tests and generators ship in the source distribution. See the
[provenance and regeneration guide](../../docs/osteosarc-shared-data.md) for
selection policies, source identity, scientific expectations and cache options.
