# Sid regression data

The Sid fixtures are an offline, checksum-verified Osteosarc export described by
`manifest.json`. Only reads overlapping the loci exercised by the tests are
bundled; unused vaccine read sets and whole alignments are excluded.

From the repository root, generate the complete corpus into a **new** directory:

```sh
python -m scripts.generate_sid_fixtures --output /tmp/new-sid-fixtures
```

This uses Osteosarc's cache and indexed extractor with the pinned sources and
regions in `sid-fixtures.json`. After the first acquisition, pass `--offline`
to regenerate from cache. `--live-alignments` additionally compares the selected
records against the original public indexed BAMs. Neither mode downloads an
entire BAM. Python dependencies come with Topiary; extraction needs samtools.

Verify the checked-in bundle without downloading or changing anything:

```sh
python -m scripts.osteosarc_test_data --verify tests/data
```

Fixtures, tests and generators ship in the source distribution. See the
[provenance and regeneration guide](../../docs/osteosarc-shared-data.md) for
selection policies, source identity, scientific expectations and cache options.
