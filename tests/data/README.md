# Sid regression data

The Sid fixtures are checked-in files plus shared reads. Every checked-in file
is pinned in `manifest.json` and verified offline. The alignment records come
from `openvax-v1`, the OpenVax libraries' shared Sid test data
([iskandr/osteosarc#56](https://github.com/iskandr/osteosarc/issues/56)): each
read file the tests use is the `openvax-v1` member `topiary/<path under
tests/data>`, which holds exactly its original records, repeats included.

`tests/sid_data.py` lists those files in `SHARED_READS`. The first test that
reads one downloads and verifies the bundle (28 MB) into the osteosarc cache,
`OSTEOSARC_CACHE`, else the shared OpenVax cache, then exports Topiary's reads
once per test process as indexed BAMs. Later runs are offline. Exports are
coordinate-sorted with the source's full header, so records at one position
may be ordered differently from the original files; tests compare records, not
file bytes. `read_selections` in `manifest.json` keeps each file's regions,
source and record-multiset hash, and the tests check every export against it.

The rearrangement inputs in `osteosarc_rearrangements/*.json.gz` keep their
original records as the audit envelope of Isovar's supplied-fusion input; a
test checks those records against their `openvax-v1` members.

To look at the reads outside the tests:

```sh
osteosarc test-data list openvax-v1
osteosarc test-data export openvax-v1 /tmp/topiary-reads \
  --member topiary/osteosarc_all_variants/source/t2-all-variant-regions.bam
```

Verify the checked-in files without downloading or changing anything:

```sh
python -m scripts.osteosarc_test_data --verify tests/data
```

Fixtures and tests ship in the source distribution. See the
[provenance and regeneration guide](../../docs/osteosarc-shared-data.md) for
selection policies, source identity, scientific expectations and cache options.
