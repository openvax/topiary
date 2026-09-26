# Shared fixture mechanics

Acquisition, original-record selection, transport, and integrity mechanics
come from Osteosarc. Topiary's Sid reads are the `topiary/...` members of
`openvax-v1`, the OpenVax libraries' shared Sid test data, published by
Osteosarc 0.11 ([iskandr/osteosarc#56](https://github.com/iskandr/osteosarc/issues/56)).
Each member holds exactly the records of the file it replaces. The reviewed
source/correction/reference pins, the non-read fixture files and the
scientific expectations stay checked in and unchanged.

Tests download the bundle once and export the reads offline afterwards; see
`tests/data/README.md`. Before replacing any other read file with a member,
check that the two hold the same records:

```sh
osteosarc test-data check openvax-v1 fixtures.json
```

`fixtures.json` maps member names to local files, or to
`{"json": path, "pointer": "/path"}` for SAM lines inside JSON. Paths are
relative to that file. Historical adapters state their SAM-text fidelity. See
Osteosarc's [shared test data guide](https://iskandr.github.io/osteosarc/test-data/#shared-test-data-openvax-v1).
