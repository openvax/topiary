# Required Osteosarc dependency

Osteosarc must be installed by ordinary Topiary installation, including its
read-extraction dependency. Move `osteosarc[reads]==0.1.0` from an optional extra
to the base requirements and require Python 3.10+, matching Osteosarc's published
minimum. Release as 5.66.0 because Python 3.9 is no longer supported.

Run the existing offline cache, asset integrity and complete read-record tests
in base CI. Keep the Osteosarc marker for selecting tests, but remove its
optional-dependency skip path. Install samtools in base CI and retain the
Isovar reconstruction/prediction workflow in the Isovar integration jobs.
Verify both distribution metadata and a clean wheel installation without
extras, then run lint, the full suite, CI and the normal deployment gate.

The upstream catalogue gaps are tracked in [Osteosarc #4](https://github.com/iskandr/osteosarc/issues/4)
(per-entry parsing continuity) and [#5](https://github.com/iskandr/osteosarc/issues/5)
(remaining allele resolution). Historical audit inputs remain pinned while those gaps are addressed;
requiring the package does not itself change alleles or scientific outcomes.
