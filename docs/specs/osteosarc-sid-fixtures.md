# All Sid fixtures through Osteosarc

Use published Osteosarc 0.1.1 as a required dependency. Replace the local
website/VAF parser with `parse_variants`, preserving per-entry diagnostics and
the existing audit record format. Historical catalogue alleles and scientific
expectations remain separately pinned; this does not resolve upstream #5.

Cover all seven Sid fixture groups (six-locus RNA, indels, rearrangements,
RNA overlay, all-variant audit, shared vaccine reads and pVACseq reports) in
one content-pinned export manifest. Reuse `osteosarc_fixture_paths` and the
explicit acquisition/export CLI for the whole bundle. Tests verify the bundled
objects offline through Osteosarc; regeneration acquires through its cache
and extracts only relevant alignment regions. Use a checked-in recipe with allele intervals plus two bases of padding.
Remove unused vaccine read sets, keeping only NTF3. Full alignments never enter
the package; fixtures stay with tests in the source distribution, not the wheel. Keep original SAM records, source URLs, selection policies and
independent reference/expected-result fixtures.

Validate the malformed-row battery against both public parsing doors, compare
complete read records through real extraction, replay all 184 audit outcomes,
and check every bundled asset's size/hash in built distributions. Run lint,
full tests, CI and the clean-master deployment gate for release 5.67.0.
