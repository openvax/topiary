# Reject ambiguous extra metadata keys (#352)

TSV/CSV extras currently share the comment-key namespace with built-in
metadata. Writing `extra={"source": "dataset digest"}` silently turns it into
file provenance on read. Reject ambiguous keys at serialization rather than
introducing a new file format or interpreting old files differently.

- Both writers reject `topiary_version`, `form`, `source`, `filter_by`,
  `sort_by`, and every `model:` key in top-level extras.
- Keys must be nonempty strings without surrounding whitespace, line breaks
  or `=`; these cannot retain their identity in the comment syntax.
- Validation happens before the output is opened, including extras added
  after result construction and explicit metadata overrides.
- Valid custom names and JSON-compatible structured values retain their
  existing representation. Reserved names nested inside custom metadata are
  ordinary data. Existing files retain their current parsing behavior.
- Register TSV/CSV and result/function writer paths as twins. Regressions
  exercise every reserved key, malformed keys, unchanged existing files,
  fresh destinations, legacy reads and structured round-trips. A composed
  consumer test filters/sorts, writes, reads and verifies nested provenance.

Ship 5.65.1 after lint, the full suite and GitHub CI pass, then deploy from
clean master and verify PyPI artifacts and the release tag.
