# Preserve plain-text extra values (#357)

Extra metadata strings currently enter the comment syntax verbatim. Embedded
CR/LF can introduce metadata or data rows, surrounding whitespace disappears,
and the `json:` marker or legacy `kind_support` syntax can change a string's
type. Reuse the existing JSON-string encoding to keep these values as data.

- Keep ordinary text in its current human-readable form.
- JSON-quote text containing CR/LF, surrounding whitespace or the `json:`
  prefix. Quote all text under `kind_support`, whose legacy reader can parse
  unmarked dictionaries. Existing dictionary/list serialization stays intact.
- Apply the same text encoding after fallback string conversion of unsupported
  objects. Retain existing scalar/temporal conversion behavior.
- Leave the reader unchanged, preserving old plain-text and structured-extra
  files. New escaped strings must also be readable by the 5.65.1 decoder.
- Drive the registered TSV/CSV twins through function, result-method and
  explicit-metadata paths. Check exact string types/content, built-in metadata,
  data rows, repeated write/read cycles and output preservation on conversion
  failures. Extend the filter/sort/file consumer workflow with ambiguous text.

Bump to 5.65.2. Run lint, focused regressions, the full test suite and GitHub
CI. Follow the repository's merge/deploy gates and verify the PyPI artifacts.
