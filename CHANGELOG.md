# Changelog

Each 5.x section names a published PyPI release (except the current release
while its PR is open). Changes merged under unpublished version numbers are
included in the release that shipped them. Links show the complete changes
between published tags; older pre-5.0 notes are retained below.

For current interfaces, see the [consumer guide](docs/consumer-guide.md).

## 5.94.4

- Round-trip categorical equality, inequality and membership through both DSL
  renderers and the parser, including negation, escaped strings, quoted column
  names, exact integers and missing-value literals (#472).
- Preserve categorical filters and arithmetic scores through saved selection
  policies and evidence replay without coercing text columns to numbers.

## 5.94.2

- Require gtfparse 3.0.2+ and PyEnsembl 2.24.1+, allowing the current pandas 3
  annotation stack alongside Varcode (#484). Retain pandas 2.2.2+ for Python
  3.10 and explicitly test both pandas 2 and 3 on Python 3.11.
- Select pandas output in both GTF expression readers and verify their
  transcript FPKM values against the same StringTie fixture.

## 5.94.1

- Apply explicit Cufflinks HIDATA expression replacements under pandas
  Copy-on-Write, including an explicitly requested zero. All public Cufflinks
  readers retain the corrected values; the default HIDATA drop policy is
  unchanged (#476, #482).

## 5.94.0

- Add contextual exact-peptide cache queries through the occurrence API, with
  explicit unknown-context handling, per-kind/allele/genotype coverage,
  primary/comparator replay, and version-checked fallback (#468).
- Preserve exact floating-point scores when loading text caches (#477) and
  verify model identity on every fallback batch (#478).

## 5.93.6

- Preserve case-insensitive `none` clip limits and boolean transform parameters,
  including Vaxrank's default percentile-scoring formula (#480).
- Includes the signed-parameter correction from unpublished 5.93.5:

- Decode signed scalar transform arguments and explicit unbounded clip limits
  in the DSL. Reject data-dependent parameters instead of passing expression
  nodes into numerical transforms; preserve negative bounds in saved policies
  (#471).

## 5.93.4

- Preserve arithmetic associativity, complete transform receivers, nested
  comparisons and boolean negation when rendering DSL expressions (#448).
  Saving and reparsing a rendered expression retains its scores and selections.

## 5.93.0

- Report policy coverage separately from score filling and acceptance, with
  explicit occurrence and model-request denominators, missing/failure reasons,
  and distinct raw-row counts that do not multiply allele-free projections (#457).
- Compare complete policy membership and raw scores on the shared assessable
  set; retain unknown criteria, effective scores, definitions and provenance.
  Typed CSV/TSV preserves reports and their replay inputs.
- Exercise the released Vaxrank `openvax-v1` bundle with a synthetic context
  overlay: missing short-flank evidence changes the chosen vaccine window while
  shared assessable scores remain unchanged. No default recipe changes.

## 5.92.0

- Add all-candidate self matching within an explicit same-length Hamming radius
  and exact matching across complete vaccine windows. Retain every origin,
  including shared CTA/non-CTA sequences, with caller-resolved exclusions (#455).
- Preserve supplied presentation observations, allele attribution, conflicting
  predictions and same-allele coverage separately from search completeness.
  Typed CSV/TSV retains evidence and search identity; missing evidence remains
  unknown. Neither presentation nor similarity establishes TCR recognition.
- Test against Vaxrank 3.36.0's published `builtin:openvax-v1` bundle. A synthetic
  consumer workflow changes the selected window using self evidence while
  retaining both intended epitopes and alleles; Topiary adds no selection rule.

## 5.91.0

- Score explicitly identified peptide occurrences with their original flanks,
  coordinates and source evidence, without scanning additional windows. Share
  identical inference inputs while preserving every observation; retain model,
  allele-free/per-allele/haplotype scope and explicit comparator context (#367).
- Reuse occurrence prediction in additive table rescoring, batching compatible
  peptides while retaining original measurements and selection behavior.
- Preserve the canonical prediction schema for empty named-peptide output,
  including explicitly reported all-missing cache batches (#450).
- Match fragment and explicit-occurrence WT scores by declared MHC scope, so
  a different haplotype presenter does not erase the comparator score. Export
  `prediction_mhc_scope` for consumers sharing that identity rule (#453).

## 5.90.0

- Evaluate saved policies at explicit occurrence/allele identity before selecting
  representatives. Retain all evidence, concrete genotype declarations,
  source-local model choices, score fill/gates and exact replay context (#444).
- Add reusable named DSL criteria, explicit composition and ordered tie-breaks,
  with pass/fail/unknown/not-applicable/not-evaluated audit records. Schema 2
  preserves expanded definitions; schema-1 definitions keep their digests (#445).
- Fix context derivation with genotype callbacks and changed mappings (#447).
- Verify released Vaxrank 3.35.0 scoring, window selection, peptide/mRNA
  construction and native dataset reload against the shared policy evaluator.

## 5.89.0

- Save and replay named `SelectionPolicy` definitions with complete DSL expressions,
  method/version selections, ranking settings, and a content digest. Ranked
  exports retain the policy, derivation and separate execution provenance.
  Public mapping IO accepts composed consumer configuration and freezes defaults
  without running predictors or changing scientific defaults (#442, #443).
- Preserve binary floating-point measurements and annotations through typed
  CSV/TSV reload, keeping threshold decisions and scores reproducible (#441).

## 5.88.1

- Prefer the current predictor DataFrame API for cache fallbacks, retaining
  compatibility with predictors that only implement the legacy API (#432).
- Require GTFParse's released Polars deprecation fix and mhctools' predictor
  output handle cleanup. Close a test fixture's input file explicitly.
- Assert expected scientific diagnostics in tests and fail the suite on any
  unexpected warning.

## 5.88.0

- Import native Exacto peptide variants, primary structures and translations,
  retaining original records, DNA/RNA links, alternative ORFs and unknown
  specificity. Validate recovered sequence/novelty geometry against a pinned
  public corpus; preserve transcript-scoped read membership when supplied (#365).
- Combine native observations with reported LENS/pVACseq candidates without
  prediction, preserving evidence through CSV/TSV and explicit rescoring.
  Add optional `read_exacto_fragments` for separate new-window scanning.

## 5.87.0

- Link normalized events, nucleotide ORF hypotheses, full protein products and
  peptide occurrences without collapsing alternative translations or conflicting
  source measurements. Add relational evidence views and explicit RNA count union
  that requires known membership and measurement scope (#370).
- Preserve those identities through long/wide CSV and TSV, original/additive
  ranking and supporting-hypothesis selection. Distinguish known terminal flanks
  from missing flanks in source observation IDs (#435).
- Keep empty-source validation, genotype-aware filtering and output-warning
  tests compatible with pandas 3 (#437).

## 5.86.2

- Make dtype inference explicit when stacking sparse prediction frames and
  grouping null keys, removing pandas FutureWarnings without dropping missing
  measurements or provenance. Clean up warning-producing test fixtures and TSV
  round-trip comparisons.

## 5.86.1

- Reconcile published 5.x release notes, remove completed planning documents and
  duplicate history, and correct installation/consumer documentation.

- Test released Vaxrank 3.32.0, align the Isovar minimum with Osteosarc 0.14, and
  document a repository-local development/release environment. Avoid duplicate CI
  matrices for feature-branch pushes and PR updates.

## 5.86.0

- Centralize CTA membership in `cta_gene_ids(source, tier)` and support explicit
  oncoref/tsarina sources and CTA tiers.

- Record the oncoref authority and selected tier in non-CTA reference identities; add
  the `oncoref` extra. The downloadable proteome artifact remains deferred.

[Changes](https://github.com/openvax/topiary/compare/v5.85.1...v5.86.0)

## 5.85.1

- Isolate each pytest run's temporary root and retry CI dependency installs without
  relaxing requirements.

[Changes](https://github.com/openvax/topiary/compare/v5.85.0...v5.85.1)

## 5.85.0

- Remove `topiary.pypi_release_exists` and the release CLI module from the public
  package; release preflight now lives under `scripts/`.

- Remove tests tied to incidental CI/deployment script text.

[Changes](https://github.com/openvax/topiary/compare/v5.84.3...v5.85.0)

## 5.84.3

- Replace duplicate and vacuous tests with behavioral assertions; no library behavior
  change.

[Changes](https://github.com/openvax/topiary/compare/v5.84.2...v5.84.3)

## 5.84.2

- Remove the obsolete expression specification and document producer-supplied
  `self`/`shuffled` scopes; correct release instructions.

[Changes](https://github.com/openvax/topiary/compare/v5.84.1...v5.84.2)

## 5.84.1

- Document the actual PyEnsembl default-release rule and Topiary's nearest-self
  computation.

[Changes](https://github.com/openvax/topiary/compare/v5.84.0...v5.84.1)

## 5.84.0

- Execute `--protein-change GENE CHANGE`; reject mixing it with genomic inputs until a
  combined contract is supported.

- Correct WT DSL help and display deprecated expression-flag notices.

[Changes](https://github.com/openvax/topiary/compare/v5.83.0...v5.84.0)

## 5.83.0

- Write CLI CSV without a pandas index and with six significant digits; emit complete
  HTML pages and nullable integer WT lengths.

- Preserve source-window order by default and add `output_row` for explicitly sorted
  output. Exact-text/cache replay consumers will see rounded floats.

[Changes](https://github.com/openvax/topiary/compare/v5.82.0...v5.83.0)

## 5.82.0

- Require Osteosarc 0.14 and test its compatible Isovar 1.39.5 and Vaxrank integrations.

[Changes](https://github.com/openvax/topiary/compare/v5.81.0...v5.82.0)

## 5.81.0

- Return exit 1 with concise messages for actionable runtime CLI errors; malformed
  arguments retain exit 2.

- Validate output paths, predictor paths and output-column names before
  prediction/writing.

[Changes](https://github.com/openvax/topiary/compare/v5.80.0...v5.81.0)

## 5.80.0

- Include content and indexed-length digests in `SelfProteome.reference_version`; expose
  `source` and `content_digest` and make reference identities reproducible.

[Changes](https://github.com/openvax/topiary/compare/v5.79.0...v5.80.0)

## 5.79.0

- Route DSL arguments through `as_dsl_node`/`as_dsl_nodes`; accept strings/accessors
  consistently and reject accidental Python booleans.

[Changes](https://github.com/openvax/topiary/compare/v5.78.0...v5.79.0)

## 5.78.0

- Round-trip dict/list cells as declared JSON in CSV/TSV; write structured CLI output
  cells as JSON.

[Changes](https://github.com/openvax/topiary/compare/v5.77.0...v5.78.0)

## 5.77.0

- Require Osteosarc 0.13 and update compatible integration checks.

[Changes](https://github.com/openvax/topiary/compare/v5.76.0...v5.77.0)

## 5.76.0

- Require Osteosarc 0.12 and reuse individual bundle files instead of repeatedly
  exporting the whole corpus.

[Changes](https://github.com/openvax/topiary/compare/v5.75.0...v5.76.0)

## 5.75.0

- Read Sid test records from the shared `openvax-v1` bundle; replace duplicate read
  files and local fixture-generation recipes with shared bundle members.

[Changes](https://github.com/openvax/topiary/compare/v5.74.0...v5.75.0)

## 5.74.0

- Require Osteosarc 0.11.1 and update shared-data and integration APIs.

[Changes](https://github.com/openvax/topiary/compare/v5.73.0...v5.74.0)

## 5.73.0

- Require Osteosarc 0.9 and adopt its `File` API; update integration dependency checks.

[Changes](https://github.com/openvax/topiary/compare/v5.72.0...v5.73.0)

## 5.72.0

- Require Osteosarc 0.7 and use its shared fixture generation API; test compatible
  Isovar and Vaxrank integrations.

[Changes](https://github.com/openvax/topiary/compare/v5.71.3...v5.72.0)

## 5.71.3

- Coalesce identical repeated Isovar translations without multiplying support and
  refresh the Sid regression corpus with Osteosarc 0.7.

[Changes](https://github.com/openvax/topiary/compare/v5.71.2...v5.71.3)

## 5.71.2

- Accept current Isovar protein/SV exports with read, fragment, UMI and junction support
  preserved.

[Changes](https://github.com/openvax/topiary/compare/v5.71.1...v5.71.2)

## 5.71.1

- Import Isovar protein hypotheses without changing default candidate selection; retain
  nucleotide/ORF identities and reject malformed records.

- Preserve literal text, missingness and hypothesis identities through serialization.

[Changes](https://github.com/openvax/topiary/compare/v5.70.2...v5.71.1)

## 5.70.2

- Preserve known-empty terminal flanks separately from missing context through CSV/TSV;
  support Isovar's updated rearrangement provenance.

[Changes](https://github.com/openvax/topiary/compare/v5.70.1...v5.70.2)

## 5.70.1

- Reject conflicting numeric column values within a DSL observation.

[Changes](https://github.com/openvax/topiary/compare/v5.70.0...v5.70.1)

## 5.70.0

- Add SV nomination reports with explicit protein/RNA evidence tiers, candidate-local
  sequence-change ranking and reference-conflict validation.

[Changes](https://github.com/openvax/topiary/compare/v5.69.0...v5.70.0)

## 5.69.0

- Add opt-in partial results with explicit cache-miss reports; accept compatible
  Osteosarc 0.2.3 patch releases.

[Changes](https://github.com/openvax/topiary/compare/v5.68.4...v5.69.0)

## 5.68.4

- Delegate shared Sid fixture generation and verification to Osteosarc's public API.

[Changes](https://github.com/openvax/topiary/compare/v5.68.3...v5.68.4)

## 5.68.3

- Preserve imported mutation geometry and comparator scope; provision and verify the
  dependency-selected Ensembl reference in CI.

[Changes](https://github.com/openvax/topiary/compare/v5.68.2...v5.68.3)

## 5.68.2

- Reject conflicting measurements during DSL evaluation; prefer exact model names and
  reject ambiguous partial matches.

[Changes](https://github.com/openvax/topiary/compare/v5.68.1...v5.68.2)

## 5.68.1

- Refresh installation and consumer documentation; document isolated release
  environments.

[Changes](https://github.com/openvax/topiary/compare/v5.68.0...v5.68.1)

## 5.68.0

- Combine labelled source tables without prediction, preserve original observations, and
  add explicit additive rescoring and duplicate-selection policies.

- Preserve model/version mappings in wide files and verify combined evidence through
  Vaxrank scoring/construction.

[Changes](https://github.com/openvax/topiary/compare/v5.67.1...v5.68.0)

## 5.67.1

- Require Osteosarc 0.1.2 without changing pinned scientific fixtures.

[Changes](https://github.com/openvax/topiary/compare/v5.67.0...v5.67.1)

## 5.67.0

- Require Osteosarc 0.1.1 and delegate catalogue parsing and Sid fixture
  acquisition/generation to its public API.

[Changes](https://github.com/openvax/topiary/compare/v5.66.0...v5.67.0)

## 5.66.0

- Require Python 3.10+ and install Osteosarc, including read-extraction support, as a
  base dependency.

[Changes](https://github.com/openvax/topiary/compare/v5.65.2...v5.66.0)

## 5.65.2

- Quote ambiguous extra-metadata text so line breaks, whitespace and JSON-like strings
  survive CSV/TSV round-trips.

[Changes](https://github.com/openvax/topiary/compare/v5.65.1...v5.65.2)

## 5.65.1

- Reject reserved or malformed top-level extra-metadata keys before opening output
  files; preserve valid structured extras.

[Changes](https://github.com/openvax/topiary/compare/v5.65.0...v5.65.1)

## 5.65.0

- Keep explicit unannotated-region outcomes in RNA audits; delegate data acquisition to
  Osteosarc and include workflow scripts in sdists.

[Changes](https://github.com/openvax/topiary/compare/v5.64.0...v5.65.0)

## 5.64.0

- Separate candidate `fragment_id` from `sample_name`; use the pair for observation
  identity and carry sample labels from fragments into predictions.

- Normalize duplicate content through fragment IO, retain transcript identity, and
  validate Isovar filter outcomes and malformed audit rows.

[Changes](https://github.com/openvax/topiary/compare/v5.63.0...v5.64.0)

## 5.63.0

- Derive fragment identities from producer grouping, add sample labelling, and share
  Isovar result/filter handling.

- Validate optional-dependency floors at runtime and preserve per-entry audit errors.
  Sample identity handling is revised in 5.64.0.

[Changes](https://github.com/openvax/topiary/compare/v5.62.0...v5.63.0)

## 5.62.0

- Add explicit outcomes for all 184 audited Osteosarc entries and
  `describe_isovar_result` for empty/filtered results.

- Reject contradictory fragment IDs before prediction and require Isovar 1.18.1 for
  corrected insertion-boundary evidence.

[Changes](https://github.com/openvax/topiary/compare/v5.61.0...v5.62.0)

## 5.61.0

- Add `join_annotations` for namespaced evidence joins with provenance and a
  reproducible RNA overlay for the archived Osteosarc reports.

[Changes](https://github.com/openvax/topiary/compare/v5.60.3...v5.61.0)

## 5.60.3

- Add source-pinned rearrangement RNA fixtures for GABBR1–SLC29A1 and OTUD7A–FMN1.

[Changes](https://github.com/openvax/topiary/compare/v5.60.2...v5.60.3)

## 5.60.2

- Add original-read GLIS3/KTN1 fixtures with exact reference context and separate
  diagnostic reconstruction from default acceptance.

[Changes](https://github.com/openvax/topiary/compare/v5.60.1...v5.60.2)

## 5.60.1

- Add original-source fixtures for all 21 Osteosarc pVACseq reports while retaining
  missing RNA annotations as missing.

[Changes](https://github.com/openvax/topiary/compare/v5.60.0...v5.60.1)

## 5.60.0

- Require Isovar 1.17.0 for RNA integration and preserve sequenced support under its
  corrected assembly rules.

[Changes](https://github.com/openvax/topiary/compare/v5.59.2...v5.60.0)

## 5.59.2

- Require mhctools 3.44.26 for model provenance; honor missing-provenance options across
  cache loaders and explain generic TSV kind requirements.

[Changes](https://github.com/openvax/topiary/compare/v5.59.1...v5.59.2)

## 5.59.1

- Honor requested peptide lengths in cached protein scans; expose available lengths
  separately from selected scan lengths.

[Changes](https://github.com/openvax/topiary/compare/v5.59.0...v5.59.1)

## 5.59.0

- Show a bounded prediction preview when no output file is requested; add complete CSV
  output through `--output-csv -` and explicit result counts.

[Changes](https://github.com/openvax/topiary/compare/v5.58.0...v5.59.0)

## 5.58.0

- Require Isovar 1.11.0 and update the H1-2 deletion regression to check its supported
  reconstruction.

- Honor requested alleles and lengths in CLI cache mode, reject simultaneous live/cache
  sources, and classify whole-peptide half-life evidence.

[Changes](https://github.com/openvax/topiary/compare/v5.57.0...v5.58.0)

## 5.57.0

- Align declared dependencies with imports, repair GTF expression loading with gtfparse
  2, and remove unused private CLI/filter plumbing.

- Add `PredictorSetupError`, make cache-coverage messages readable to library callers,
  and classify the added pharmacokinetic prediction kinds.

[Changes](https://github.com/openvax/topiary/compare/v5.56.1...v5.57.0)

## 5.56.1

- Expose `CachedPredictorCoverageError` for intentional cache gaps; preserve useful
  file-error messages and let unrelated programming errors propagate.

[Changes](https://github.com/openvax/topiary/compare/v5.55.1...v5.56.1)

## 5.55.1

- Rebind cached protein scans to the requested occurrence's coordinates/flanks and
  reject missing kind/genotype coverage instead of silently dropping it.

- Support structured annotations in wide conversion, report cache coverage errors
  cleanly in the CLI, and fail closed when PyPI release status cannot be verified.

[Changes](https://github.com/openvax/topiary/compare/v5.53.0...v5.55.1)

## 5.53.0

- Expose peptide-aware RNA reconstruction options and integrate whole-peptide half-life
  predictions from mhctools.

[Changes](https://github.com/openvax/topiary/compare/v5.52.10...v5.53.0)

## 5.52.10

- Make CI coverage retries reliable and keep test collection offline.

[Changes](https://github.com/openvax/topiary/compare/v5.52.9...v5.52.10)

## 5.52.9

- Require Isovar 1.7.10 for corrected RNA assembly; test minimum and latest
  supported releases.

[Changes](https://github.com/openvax/topiary/compare/v5.52.8...v5.52.9)

## 5.52.8

- Restrict wheels to runtime packages, preserve source/test resources in sdists, and
  validate packaging metadata.

[Changes](https://github.com/openvax/topiary/compare/v5.52.7...v5.52.8)

## 5.52.7

- Test PirlyGenes both absent and installed in CI.

[Changes](https://github.com/openvax/topiary/compare/v5.52.6...v5.52.7)

## 5.52.6

- Make nearest-self distances symmetric for ambiguous amino acids.

[Changes](https://github.com/openvax/topiary/compare/v5.52.5...v5.52.6)

## 5.52.5

- Use the same Python interpreter for lint, tests and deployment.

[Changes](https://github.com/openvax/topiary/compare/v5.52.4...v5.52.5)

## 5.52.4

- Use a normal, introspectable `ProteinFragment` constructor.

[Changes](https://github.com/openvax/topiary/compare/v5.52.3...v5.52.4)

## 5.52.3

- Cache model-name resolution without mutable function-default state.

[Changes](https://github.com/openvax/topiary/compare/v5.52.2...v5.52.3)

## 5.52.2

- Centralize amino-acid data in a public API.

[Changes](https://github.com/openvax/topiary/compare/v5.52.1...v5.52.2)

## 5.52.1

- Read optional-dependency requirements from package metadata.

[Changes](https://github.com/openvax/topiary/compare/v5.52.0...v5.52.1)

## 5.52.0

- Preserve pVACtools presentation, processing and immunogenicity measurements as native
  prediction rows; share external metric parsing with LENS.

[Changes](https://github.com/openvax/topiary/compare/v5.51.0...v5.52.0)

## 5.51.0

- Use `EvalContext.df` for the frame DSL nodes evaluate; remove the redundant
  `evaluation_df` attribute and document custom-node grouping.

[Changes](https://github.com/openvax/topiary/compare/v5.49.1...v5.51.0)

## 5.49.1

- Harden cross-sample aggregation against equivalent identities and inconsistent
  observations.

[Changes](https://github.com/openvax/topiary/compare/v5.49.0...v5.49.1)

## 5.49.0

- Add strict cross-sample evidence aggregation with explicit units and sample handling.

[Changes](https://github.com/openvax/topiary/compare/v5.48.1...v5.49.0)

## 5.48.1

- Fix lazy discovery of mhctools model names.

[Changes](https://github.com/openvax/topiary/compare/v5.48.0...v5.48.1)

## 5.48.0

- Make evidence units assay-specific and omit unavailable all-null evidence columns
  consistently.

- Coalesce identical cache rows and reject conflicting measurements in both constructors
  and concatenation; correct aggregated pVACseq expression semantics.

[Changes](https://github.com/openvax/topiary/compare/v5.47.0...v5.48.0)

## 5.47.0

- Add symmetric DNA/RNA evidence, canonical VAFs and third-allele counts; preserve
  source-prefixed originals and export `PREDICTION_KEY_COLUMNS`.

[Changes](https://github.com/openvax/topiary/compare/v5.46.0...v5.47.0)

## 5.46.0

- Document source-dependent evidence availability; add `available_evidence_columns` and
  `EVIDENCE_COLUMNS`, and include docs in source distributions.

[Changes](https://github.com/openvax/topiary/compare/v5.45.1...v5.46.0)

## 5.45.1

- Add the downstream consumer guide.

[Changes](https://github.com/openvax/topiary/compare/v5.45.0...v5.45.1)

## 5.45.0

- Replace the earlier count-unit API with canonical `n_rna_*` quantities, explicit
  read/fragment counts, evidence subjects and derivation methods. Remove `count_in`,
  `read_count_subject`, `count_column_for_subject` and `subject_for_method`.

- Keep LENS's unstated-assay VAF distinct from RNA/DNA fractions and stop labelling CDS-
  overlap counts as assembled-protein support. Rename estimated expression to
  `rna_alt_expression`.

[Changes](https://github.com/openvax/topiary/compare/v5.44.0...v5.45.0)

## 5.44.0

- Carry read-count units through fragment prediction and enforce unit-aware access. This
  API is superseded by 5.45.0.

[Changes](https://github.com/openvax/topiary/compare/v5.43.0...v5.44.0)

## 5.43.0

- Record whether read evidence counts reads or fragments. This API is superseded by
  5.45.0.

[Changes](https://github.com/openvax/topiary/compare/v5.42.0...v5.43.0)

## 5.42.0

- Use consistent RNA evidence names for aggregated and all-epitopes pVACseq reports;
  record source-reported estimates.

[Changes](https://github.com/openvax/topiary/compare/v5.41.0...v5.42.0)

## 5.41.0

- Standardize gene-level expression under `gene_expression`.

[Changes](https://github.com/openvax/topiary/compare/v5.40.0...v5.41.0)

## 5.40.0

- Add `fragments_from_variants` for RNA-backed reconstruction through Isovar alongside
  the existing reference-translation path.

[Changes](https://github.com/openvax/topiary/compare/v5.39.1...v5.40.0)

## 5.39.1

- Add composed consumer-workflow regression coverage.

[Changes](https://github.com/openvax/topiary/compare/v5.39.0...v5.39.1)

## 5.39.0

- Keep named-allele peptide-level predictions scoped to their stated allele; preserve
  distinct allele observations.

[Changes](https://github.com/openvax/topiary/compare/v5.38.0...v5.39.0)

## 5.38.0

- Add Isovar-result-to-fragment conversion with lazy optional imports, shared semantic
  fields and recorded evidence derivations.

[Changes](https://github.com/openvax/topiary/compare/v5.37.0...v5.38.0)

## 5.37.0

- Import RNA evidence from LENS and pVACseq with derivation provenance; add estimated
  allele expression and `sequence_source`.

[Changes](https://github.com/openvax/topiary/compare/v5.36.0...v5.37.0)

## 5.36.0

- Include genotype (`allele_set`) in cache identity so different haplotype predictions
  cannot collide.

[Changes](https://github.com/openvax/topiary/compare/v5.35.1...v5.36.0)

## 5.35.1

- Keep missing cache alleles missing instead of creating an allele named `None`.

[Changes](https://github.com/openvax/topiary/compare/v5.35.0...v5.35.1)

## 5.35.0

- Export `fragment_from_effect`, `is_named_version`, `known_versions` and shared
  missing-value helpers.

- Validate fragment padding and clamp sequence/comparator intervals at stop codons.

[Changes](https://github.com/openvax/topiary/compare/v5.34.0...v5.35.0)

## 5.34.0

- Reject selecting missing versions as the string `nan`; avoid fabricating pVACseq
  variant IDs from missing coordinates.

[Changes](https://github.com/openvax/topiary/compare/v5.33.0...v5.34.0)

## 5.33.0

- Support per-peptide allele sets in `EvalContext` and add `describe_default_versions`.

[Changes](https://github.com/openvax/topiary/compare/v5.32.0...v5.33.0)

## 5.32.0

- Add read evidence and per-field knownness to `ProteinFragment`; preserve unknown
  versus zero through IO and accept all dataclass fields during loading.

[Changes](https://github.com/openvax/topiary/compare/v5.31.1...v5.32.0)

## 5.31.1

- Return no nearest-self match for empty length buckets and reject
  annotation/prediction-name collisions during wide conversion.

[Changes](https://github.com/openvax/topiary/compare/v5.31.0...v5.31.1)

## 5.31.0

- Add `default_versions` and `resolve_default_versions`; treat unstated versions as
  missing rather than as selectable model versions.

[Changes](https://github.com/openvax/topiary/compare/v5.30.0...v5.31.0)

## 5.30.0

- Add `read_lens(binding_metrics=...)` overrides with recorded provenance and actionable
  unmapped-column warnings.

[Changes](https://github.com/openvax/topiary/compare/v5.29.0...v5.30.0)

## 5.29.0

- Allow reuse of an `EvalContext` on the same unchanged frame and expose
  `default_methods` on predictors and result filtering/sorting.

[Changes](https://github.com/openvax/topiary/compare/v5.28.2...v5.29.0)

## 5.28.2

- Preserve multiple predictor versions in LENS and wide/long conversion; reject
  ambiguous unqualified version selection.

- Derive MHC class through mhcgnomes rather than allele-name prefixes.

[Changes](https://github.com/openvax/topiary/compare/v5.28.0...v5.28.2)

## 5.28.0

- Recognize additional predictor-version spellings in LENS; warn about unrecognized
  prediction columns while retaining their values.

[Changes](https://github.com/openvax/topiary/compare/v5.27.0...v5.28.0)

## 5.27.0

- Allow custom columns and per-prediction genotype sets in `from_predictions`. Canonical
  method selection remains opt-in and can change a consumer's scores.

[Changes](https://github.com/openvax/topiary/compare/v5.26.0...v5.27.0)

## 5.26.0

- Add `predict_self_nearest` for paired predictions against the nearest self peptide.
  Comparator prediction uses peptide sequence without reference flanks.

- Allow scoped fields in filters, warning when comparator columns are absent. Add
  `from_predictions` and opt-in canonical method resolution.

[Changes](https://github.com/openvax/topiary/compare/v5.22.0...v5.26.0)

## 5.22.0

- Expose `KIND_MHC_DEPENDENCE` and `mhc_dependence`; reject malformed allele-scoped rows
  instead of treating them as allele-free.

[Changes](https://github.com/openvax/topiary/compare/v5.21.1...v5.22.0)

## 5.21.1

- Make sorting with missing values deterministic and independent of input row order.

[Changes](https://github.com/openvax/topiary/compare/v5.21.0...v5.21.1)

## 5.21.0

- Store genotype context in `allele_set`, include it in grouping, and add
  `Column.includes` for set membership.

[Changes](https://github.com/openvax/topiary/compare/v5.20.1...v5.21.0)

## 5.20.1

- Apply the same haplotype projection warnings to labelled and unlabelled kinds.

[Changes](https://github.com/openvax/topiary/compare/v5.20.0...v5.20.1)

## 5.20.0

- Automatically project unqualified allele-free fields, with a warning; continue
  rejecting inconsistent values for one peptide.

[Changes](https://github.com/openvax/topiary/compare/v5.19.0...v5.20.0)

## 5.19.0

- Project allele-free predictions onto explicit patient alleles and preserve that
  evidence through allele-scoped filters.

[Changes](https://github.com/openvax/topiary/compare/v5.18.1...v5.19.0)

## 5.18.1

- Resolve peptide-level projection from the selected model's metadata, reject
  contradictory values and unknown dependence modes, and pass kind support through
  object APIs.

- Apply scoped-field filter restrictions to best-allele fields; these restrictions are
  lifted in 5.26.0.

[Changes](https://github.com/openvax/topiary/compare/v5.18.0...v5.18.1)

## 5.18.0

- Add `peptide_view` with MHC-dependence-aware projection; fix automatic sort direction
  for best-allele fields.

[Changes](https://github.com/openvax/topiary/compare/v5.17.1...v5.18.0)

## 5.17.1

- Expose explicit group keys across DSL operations; preserve null identities and support
  single-column grouping. Context options are keyword-only and empty group-key lists
  raise.

- Fix tissue-expression lookups against current PirlyGenes.

[Changes](https://github.com/openvax/topiary/compare/v5.16.2...v5.17.1)

## 5.16.2

- Add `combine_predictions` for separate predictor runs, with strict coverage
  checks and explicit sparse unions.

[Changes](https://github.com/openvax/topiary/compare/v5.16.1...v5.16.2)

## 5.16.1

- Add `read_pvacseq`, MHC-class filters, mutation-overlap annotations, and categorical
  DSL comparisons.

- Update tissue-expression integration for PirlyGenes 5.1.0.

[Changes](https://github.com/openvax/topiary/compare/v5.15.0...v5.16.1)

## 5.15.0

- Populate raw values from scores for prediction kinds whose values are defined on the
  same [0, 1] scale.

[Changes](https://github.com/openvax/topiary/compare/v5.14.1...v5.15.0)

## 5.14.1

- Require Varcode 4.18.0, which removes PyVCF3 from the runtime import path and
  avoids embedded-R import noise.

[Changes](https://github.com/openvax/topiary/compare/v5.14.0...v5.14.1)

## 5.14.0

- Expose peptide properties as DSL nodes.

[Changes](https://github.com/openvax/topiary/compare/v5.13.0...v5.14.0)

## 5.13.0

- Add best-allele aggregation for haplotype-mode presentation.

[Changes](https://github.com/openvax/topiary/compare/v5.12.0...v5.13.0)

## 5.12.0

- Export `KIND_ALIASES` as a public constant.

[Changes](https://github.com/openvax/topiary/compare/v5.11.0...v5.12.0)

## 5.11.0

- Carry mhctools `kind_support` metadata through Topiary predictors.

[Changes](https://github.com/openvax/topiary/compare/v5.10.8...v5.11.0)

## 5.10.8

- Restore Varcode's CLI argument parser while retaining lazy non-CLI imports.

[Changes](https://github.com/openvax/topiary/compare/v5.10.7...v5.10.8)

## 5.10.7

- Delay Varcode imports until variant-dependent operations run.

[Changes](https://github.com/openvax/topiary/compare/v5.10.6...v5.10.7)

## 5.10.6

- Reduce cache index memory use and speed up peptide/allele/length lookups.

[Changes](https://github.com/openvax/topiary/compare/v5.10.5...v5.10.6)

## 5.10.5

- Use one configurable Python interpreter for deployment and remove stale build outputs
  before packaging.

[Changes](https://github.com/openvax/topiary/compare/v5.10.4...v5.10.5)

## 5.10.4

- Remove experimental colon-separated version syntax; use
  `mhcflurry[release-2.2.0]:ba.score`.

[Changes](https://github.com/openvax/topiary/compare/v5.10.3...v5.10.4)

## 5.10.3

- Accept quote-free model-qualified DSL forms such as `mhcflurry:affinity` and bracketed
  model versions.

[Changes](https://github.com/openvax/topiary/compare/v5.10.2...v5.10.3)

## 5.10.2

- Add `predict_wt=True` and `--predict-wt`; join wildtype scores by model, version,
  kind, allele and peptide length. Rows without a compatible WT peptide retain missing
  scores.

[Changes](https://github.com/openvax/topiary/compare/v5.10.1...v5.10.2)

## 5.10.1

- Report missing CLI inputs and prediction sources as argument errors.

[Changes](https://github.com/openvax/topiary/compare/v5.10.0...v5.10.1)

## 5.10.0

- Add row-aligned `evaluate_scores`, `default_methods` for multi-model DSL evaluation,
  and bare model identifiers in kind brackets.

- Unqualified directional filter comparisons accept a candidate when any model passes.

[Changes](https://github.com/openvax/topiary/compare/v5.9.1...v5.10.0)

## 5.9.1

- Clarify newcomer documentation and add repository contribution/release instructions.

[Changes](https://github.com/openvax/topiary/compare/v5.9.0...v5.9.1)

## 5.9.0

- Rename the self-proteome `scope` argument to `include`; add protected-tissue
  selection, BLOSUM62 distances and one-residue indel candidates.

[Changes](https://github.com/openvax/topiary/compare/v5.8.0...v5.9.0)

## 5.8.0

- Add `SelfProteome` for nearest-self sequence lookup from Ensembl, FASTA or explicit
  peptides, with optional CTA exclusion.

[Changes](https://github.com/openvax/topiary/compare/v5.7.0...v5.8.0)

## 5.7.0

- Preserve separate prediction kinds in cached outputs, parse multi-allele NetMHC output
  correctly, and include flanks in cache keys. Generic TSV caches now require `kind`.

[Changes](https://github.com/openvax/topiary/compare/v5.6.0...v5.7.0)

## 5.6.0

- Add NetMHC-family cache loaders and cache sharding with `concat`/`from_directory`,
  including explicit overlap policies.

- Validate fallback identities and reject saving an unidentified empty cache.

[Changes](https://github.com/openvax/topiary/compare/v5.5.0...v5.6.0)

## 5.5.0

- Add `CachedPredictor` with dataframe, TSV, Topiary and MHCflurry loaders, strict
  model/version identity, and live fallback.

- Add `mhcflurry_composite_version` to identify both package and model-data versions.

[Changes](https://github.com/openvax/topiary/compare/v5.4.0...v5.5.0)

## 5.4.0

- Rename `AntigenFragment` to `ProteinFragment`, `predict_from_antigens` to
  `predict_from_fragments`, and antigen IO helpers to fragment IO helpers. Old Python
  names are removed; existing TSV files remain readable.

- Route variant prediction through fragments while preserving absolute offsets and
  variant metadata. Add `transcript_name`.

[Changes](https://github.com/openvax/topiary/compare/v5.2.0...v5.4.0)

## 5.2.0

- Add the universal `AntigenFragment` record, fragment IO and prediction, and the
  reserved `self_nearest` DSL scope.

[Changes](https://github.com/openvax/topiary/compare/v5.1.0...v5.2.0)

## 5.1.0

- Add `read_lens` and `logistic_normalized`; normalize alleles through mhcgnomes.

[Changes](https://github.com/openvax/topiary/compare/v5.0.1...v5.1.0)

## 5.0.1

- Add `DSLNode.child_nodes()` for custom nodes and preserve row alignment when
  filtering.

[Changes](https://github.com/openvax/topiary/compare/v5.0.0...v5.0.1)

## 5.0.0

- Replace filter/ranker classes with one DSL node tree. Use `parse`, `apply_filter`,
  `apply_sort`, and predictor `filter_by`/`sort_by` arguments; old parser and strategy
  names are removed.

- Add version-qualified fields, expression serialization, and explicit errors for
  ambiguous models and non-boolean filters.

[Changes](https://github.com/openvax/topiary/releases/tag/v5.0.0)

## 4.12.0

**Breaking changes:**

- `topiary.read_tsv` and `topiary.read_csv` now return a `TopiaryResult`
  instead of an `(DataFrame, Metadata)` tuple. Callers using tuple
  unpacking must migrate: `df, meta = read_tsv(path)` →
  `result = read_tsv(path); df, meta = result.df, result.metadata`.

**New features:**

- `TopiaryResult` class bundling a predictions DataFrame with provenance
  (model versions, source files, form, filter/sort history).  Delegates
  common DataFrame operations (`len`, `iter`, `columns`, `shape`, `head`,
  `iterrows`, etc.) so most existing DataFrame-style code continues to
  work.  Provides `to_wide()`, `to_long()`, `to_tsv()`, `to_csv()`,
  `filter_by()`, `sort_by()`.
- `topiary.stack_results([r1, r2, ...])` merges `TopiaryResult`s, unioning
  models (warns on version conflicts), concatenating sources, and
  preserving filter/sort history only if all inputs agree.
- `read_tsv` / `read_csv` accept a `tag=` kwarg to label the source of
  the loaded rows; defaults to the filename.  Auto-populates a `source`
  column on the DataFrame.
- `Metadata` gains a `sources: list[str]` field; the comment block
  supports multiple `#source=...` lines.

**Deprecations (removed in 5.0 alongside the DSL refactor,
[#111](https://github.com/openvax/topiary/issues/111)):**

- `EpitopeFilter`, `ColumnFilter`, `ExprFilter`, `RankingStrategy`
  replaced by a unified `Comparison` / `BoolOp` DSL tree. See the 5.0.0
  entry above for migration details.

## 4.9.0

- Require `mhctools>=3.7.0`.
- Rename CLI sorting flags to `--sort-by` and `--sort-direction`.
- Add Python API `sort_by=` and `.sort_by(...)`, while keeping `rank_by` as a compatibility alias.
- Treat comma-separated `--sort-by` keys as lexicographic tie breakers, with fallthrough on missing values.
- Document upstream `mhctools 3.7.0+` support for multi-predictor CLI invocations, the simplified `Kind` API, and the updated NetChop/Pepsickle behavior.
