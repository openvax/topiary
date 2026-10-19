# Command-line interface

The `topiary` command runs the whole pipeline: read an antigen source, call one
or more MHC predictors, filter and rank the predictions, write a table. Every
run needs an input, a predictor (or a [cache](cached.md)) and, for allele-specific
models, a list of alleles.

```bash
topiary \
  --fasta proteins.fasta \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --ic50-cutoff 500 \
  --output-csv results.csv
```

## Predictors and alleles

Name one or more predictors with `--mhc-predictor` and alleles with
`--mhc-alleles`:

```bash
--mhc-predictor netmhcpan mhcflurry --mhc-alleles HLA-A*02:01,HLA-B*07:02
```

All predictors come from [mhctools](https://github.com/openvax/mhctools), which
also handles their installation and licensing.

| CLI name | Predicts |
|----------|----------|
| `netmhcpan` | affinity + presentation (auto-detects installed version) |
| `netmhcpan4` / `netmhcpan41` / `netmhcpan42` | specific NetMHCpan 4.x versions |
| `netmhcpan4-ba` / `netmhcpan4-el` | single-mode (binding affinity or eluted ligand) |
| `mhcflurry` | affinity + presentation + processing |
| `mixmhcpred` | presentation |
| `netmhciipan` / `netmhciipan4` / `netmhciipan43` | MHC class II |
| `bigmhc` / `bigmhc-el` / `bigmhc-im` | presentation / immunogenicity |
| `netmhcstabpan` | pMHC stability |
| `pepsickle` / `netchop` | proteasomal cleavage |
| `netmhcpan-iedb` / `netmhccons-iedb` / `smm-iedb` | local compatibility wrappers; require NetMHCpan 4.1 BA, NetMHCcons or SMM, respectively |
| `random` | random predictions (for testing) |

The historical `*-iedb` names run installed tools locally and have no HTTP
fallback. See [MHCTools' compatibility-wrapper requirements](https://openvax.github.io/mhctools/predictors/binding/#compatibility-names-for-the-old-iedb-predictors).

Peptide lengths for sliding-window scans come from the predictor unless you set
them: `--mhc-epitope-lengths 8,9,10,11`.

## Input modes

### Peptides scored as given

```bash
topiary \
  --peptide-csv peptides.csv \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --output-csv results.csv
```

The CSV needs a `peptide` column and may have a `name` column. No sliding
window is applied.

### Proteins scanned with a sliding window

```bash
topiary \
  --fasta proteins.fasta \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --output-csv results.csv
```

Restrict the scan to parts of each sequence with `--regions NAME:START-END`
(0-indexed, half-open):

```bash
--regions spike:319-541 nucleocapsid:0-50
```

### Somatic variants

```bash
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --filter-by "ba <= 500 | el.rank <= 2" \
  --sort-by "0.6 * affinity.descending_cdf(500, 200) + 0.4 * presentation.score" \
  --only-novel-epitopes \
  --output-csv epitopes.csv
```

`--maf`, `--variant` and `--json-variants` take the same pipeline from other
variant formats. [RNA expression](expression.md) can annotate and filter these
rows.

### A protein change with no genomic coordinate

Hotspots are usually named at the protein level. `--protein-change GENE CHANGE`
takes one directly, picks a protein-coding transcript whose residue at that
position matches the reference amino acid, and tiles peptides over it:

```bash
topiary \
  --protein-change EGFR T790M \
  --genome grch37 \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --output-csv hotspot.csv
```

The `variant` column is empty for these rows: no genomic position was named, so
none is invented. Read-level DNA/RNA evidence therefore cannot attach to them,
which is why this is a separate input mode. Passing it together with `--vcf`,
`--maf`, `--variant` or `--json-variants` is refused rather than mixing rows
whose evidence differs. Run each mode separately and combine the results with
`combine_sources`, which keeps each source labelled
([#421](https://github.com/openvax/topiary/issues/421)).

Where several transcripts match, topiary takes the one with the longest protein
sequence, breaking ties on transcript length and then name. A change can
therefore land on a different transcript as the annotation release changes; pass
`--ensembl-release` to pin it. The chosen transcript is named in the
`transcript_id` and `transcript_name` columns.

### Genes, transcripts and proteome-scale sources

```bash
topiary --gene-names BRAF TP53 EGFR --mhc-predictor netmhcpan --mhc-alleles HLA-A*02:01
topiary --gene-ids ENSG00000157764 --mhc-predictor netmhcpan --mhc-alleles HLA-A*02:01
topiary --transcript-ids ENST00000288602 --mhc-predictor netmhcpan --mhc-alleles HLA-A*02:01
topiary --ensembl-proteome --mhc-predictor netmhcpan --mhc-alleles HLA-A*02:01
topiary --cta --mhc-predictor netmhcpan --mhc-alleles HLA-A*02:01  # cancer-testis antigens
```

`--cta` needs the `pirlygenes` extra. The same sources are available in Python
through `topiary.sources` — see the [quickstart](quickstart.md#ensembl-and-tissue-lookups).

## Filtering and ranking

`--filter-by` and `--sort-by` take [ranking DSL](ranking.md) strings:

```bash
# Keep a peptide if affinity or presentation passes
topiary \
  --fasta proteins.fasta \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --filter-by "ba <= 500 | el.rank <= 2" \
  --output-csv results.csv

# Rank by a composite score from two models
topiary \
  --fasta antigens.fasta \
  --mhc-predictor netmhcpan mhcflurry \
  --mhc-alleles HLA-A*02:01 \
  --filter-by "ba <= 500" \
  --sort-by "mean(netmhcpan:affinity.logistic(350, 150), mhcflurry:affinity.logistic(350, 150))" \
  --output-csv results.csv
```

`--ic50-cutoff`, `--percentile-cutoff` and `--presentation-cutoff` are shorthand
for the common single-threshold filters. For variant runs, `--predict-wt` scores
each wildtype peptide too, which is what `wt.*` reads:

```bash
--predict-wt --sort-by "affinity.score - wt.affinity.score"
```

### Exclusion filtering

Remove peptides found in reference proteomes, for tumor-specific or
pathogen-specific selection:

```bash
--exclude-ensembl                    # human Ensembl proteome
--exclude-non-cta                    # non-CTA proteins (requires pirlygenes)
--exclude-tissues heart_muscle lung  # genes expressed in these tissues
--exclude-fasta reference.fasta      # custom reference sequences
--exclude-mode substring             # "substring" (default) or "exact"
```

## Output

With no output path, the CLI prints a compact preview of the first 20 prediction
rows after filtering and ranking, and reports how many rows were omitted.
`--subset-output-columns` and `--rename-output-column` also apply to this
preview. The summary on stderr counts prediction rows, unique peptide sequences,
and named alleles in those rows; several prediction kinds can give one peptide
several rows. Empty input and empty results are reported explicitly.

Explicit CSV/HTML paths save all rows and suppress the automatic preview:

```bash
--output-csv results.csv
--output-html results.html
--output-csv-sep "\t"
--subset-output-columns peptide allele affinity
--rename-output-column value ic50
```

`--output-csv -` sends every result row to stdout for a shell pipeline;
progress messages, `--print-columns` and the summary then go to stderr:

```bash
topiary --peptide-csv peptides.csv --mhc-predictor netmhcpan41 \
  --mhc-alleles 'HLA-A*02:01' --output-csv - > results.csv
```

### Columns

Every prediction has `source_sequence_name`, `peptide`, `peptide_offset`,
`peptide_length`, `allele`, `kind`, `score`, `value`, `affinity`,
`percentile_rank` and `prediction_method_name`.

Fragment-based predictions — `predict_from_fragments` and the variant-based
methods, which build fragments internally — add `fragment_id`, `source_type`,
`overlaps_target`, `wt_peptide`, `wt_peptide_length`, and any fragment-level
annotations flattened to columns. With `--predict-wt` or
`TopiaryPredictor(predict_wt=True)` they also add `wt_value`, `wt_score`,
`wt_affinity` and `wt_percentile_rank`.

Variant predictions add `variant`, `gene`, `gene_id`, `transcript_id`,
`transcript_name`, `effect`, `effect_type`, `contains_mutant_residues`,
`mutation_start_in_peptide` and `mutation_end_in_peptide`.

Expression data adds columns named `{prefix}_{column}`, such as `gene_tpm` — see
[RNA expression](expression.md#column-naming).

Units follow the row's `kind`, which output files do not restate: `value` and
`affinity` are nM for `pMHC_affinity`; `value` and `score` are 0–1 for
`pMHC_presentation` and `antigen_processing`; `percentile_rank` is a percentage.

### Saved files

Saved CSV/TSV and HTML files hold one row per prediction, with no index column,
so `pandas.read_csv(path)` and topiary's own `read_csv` / `read_tsv` read them
back as written. (Through 5.81.0 the row number was saved under the header `#`,
which readers that treat `#` as a comment — including topiary's — took for a
metadata line.)

Floats are written to six significant digits: a predictor reporting `6.296` had
been saved as `6.296000000000002`. The digits past the sixth come from binary
floating point, not from the model. Values keep their decimal point, so a
reloaded measurement column is still a float.

Row order is the ranking when `--sort-by` is given. Otherwise it follows each
source sequence's windows in position order (`peptide_offset` ascending), with
source sequences in the order they first appear; a peptide list keeps its input
order. With `--sort-by`, an `output_row` column numbers the rows from 1 so the
order survives re-sorting in a spreadsheet. It is a row position in this file,
not a per-candidate rank — see `rank_candidates` and `candidate_rank` for that.

HTML output is a complete page, and a missing value is an empty cell whatever
its type.

### Exit status

| Status | Meaning |
|---|---|
| 0 | Success |
| 1 | Something topiary found after reading the command: a missing or unreadable input, an allele a predictor rejects, a missing optional dependency, a predictor that isn't set up, an output that can't be written. One `topiary: error:` line on stderr. |
| 2 | The command line is malformed: an unknown flag, a bad value, required flags missing or in conflict. The error follows the usage text. |
| 3 | Partial result under `--cache-miss-report` (see [cached predictions](cached.md)). |

Output paths, `--mhc-predictor-path` and the inputs are checked before any
predictor is loaded, so these fail in the first second rather than after the
run.

## Installation details

`pip install topiary` requires Python 3.11 or newer and brings in Osteosarc and
its read-extraction dependency, which supply shared source data and indexed
regional read extraction; see [shared Osteosarc data](osteosarc-shared-data.md).
Importing topiary does not acquire any data.

MHC predictors are installed separately and called through
[mhctools](https://github.com/openvax/mhctools): MHCflurry with
`pip install mhcflurry && mhcflurry-downloads fetch`, NetMHCpan and the other
DTU tools under a license from [DTU Health Tech](https://services.healthtech.dtu.dk/).

Variant annotation, gene lookups and `SelfProteome` need a local Ensembl
annotation database. Install the release your PyEnsembl selects by default:

```bash
pyensembl install --release "$(python -c 'from pyensembl import ensembl_grch38; print(ensembl_grch38.release)')" --species human
```

Installing some other release does not change that default — select it
explicitly with `--ensembl-release` or the API's `release=` argument. GRCh37
inputs need release 75.

Two extras add input sources:

- `pip install 'topiary[pirlygenes]'` — cancer-testis antigens and tissue
  expression (`--cta`, `--exclude-non-cta`, `--exclude-tissues`).
- `pip install 'topiary[isovar]'` — protein fragments assembled from RNA reads.
  The extra requires Isovar 1.39.5 or newer, which Topiary also checks at run
  time, and works with the Osteosarc 0.14 and 0.15 series. Isovar is not needed
  for any other input source; see the
  [consumer guide](consumer-guide.md#fragments) for the version rules.
