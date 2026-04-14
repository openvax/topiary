[![Tests](https://github.com/openvax/topiary/actions/workflows/tests.yml/badge.svg)](https://github.com/openvax/topiary/actions/workflows/tests.yml)
<a href="https://coveralls.io/github/openvax/topiary?branch=master">
    <img src="https://coveralls.io/repos/openvax/topiary/badge.svg?branch=master&service=github" alt="Coverage Status" />
</a>
<a href="https://pypi.python.org/pypi/topiary/">
    <img src="https://img.shields.io/pypi/v/topiary.svg?maxAge=1000" alt="PyPI" />
</a>

# Topiary

Topiary predicts, filters, and ranks MHC-presented peptides from any antigen source. It wraps [mhctools](https://github.com/openvax/mhctools) binding predictors (NetMHCpan, MHCflurry, etc.) with a composable filtering and ranking DSL, expression-aware prioritization, and wide/long serialization.

Applications include personalized cancer vaccine design, viral epitope mapping, and characterizing T-cell responses.

## Installation

```bash
pip install topiary
```

For variant annotation and gene lookups, download Ensembl reference data:

```bash
pyensembl install --release 110 --species human
```

## Predicting MHC binding

The simplest use case: score a handful of peptides against one or more HLA alleles.

```python
from topiary import TopiaryPredictor
from mhctools import NetMHCpan

predictor = TopiaryPredictor(
    models=[NetMHCpan],
    alleles=["HLA-A*02:01"],
)

df = predictor.predict_from_named_peptides({
    "peptide_1": "YLQLVFGIEV",
    "peptide_2": "LLFNILGGWV",
    "peptide_3": "GILGFVFTL",
})
```

The result is a DataFrame with one row per peptide-allele-kind combination. Each row has `score` (normalized, higher = better), `value` (raw prediction, e.g. IC50 nM), and `percentile_rank` (lower = better).

To scan full-length proteins with a sliding window instead:

```python
df = predictor.predict_from_named_sequences({
    "BRAF_V600E": "MAALSGGGGG...LATEKSRWSG",
})
```

Multiple models can run together. The DataFrame will have rows for each model's prediction kinds:

```python
from mhctools import NetMHCpan, MHCflurry

predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
)
```

## Filtering and ranking

Topiary has an expression language for filtering and ranking predictions. It works identically as Python objects and as CLI strings.

### Filters

Keep only peptides that pass a threshold:

```python
from topiary import TopiaryPredictor, Affinity, Presentation
from mhctools import NetMHCpan

predictor = TopiaryPredictor(
    models=[NetMHCpan],
    alleles=["HLA-A*02:01"],
    filter_by=Affinity <= 500,                       # IC50 nM cutoff
)

# Or with presentation score:
predictor = TopiaryPredictor(
    models=[NetMHCpan],
    alleles=["HLA-A*02:01"],
    filter_by=(Affinity <= 500) | (Presentation.rank <= 2.0),  # keep if either passes
)
```

### Prediction kinds and fields

| Accessor | Aliases | What it measures |
|----------|---------|------------------|
| `Affinity` | `ba`, `aff`, `ic50` | Binding affinity (IC50 nM) |
| `Presentation` | `el` | Eluted ligand / presentation score |
| `Stability` | | pMHC complex stability |
| `Processing` | | Antigen processing / cleavage |

Each kind has three fields:

- `.value` — raw prediction (e.g. IC50 in nM). Default when no field is specified, so `Affinity <= 500` means `Affinity.value <= 500`.
- `.rank` — percentile rank (lower = better).
- `.score` — normalized score (higher = better).

### Ranking with transforms

Sort surviving peptides with `sort_by`. Transforms normalize heterogeneous scores to a common scale:

```python
from topiary import TopiaryPredictor, Affinity, Presentation, mean
from mhctools import NetMHCpan, MHCflurry

predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["HLA-A*02:01"],
    filter_by=Affinity <= 500,
    sort_by=mean(
        Affinity["netmhcpan"].logistic(350, 150),
        Affinity["mhcflurry"].logistic(350, 150),
    ),
)
```

When multiple models produce the same kind, qualify with brackets: `Affinity["netmhcpan"]`.

Available transforms:

| Transform | What it does |
|-----------|-------------|
| `.descending_cdf(mean, std)` | Lower input -> higher output (for IC50, rank) |
| `.ascending_cdf(mean, std)` | Higher input -> higher output (for scores) |
| `.logistic(midpoint, width)` | Sigmoid normalization |
| `.clip(lo, hi)` | Clamp to range |
| `.hinge()` | `max(0, x)` |
| `.log()` / `.log2()` / `.log10()` / `.log1p()` | Logarithms |
| `.sqrt()` / `.exp()` | Square root, exponential |
| `abs(...)` | Absolute value |

Aggregations: `mean()`, `geomean()`, `minimum()`, `maximum()`, `median()`

### String expressions

Every filter and ranking expression has a string form for use at the CLI or in configuration:

| Python | String |
|--------|--------|
| `Affinity <= 500` | `affinity <= 500` / `ba <= 500` |
| `Affinity.rank <= 2` | `affinity.rank <= 2` |
| `Presentation.score >= 0.5` | `el.score >= 0.5` |
| `(Affinity <= 500) \| (Presentation.rank <= 2)` | `affinity <= 500 \| el.rank <= 2` |
| `Affinity["netmhcpan"] <= 500` | `netmhcpan_ba <= 500` |
| `0.5 * Affinity.score + 0.5 * Presentation.score` | `0.5 * affinity.score + 0.5 * presentation.score` |
| `Affinity.logistic(350, 150)` | `affinity.logistic(350, 150)` |
| `mean(Affinity.score, Presentation.score)` | `mean(affinity.score, presentation.score)` |
| `Column("gene_tpm") >= 5` | `gene_tpm >= 5` |
| `Column("gene_tpm").log()` | `gene_tpm.log()` |

### Column references

Any DataFrame column can be used directly in expressions — expression data, peptide properties, custom annotations. Unknown identifiers are automatically treated as column references:

```python
from topiary import Affinity, Column

# As a filter — bare name or Column() both work
filter_by = Column("gene_tpm") >= 5.0

# As a sorting signal
sort_by = 0.5 * Affinity.logistic(350, 150) - 0.1 * Column("cysteine_count")

# Transforms work on columns too
sort_by = Column("gene_tpm").log1p()
```

In CLI strings, bare names work directly:

```bash
--filter-by "ba <= 500 & gene_tpm >= 5"
--sort-by "affinity.logistic(350, 150) + 0.1 * gene_tpm.log1p()"
```

The explicit `column(name)` syntax also works: `gene_tpm >= 5`.

Misspelled column names get a helpful error at evaluation time: `Column 'hydrophobicty' not found. Did you mean: ['hydrophobicity']?`

### Scope prefixes

The `wt.` prefix reads predictions for the wildtype peptide at the same position (variant workflows only). Use it for differential binding:

```python
from topiary import Affinity, wt

sort_by = Affinity.score - wt.Affinity.score  # mutant advantage
```

`shuffled.` and `self.` prefixes work the same way for shuffled-decoy and self-proteome contexts.

### Peptide-level expressions

`Len()` reads the peptide length. `Count("C")` counts amino acid occurrences:

```python
from topiary import Len, Count, wt

sort_by = Count("C") - wt.Count("C")   # gained/lost cysteines vs wildtype
```

## Expression data

Load gene-level, transcript-level, or variant-level quantification to annotate predictions. Expression values become DataFrame columns that you can reference in ranking and filtering.

### How it works

Each `--*-expression` flag takes a spec string: `[name:]file[:id_col[:val_col]]`.

1. The file is auto-detected (Salmon `.sf`, Kallisto `abundance.tsv`, RSEM `.genes.results` / `.isoforms.results`, StringTie `.gtf`, Cufflinks `.fpkm_tracking`) or treated as generic TSV/CSV.
2. The loaded columns are joined onto the prediction DataFrame by `gene_id`, `transcript_id`, or `variant` respectively.
3. Value columns are prefixed with the name (default: `gene`, `transcript`, or `variant`) and lowercased. For example, Salmon's `TPM` column loaded with `--gene-expression` becomes `gene_tpm`.

### Python API

```python
import pandas as pd
from topiary import TopiaryPredictor, Affinity, Column
from topiary.rna import load_expression_from_spec
from mhctools import NetMHCpan
from varcode import load_vcf

# Load expression data
gene_name, gene_id_col, gene_df = load_expression_from_spec(
    "salmon_quant.sf", default_name="gene"
)
# gene_df has columns: ["Name", "TPM"]
# gene_name = "gene", gene_id_col = "Name"

expression_data = {
    "gene": [(gene_name, gene_id_col, gene_df)],
    "transcript": [],
    "variant": [],
}

predictor = TopiaryPredictor(
    models=[NetMHCpan],
    alleles=["HLA-A*02:01"],
    filter_by=(Affinity <= 500) & (Column("gene_tpm") >= 5.0),
)

variants = load_vcf("somatic.vcf")
df = predictor.predict_from_variants(
    variants,
    expression_data=expression_data,
)
# df now has a "gene_tpm" column from the Salmon file
```

### Column naming

The name prefix and the original column name are combined as `{prefix}_{column}`, both lowercased:

| Flag | File | Original column | Result column |
|------|------|----------------|---------------|
| `--gene-expression quant.sf` | Salmon | `TPM` | `gene_tpm` |
| `--gene-expression quant.sf` | Salmon | `NumReads` | `gene_numreads` |
| `--transcript-expression abundance.tsv` | Kallisto | `tpm` | `transcript_tpm` |
| `--variant-expression isovar.tsv` | generic | `num_alt_reads` | `variant_num_alt_reads` |
| `--gene-expression mygene:quant.sf` | Salmon | `TPM` | `mygene_tpm` |

The join keys are implicit: gene-level data joins on `gene_id`, transcript-level on `transcript_id`, variant-level on `variant`. These columns are present in variant-pipeline output.

### Transcript selection

When transcript-level expression data is provided, Topiary uses it to select the highest-expressed transcript per variant (instead of the default priority-based selection). This matches the behavior of the legacy `--rna-transcript-fpkm-tracking-file` flag.

### Auto-detected formats

| Extension / pattern | Format | Default ID column | Default value column |
|---------------------|--------|-------------------|---------------------|
| `.sf` | Salmon | `Name` | `TPM` |
| `abundance*.tsv` (with `target_id` + `tpm` header) | Kallisto | `target_id` | `tpm` |
| `.genes.results` | RSEM (gene) | `gene_id` | `TPM` |
| `.isoforms.results` | RSEM (transcript) | `transcript_id` | `TPM` |
| `.gtf` | StringTie | `reference_id` | `TPM` (or `FPKM`) |
| `.fpkm_tracking` | Cufflinks | `tracking_id` | `FPKM` |
| anything else | generic | first column | all numeric columns |

Override defaults with the full spec: `name:file:id_col:val_col`.

## CLI usage

### Score peptides

```bash
topiary \
  --peptide-csv peptides.csv \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --output-csv results.csv
```

The CSV needs a `peptide` column (and optionally `name`). Each peptide is scored as-is.

### Scan proteins

```bash
topiary \
  --fasta proteins.fasta \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --ic50-cutoff 500 \
  --output-csv results.csv
```

### Filter and rank

```bash
# Filter: keep if affinity OR presentation passes
topiary \
  --fasta proteins.fasta \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --filter-by "ba <= 500 | el.rank <= 2" \
  --output-csv results.csv

# Rank: composite score from two models
topiary \
  --fasta antigens.fasta \
  --mhc-predictor netmhcpan mhcflurry \
  --mhc-alleles HLA-A*02:01 \
  --filter-by "ba <= 500" \
  --sort-by "mean(affinity['netmhcpan'].logistic(350, 150), affinity['mhcflurry'].logistic(350, 150))" \
  --output-csv results.csv
```

### Neoantigen discovery from somatic variants

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

### Add expression data

```bash
# Gene-level expression from Salmon (auto-detected)
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --gene-expression salmon_quant.sf \
  --filter-by "ba <= 500 & gene_tpm >= 5" \
  --only-novel-epitopes \
  --output-csv results.csv

# Transcript-level from Kallisto
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --transcript-expression kallisto/abundance.tsv \
  --filter-by "ba <= 500" \
  --sort-by "affinity.descending_cdf(500, 200) + 0.1 * transcript_tpm.log1p()" \
  --output-csv results.csv

# Variant-level read support from isovar
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --variant-expression isovar_output.tsv \
  --filter-by "ba <= 500 & variant_num_alt_reads >= 3" \
  --output-csv results.csv

# Combine all three levels
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --gene-expression salmon_quant.sf \
  --transcript-expression kallisto/abundance.tsv \
  --variant-expression isovar_output.tsv \
  --filter-by "ba <= 500 & gene_tpm >= 5 & variant_num_alt_reads >= 3" \
  --output-csv results.csv

# Override auto-detection with explicit spec
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --gene-expression mygene:custom_quant.tsv:ensembl_id:tpm_value \
  --output-csv results.csv
```

Expression flags are only valid with the variant pipeline (`--vcf`, `--maf`, `--variant`). They are rejected with direct sequence inputs (`--fasta`, `--peptide-csv`, etc.) because those outputs lack the `gene_id` / `transcript_id` / `variant` columns needed for joining.

### Gene and transcript lookups

Pull protein sequences from Ensembl:

```bash
topiary --gene-names BRAF TP53 EGFR --mhc-predictor netmhcpan --mhc-alleles HLA-A*02:01
topiary --gene-ids ENSG00000157764 --mhc-predictor netmhcpan --mhc-alleles HLA-A*02:01
topiary --transcript-ids ENST00000288602 --mhc-predictor netmhcpan --mhc-alleles HLA-A*02:01
topiary --ensembl-proteome --mhc-predictor netmhcpan --mhc-alleles HLA-A*02:01
topiary --cta --mhc-predictor netmhcpan --mhc-alleles HLA-A*02:01  # cancer-testis antigens
```

### Restrict to protein regions

```bash
topiary \
  --fasta proteins.fasta \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --regions spike:319-541 nucleocapsid:0-50 \
  --output-csv results.csv
```

Format: `NAME:START-END` (0-indexed, half-open).

### Exclusion filtering

Remove peptides found in reference proteomes — for tumor-specific or pathogen-specific peptide selection:

```bash
--exclude-ensembl                    # human Ensembl proteome
--exclude-non-cta                    # non-CTA proteins (requires pirlygenes)
--exclude-tissues heart_muscle lung  # genes expressed in these tissues
--exclude-fasta reference.fasta      # custom reference sequences
--exclude-mode substring             # "substring" (default) or "exact"
```

## MHC prediction models

Specify one or more predictors with `--mhc-predictor` and alleles with `--mhc-alleles`:

```bash
--mhc-predictor netmhcpan mhcflurry --mhc-alleles HLA-A*02:01,HLA-B*07:02
```

All predictors come from [mhctools](https://github.com/openvax/mhctools).

With `mhctools 3.7.0+`, upstream predictor parsing supports multiple
predictors in one CLI invocation, so commands like
`--mhc-predictor netmhcpan42 bigmhc-el` are supported directly. Topiary keeps
its higher-level `--filter-by` / `--sort-by` DSL on top of that lower-level
predictor interface. Topiary's ranking/filtering DSL is also compatible with
the simplified `mhctools 3.7.0+` kind constants API. NetChop and Pepsickle
behavior follows the upstream changes as well: improved NetChop error handling
and Pepsickle's epitope-focused model selection.

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
| `netmhcpan-iedb` / `netmhccons-iedb` / `smm-iedb` | IEDB web API (no local install) |
| `random` | random predictions (for testing) |

Peptide lengths: `--mhc-epitope-lengths 8,9,10,11` (defaults come from the predictor).

## Output

```bash
--output-csv results.csv
--output-html results.html
--output-csv-sep "\t"
--subset-output-columns peptide allele affinity
--rename-output-column value ic50
```

**All predictions:** `source_sequence_name`, `peptide`, `peptide_offset`, `peptide_length`, `allele`, `kind`, `score`, `value`, `affinity`, `percentile_rank`, `prediction_method_name`

**AntigenFragment predictions add** (both `predict_from_antigens` and the variant-based methods that build fragments internally): `fragment_id`, `source_type`, `overlaps_target`, `wt_peptide`, `wt_peptide_length`, plus any fragment-level annotations flattened to columns.

**Variant predictions additionally add:** `variant`, `gene`, `gene_id`, `transcript_id`, `transcript_name`, `effect`, `effect_type`, `contains_mutant_residues`, `mutation_start_in_peptide`, `mutation_end_in_peptide`

**Expression data adds:** columns named `{prefix}_{column}` as described in [Column naming](#column-naming), e.g. `gene_tpm`, `transcript_tpm`, `variant_num_alt_reads`.

## Peptide properties

Compute amino acid properties and use them in ranking:

```python
from topiary import TopiaryPredictor, Affinity, Column
from topiary.properties import add_peptide_properties

df = predictor.predict_from_named_sequences(seqs)
df = add_peptide_properties(df, groups=["manufacturability"])

# Properties become ranking signals
score = Affinity.logistic(350, 150) - 0.1 * Column("cysteine_count")
# CLI: --sort-by "affinity.logistic(350, 150) - 0.1 * cysteine_count"
```

Named groups:
- `"core"` — charge, hydrophobicity, aromaticity, molecular_weight
- `"manufacturability"` — core + cysteine_count, instability_index, max_7mer_hydrophobicity, difficult_nterm/cterm, asp_pro_bonds
- `"immunogenicity"` — core + tcr_charge, tcr_aromaticity, tcr_hydrophobicity (TCR-facing positions for MHC-I)

## Protein sources (Python API)

```python
from topiary.sources import (
    ensembl_proteome,
    sequences_from_gene_names,
    cta_sequences,
    non_cta_sequences,
    tissue_expressed_sequences,
    available_tissues,
)

# All return dict[name -> amino_acid_sequence]
seqs = sequences_from_gene_names(["BRAF", "TP53", "EGFR"])
cta = cta_sequences()                          # cancer-testis antigens
heart = tissue_expressed_sequences(["heart_muscle"])
print(available_tissues())                      # list tissue names
```
