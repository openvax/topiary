[![Tests](https://github.com/openvax/topiary/actions/workflows/tests.yml/badge.svg)](https://github.com/openvax/topiary/actions/workflows/tests.yml)
<a href="https://coveralls.io/github/openvax/topiary?branch=master">
    <img src="https://coveralls.io/repos/openvax/topiary/badge.svg?branch=master&service=github" alt="Coverage Status" />
</a>
<a href="https://pypi.python.org/pypi/topiary/">
    <img src="https://img.shields.io/pypi/v/topiary.svg?maxAge=1000" alt="PyPI" />
</a>

# Topiary

Topiary predicts which peptides from protein sequences will be presented by MHC molecules, making them potential T-cell epitopes. It is used in cancer immunotherapy research to find mutant peptides (neoantigens) that the immune system could target.

**Core idea:** Given protein sequences + HLA alleles + one or more MHC prediction models, Topiary scans all possible peptides and returns those predicted to be presented by MHC, ranked by any combination of binding affinity, presentation score, processing score, and stability.

Topiary can start from several types of input:

- **Somatic variants** (VCF/MAF) — the original use case: find mutant peptides from cancer sequencing data
- **Protein sequences** (FASTA/CSV) — scan full-length proteins with a sliding window
- **Peptide lists** (FASTA/CSV) — score specific peptides directly, no sliding window
- **Gene/transcript IDs** — pull sequences from Ensembl automatically
- **Built-in gene sets** — cancer-testis antigens (CTA), tissue-expressed genes

## How it works

1. **Get protein sequences** — from variants (via [varcode](https://github.com/openvax/varcode)), FASTA/CSV files, or Ensembl lookups
2. **Generate candidate peptides** — sliding window over proteins, or use peptides as-is
3. **Predict MHC presentation** — via [mhctools](https://github.com/openvax/mhctools) (NetMHCpan, MHCflurry, etc.), producing binding affinity, presentation, processing, and/or stability scores depending on the model
4. **Filter and rank** — by binding affinity, percentile rank, presentation score, or custom expressions
5. **Annotate** — with gene/transcript info, mutation positions, RNA expression levels

For variant inputs, Topiary also filters by RNA expression and identifies which predicted epitopes actually overlap the mutation.

## Installation

```bash
pip install topiary
```

For Ensembl-based features (variant annotation, gene lookups), download reference data:

```bash
# GRCh38 (hg38) — most common
pyensembl install --release 93 --species human

# GRCh37 (hg19) — if your variants use this reference
pyensembl install --release 75 --species human
```

For cancer-testis antigen and tissue expression features:

```bash
pip install pirlygenes
```

For tab completion of command-line arguments (bash/zsh/fish):

```bash
pip install 'topiary[completion]'
activate-global-python-argcomplete
```

## Quick start

### Command line

**Scan a FASTA file for MHC binders:**

```bash
topiary \
  --fasta proteins.fasta \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --ic50-cutoff 500 \
  --output-csv results.csv
```

**Score specific peptides (no sliding window):**

```bash
topiary \
  --peptide-csv peptides.csv \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --ic50-cutoff 500 \
  --output-csv results.csv
```

**Find neoantigen candidates from somatic variants:**

```bash
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --ic50-cutoff 500 \
  --percentile-cutoff 2.0 \
  --rna-gene-fpkm-tracking-file genes.fpkm_tracking \
  --rna-min-gene-expression 4.0 \
  --only-novel-epitopes \
  --output-csv epitopes.csv
```

**Scan cancer-testis antigens, excluding peptides found in vital organs:**

```bash
topiary \
  --cta \
  --exclude-tissues heart_muscle lung liver \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --ic50-cutoff 500 \
  --output-csv cta_epitopes.csv
```

### Python API

```python
from topiary import TopiaryPredictor, Affinity, Presentation
from mhctools import NetMHCpan

# Set up predictor with filtering
predictor = TopiaryPredictor(
    models=[NetMHCpan],
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
    filter=(Affinity <= 500) | (Presentation.rank <= 2.0),
    rank_by=[Presentation.score, Affinity.score],
)

# Scan protein sequences (sliding window)
df = predictor.predict_from_named_sequences({
    "BRAF_V600E": "MAALSGGGGG...LATEKSRWSG",
    "TP53_R248W": "MEEPQSDPSV...ALPQHAHAQM",
})

# Score specific peptides (no sliding window)
df = predictor.predict_from_named_peptides({
    "peptide_1": "YLQLVFGIEV",
    "peptide_2": "LLFNILGGWV",
})

# From somatic variants (requires varcode)
from varcode import load_vcf
variants = load_vcf("somatic.vcf")
df = predictor.predict_from_variants(variants)
```

## Input modes

### Sequence and peptide files

| Flag | Format | Behavior |
|------|--------|----------|
| `--fasta FILE` | FASTA with full-length proteins | Sliding-window scan |
| `--peptide-fasta FILE` | FASTA where each entry is one peptide | Scored as-is |
| `--sequence-csv FILE` | CSV with `sequence` column (+ optional `name`) | Sliding-window scan |
| `--peptide-csv FILE` | CSV with `peptide` column (+ optional `name`) | Scored as-is |

### Gene and transcript lookups

These pull protein sequences from Ensembl automatically:

| Flag | Example |
|------|---------|
| `--gene-names NAME [NAME ...]` | `--gene-names BRAF TP53 EGFR` |
| `--gene-ids ID [ID ...]` | `--gene-ids ENSG00000157764` |
| `--transcript-ids ID [ID ...]` | `--transcript-ids ENST00000288602` |
| `--ensembl-proteome` | Scan the entire Ensembl proteome |
| `--cta` | Cancer-testis antigen genes (requires `pirlygenes`) |
| `--ensembl-release N` | Use a specific Ensembl release (default: 93 for human) |

For gene lookups, Topiary uses the longest protein-coding transcript per gene.

### Genomic variants

| Flag | Description |
|------|-------------|
| `--vcf FILE` | VCF file of somatic variants |
| `--maf FILE` | TCGA MAF file |
| `--variant CHR POS REF ALT` | Individual variant (requires `--ensembl-version`) |
| `--protein-change GENE CHANGE` | Direct protein change, e.g. `--protein-change EGFR T790M` |

Multiple input flags can be combined in a single run.

## MHC prediction models

Models predict one or more aspects of MHC presentation — binding affinity, antigen processing, stability, or an overall presentation score. Different models produce different subsets of these. You must specify a predictor and alleles:

```bash
--mhc-predictor netmhcpan \
--mhc-alleles HLA-A*02:01,HLA-B*07:02
```

**Supported predictors:** `netmhcpan`, `netmhc`, `netmhciipan`, `netmhccons`, `mhcflurry`, `random`, and IEDB web API variants (`netmhcpan-iedb`, `netmhccons-iedb`, `smm-iedb`, `smm-pmbec-iedb`).

**Alleles** can be specified as a comma-separated list (`--mhc-alleles`) or one per line in a file (`--mhc-alleles-file`).

**Peptide lengths:** `--mhc-epitope-lengths 8,9,10,11` (defaults come from the predictor).

## Filtering and ranking

### Simple cutoffs

```bash
--ic50-cutoff 500            # Keep peptides with IC50 <= 500 nM
--percentile-cutoff 2.0      # Keep peptides with percentile rank <= 2.0
--presentation-cutoff 2.0    # Keep peptides with presentation rank <= 2.0
--filter-logic any            # "any" (OR, default) or "all" (AND)
```

### Expression-based ranking

```bash
--rank-by pMHC_presentation,pMHC_affinity
```

Sort surviving peptides by presentation score, breaking ties with affinity.

### Advanced filter expressions

```bash
--ranking "affinity <= 500 | presentation.rank <= 2"
```

### Python API expressions

```python
from topiary import Affinity, Presentation, RankingStrategy

# Combine filters with | (OR) or & (AND)
my_filter = (Affinity <= 500) | (Presentation.rank <= 2.0)

# Composite scoring
my_score = 0.5 * Affinity.score + 0.5 * Presentation.score

predictor = TopiaryPredictor(
    models=[NetMHCpan],
    alleles=["HLA-A*02:01"],
    filter=my_filter,
    rank_by=[my_score],
)
```

Available prediction kinds: `Affinity`, `Presentation`, `Processing`, `Stability`. Each has `.value`, `.rank`, and `.score` attributes.

## Exclusion filtering

For direct sequence/peptide inputs, you can exclude peptides that also appear in reference proteomes — useful for finding tumor-specific or pathogen-specific peptides:

```bash
--exclude-ensembl                    # Exclude peptides in the human Ensembl proteome
--exclude-non-cta                    # Exclude non-CTA proteins (requires pirlygenes)
--exclude-tissues heart_muscle lung  # Exclude genes expressed in these tissues
--exclude-fasta reference.fasta      # Exclude peptides in custom reference sequences
--exclude-mode substring             # "substring" (default) or "exact"
```

## Region restriction

Limit prediction to specific protein regions (only applies to sequence inputs, not peptides):

```bash
--regions spike:319-541 nucleocapsid:0-50
```

Format: `name:start-end` (0-based, half-open interval).

## RNA expression filtering

For variant-based workflows, filter by gene or transcript expression:

```bash
--rna-gene-fpkm-tracking-file genes.fpkm_tracking
--rna-min-gene-expression 4.0
--rna-transcript-fpkm-tracking-file isoforms.fpkm_tracking
--rna-min-transcript-expression 1.5
```

Also supports StringTie GTF format: `--rna-transcript-fpkm-gtf-file`.

## Output

```bash
--output-csv results.csv          # CSV output
--output-html results.html        # HTML table
--output-csv-sep "\t"             # Use tab separator
--subset-output-columns peptide allele affinity  # Select columns
--rename-output-column value ic50               # Rename columns
```

### Output columns

**All predictions:** `source_sequence_name`, `peptide`, `peptide_offset`, `peptide_length`, `allele`, `kind`, `score`, `value`, `affinity`, `percentile_rank`, `prediction_method_name`

**Variant predictions add:** `variant`, `gene`, `gene_id`, `transcript_id`, `transcript_name`, `effect`, `effect_type`, `contains_mutant_residues`, `mutation_start_in_peptide`, `mutation_end_in_peptide`

## Built-in protein sources (Python API)

The `topiary.sources` module provides functions for loading protein sequences from Ensembl and PirlyGenes:

```python
from topiary.sources import (
    ensembl_proteome,
    sequences_from_gene_names,
    sequences_from_gene_ids,
    sequences_from_transcript_ids,
    cta_sequences,
    non_cta_sequences,
    tissue_expressed_sequences,
    available_tissues,
)

# All return dict[name -> amino_acid_sequence]
seqs = sequences_from_gene_names(["BRAF", "TP53", "EGFR"])
cta = cta_sequences()
tissues = available_tissues()  # list of tissue names
```

## Peptide properties

Compute amino acid properties and use them in ranking:

```python
from topiary.properties import add_peptide_properties

df = predictor.predict_from_named_sequences(seqs)
df = add_peptide_properties(df, groups=["manufacturability"])
```

Named groups:
- `"core"` — charge, hydrophobicity, aromaticity, molecular_weight
- `"manufacturability"` — core + cysteine_count, instability_index, max_7mer_hydrophobicity, difficult_nterm/cterm, asp_pro_bonds
- `"immunogenicity"` — core + tcr_charge, tcr_aromaticity, tcr_hydrophobicity (TCR-facing positions for MHC-I)

Properties become ranking signals via `Column()`:

```python
from topiary.ranking import Column

score = Affinity.logistic(350, 150) - 0.1 * Column("cysteine_count")
```

## Multi-model and composite scoring

When using multiple MHC prediction models, qualify by method name using bracket syntax to disambiguate:

```python
from topiary import TopiaryPredictor, Affinity, Presentation
from mhctools import NetMHCpan, MHCflurry

predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
)
df = predictor.predict_from_named_sequences(seqs)

# Each model's predictions are accessible separately
Affinity["netmhcpan"].value          # string: "netmhcpan_affinity" or "netmhcpan_ba"
Affinity["mhcflurry"].value          # string: "mhcflurry_affinity" or "mhcflurry_ba"
Presentation["mhcflurry"].score      # string: "mhcflurry_presentation.score" or "mhcflurry_el.score"

# Combine into a composite score (Python-only — no string form for arithmetic)
score = (
    0.3 * Affinity["netmhcpan"].logistic(350, 150)
    + 0.3 * Affinity["mhcflurry"].logistic(350, 150)
    + 0.4 * Presentation["mhcflurry"].score
)
```

On the CLI, use `tool_kind` underscore syntax:

```bash
--ranking "netmhcpan_affinity <= 500 | mhcflurry_el.rank <= 2"
--rank-by "netmhcpan_affinity,mhcflurry_presentation"
```

When only one model produces a given kind, no qualification is needed — `Affinity <= 500` works automatically.

## Advanced example: neoantigen scoring pipeline

This example shows the full Python API for a neoantigen analysis — predicting mutant epitopes, comparing against wildtype, checking the reference proteome, computing peptide properties, and ranking with a composite score across multiple models:

```python
from topiary import (
    TopiaryPredictor, Affinity, Presentation, Column, WT,
)
from topiary.properties import add_peptide_properties
from topiary.comparison import predict_column, annotate_reference
from topiary.sources import ensembl_proteome, non_cta_sequences
from mhctools import NetMHCpan, MHCflurry
from varcode import load_vcf

# --- Set up predictor with two models ---

predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
    only_novel_epitopes=True,
)

# --- Predict from somatic variants ---

variants = load_vcf("somatic.vcf")
df = predictor.predict_from_variants(variants)
# df now has: peptide, allele, kind, value, score, percentile_rank,
#   variant, gene, effect, contains_mutant_residues,
#   wt_peptide, wt_source_gene_name, wt_source_sequence_id, ...

# --- Add wildtype predictions (same models, WT peptide at same position) ---

df = predict_column(df, predictor, "wt_peptide", "wt_")
# Adds: wt_value, wt_score, wt_percentile_rank (per kind × method)

# --- Check if peptide appears in normal proteome ---

df = annotate_reference(df, ensembl_proteome())
# Adds: occurs_in_reference (boolean)

# --- Compute peptide properties ---

df = add_peptide_properties(df, groups=["manufacturability", "immunogenicity"])
# Adds: charge, hydrophobicity, aromaticity, cysteine_count,
#   instability_index, tcr_charge, tcr_aromaticity, ...

df = add_peptide_properties(df, peptide_column="wt_peptide", prefix="wt_",
                            groups=["core"])
# Adds: wt_charge, wt_hydrophobicity, wt_aromaticity, wt_molecular_weight

# --- Rank with a composite score across both models ---

score = (
    # Binding: average logistic IC50 across models
    #   string filter form: "netmhcpan_ba <= 500" or "netmhcpan_affinity <= 500"
    #   (logistic transform and arithmetic are Python-only)
    0.25 * Affinity["netmhcpan"].logistic(350, 150)
    + 0.25 * Affinity["mhcflurry"].logistic(350, 150)

    # Presentation: MHCflurry presentation score
    #   string filter form: "mhcflurry_el.score >= 0.5"
    + 0.2 * Presentation["mhcflurry"].score

    # Differential binding: mutant binds better than wildtype
    #   no string form — WT() is Python-only
    + 0.15 * (Affinity["netmhcpan"].logistic(350, 150)
              - WT(Affinity["netmhcpan"]).logistic(350, 150))

    # Manufacturability: penalize cysteines and unstable peptides
    #   string filter form: "column(cysteine_count) <= 2"
    #   (arithmetic and transforms like .clip().norm() are Python-only)
    - 0.05 * Column("cysteine_count")
    - 0.05 * Column("instability_index").clip(lo=0, hi=100).norm(50, 20)

    # Immunogenicity: reward aromatic TCR-facing residues
    #   string filter form: "column(tcr_aromaticity) >= 1"
    + 0.05 * Column("tcr_aromaticity")
)
```

### String form reference

The CLI `--ranking` flag supports filter expressions. Here's the mapping:

| Python DSL | String form | Notes |
|---|---|---|
| `Affinity <= 500` | `affinity <= 500` or `ba <= 500` | |
| `Affinity.rank <= 2` | `affinity.rank <= 2` or `ba.rank <= 2` | |
| `Affinity.score >= 0.5` | `affinity.score >= 0.5` | |
| `Affinity["netmhcpan"] <= 500` | `netmhcpan_affinity <= 500` or `netmhcpan_ba <= 500` | |
| `Presentation["mhcflurry"].rank <= 2` | `mhcflurry_presentation.rank <= 2` or `mhcflurry_el.rank <= 2` | |
| `Column("cysteine_count") <= 2` | `column(cysteine_count) <= 2` | |
| `(A <= 500) \| (B.rank <= 2)` | `affinity <= 500 \| presentation.rank <= 2` | |
| `(A <= 500) & (B.rank <= 2)` | `affinity <= 500 & presentation.rank <= 2` | |

**Python-only** (no string form):
- Arithmetic: `0.5 * Affinity.score + 0.5 * Presentation.score`
- Transforms: `.logistic()`, `.norm()`, `.clip()`, `.log()`, `.sqrt()`
- `WT()` expressions: `WT(Affinity).score`, `Affinity.score - WT(Affinity).score`
- `Column()` in arithmetic: `0.5 * Column("charge")` (only `column(x) <= N` works in strings)

For full composite scoring, use the Python API. The CLI handles filtering and simple sort-by:

```bash
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --ranking "netmhcpan_ba <= 500 & column(cysteine_count) <= 2" \
  --rank-by "netmhcpan_affinity,mhcflurry_presentation" \
  --only-novel-epitopes \
  --output-csv epitopes.csv
```
