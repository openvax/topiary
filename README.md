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

Tab completion is built in. To activate for bash/zsh/fish:

```bash
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
    filter_by=(Affinity <= 500) | (Presentation.rank <= 2.0),
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

**Alleles** can be specified as a comma-separated list (`--mhc-alleles`) or one per line in a file (`--mhc-alleles-file`).

**Peptide lengths:** `--mhc-epitope-lengths 8,9,10,11` (defaults come from the predictor).

### Supported predictors

All predictors are provided by [mhctools](https://github.com/openvax/mhctools). The `--mhc-predictor` CLI flag accepts:

**MHC-I binding / presentation:**

| CLI flag | Class | Output kinds |
|----------|-------|-------------|
| `netmhcpan` | NetMHCpan (auto-detects version) | affinity + presentation |
| `netmhcpan4` | NetMHCpan4 | affinity + presentation |
| `netmhcpan4-ba` | NetMHCpan4_BA | affinity only |
| `netmhcpan4-el` | NetMHCpan4_EL | presentation only |
| `netmhcpan41` | NetMHCpan41 | affinity + presentation |
| `netmhcpan41-ba` / `netmhcpan41-el` | NetMHCpan41_BA / _EL | single mode |
| `netmhc` | NetMHC (auto-detects 3 vs 4) | affinity |
| `netmhccons` | NetMHCcons | affinity |
| `mhcflurry` | MHCflurry | affinity + presentation + processing |
| `mixmhcpred` | MixMHCpred | presentation |
| `random` | RandomBindingPredictor | affinity (random, for testing) |

**MHC-II binding:**

| CLI flag | Class | Output kinds |
|----------|-------|-------------|
| `netmhciipan` | NetMHCIIpan (auto-detects version) | affinity |
| `netmhciipan4` | NetMHCIIpan4 | affinity + presentation |
| `netmhciipan4-ba` / `netmhciipan4-el` | NetMHCIIpan4_BA / _EL | single mode |

**IEDB web API** (no local install needed):

| CLI flag | Class |
|----------|-------|
| `netmhcpan-iedb` | IedbNetMHCpan |
| `netmhccons-iedb` | IedbNetMHCcons |
| `netmhciipan-iedb` | IedbNetMHCIIpan |
| `smm-iedb` | IedbSMM |
| `smm-pmbec-iedb` | IedbSMM_PMBEC |

**Python-only** (no CLI flag — use via the Python API with `models=[ClassName]`):

| Class | What it does |
|-------|-------------|
| `BigMHC` | Presentation and immunogenicity prediction |
| `NetMHCpan42` / `NetMHCpan42_BA` / `NetMHCpan42_EL` | NetMHCpan 4.2 |
| `NetMHCIIpan43` / `NetMHCIIpan43_BA` / `NetMHCIIpan43_EL` | NetMHCIIpan 4.3 |
| `NetMHCstabpan` | pMHC stability prediction |
| `Pepsickle` | Proteasomal cleavage prediction |
| `NetChop` | Proteasome cleavage prediction |

## Expression DSL

Topiary has an expression language for filtering and ranking predictions. It works in two forms: a **Python API** with operator overloading, and a **string syntax** for the CLI. Both compile to the same internal representation.

### Prediction kinds and fields

Four built-in accessors correspond to different aspects of MHC presentation:

| Accessor | Aliases | What it measures |
|----------|---------|------------------|
| `Affinity` | `ba`, `aff`, `ic50` | Binding affinity (IC50 nM) |
| `Presentation` | `el` | Presentation / eluted ligand score |
| `Stability` | | pMHC complex stability |
| `Processing` | | Antigen processing / cleavage |

Each has three fields:

| Field | Description | Example |
|-------|-------------|---------|
| `.value` | Raw value (IC50 nM, etc.) | `Affinity.value` |
| `.rank` | Percentile rank (lower = better) | `Affinity.rank` |
| `.score` | Normalized score (higher = better) | `Affinity.score` |

The default field is `.value`, so `Affinity <= 500` means `Affinity.value <= 500`.

### Filters: Python vs string

Filters select which peptide-allele groups to keep. The same filter can be written in Python or as a CLI string:

```python
# Python                                    # CLI string
Affinity <= 500                             # "affinity <= 500" or "ba <= 500"
Affinity.rank <= 2.0                        # "affinity.rank <= 2"
Presentation.score >= 0.5                   # "el.score >= 0.5"

# OR / AND
(Affinity <= 500) | (Presentation.rank <= 2)  # "affinity <= 500 | el.rank <= 2"
(Affinity <= 500) & (Presentation.rank <= 2)  # "affinity <= 500 & el.rank <= 2"
```

On the CLI:

```bash
--ranking "affinity <= 500 | el.rank <= 2"
--ic50-cutoff 500          # shorthand for affinity <= 500
--percentile-cutoff 2.0    # shorthand for affinity.rank <= 2
--filter-logic any         # "any" (OR, default) or "all" (AND)
```

### Multi-model disambiguation

When multiple models produce the same kind (e.g. NetMHCpan and MHCflurry both produce affinity), qualify with brackets in Python or underscores in strings:

```python
# Python                                    # CLI string
Affinity["netmhcpan"] <= 500               # "netmhcpan_affinity <= 500"
Affinity["mhcflurry"].score                # "mhcflurry_affinity.score"
Presentation["mhcflurry"].rank <= 2        # "mhcflurry_el.rank <= 2"
```

When only one model produces a kind, no qualification is needed. If you forget to qualify with multiple models, you get a clear error:

```
ValueError: Ambiguous: multiple models produce pMHC_affinity
(mhcflurry, netmhcpan). Use Affinity["modelname"] to disambiguate.
```

Typos also get caught:

```
ValueError: No pMHC_affinity predictions from method matching 'netmhcapn'.
Available: ['mhcflurry', 'netmhcpan']. Did you mean: ['netmhcpan']?
```

### Transforms (Python-only)

Expressions support mathematical transforms for composite scoring. These have no CLI string equivalent — use the Python API for composite scores:

```python
# Normalizing to [0, 1]:
Affinity.descending_cdf(mean=500, std=200)           # lower input → higher output (for IC50, rank)
Presentation.score.ascending_cdf(mean=0.5, std=0.3)  # higher input → higher output (for scores)
Affinity.logistic(midpoint=350, width=150)       # lower input → higher output (sigmoid)

# Aggregating across models:
from topiary import mean, geomean, minimum, maximum, median
mean(Affinity["netmhcpan"].logistic(350, 150),
     Affinity["mhcflurry"].logistic(350, 150))

# Arithmetic:
0.5 * Affinity.score + 0.5 * Presentation.score

# Other transforms:
Affinity.value.clip(lo=1, hi=50000)
Affinity.value.hinge()               # max(0, x)
Affinity.value.log()                 # also log2(), log10(), log1p()
Affinity.value.sqrt()
abs(Affinity.value)
```

### Column() — use any DataFrame column

`Column()` brings arbitrary DataFrame columns into the expression system. This is how peptide properties, read counts, and custom annotations participate in ranking:

```python
# Python                                    # CLI string (filters only)
Column("cysteine_count")                    # column(cysteine_count)
Column("cysteine_count") <= 2               # "column(cysteine_count) <= 2"
Column("hydrophobicity") >= -0.5            # "column(hydrophobicity) >= -0.5"

# In composite scores (Python-only)
score = (
    0.5 * Affinity.logistic(350, 150)
    - 0.2 * Column("cysteine_count")
    + 0.1 * Column("tcr_aromaticity")
)
```

Missing columns get a helpful error with typo suggestions:

```
ValueError: Column 'hydrophobicty' not found. Did you mean: ['hydrophobicity']?
```

### wt. — wildtype comparison

The `wt.` scope prefix reads wildtype prediction columns (`wt_value`, `wt_score`, `wt_percentile_rank`), populated by `predict_column` after variant-derived predictions:

```python
# Python API (capitalized kind names)
wt.Affinity.value                         # WT IC50
wt.Affinity["netmhcpan"].score            # qualified WT
Affinity.score - wt.Affinity.score        # differential binding
Affinity.logistic(350, 150) - wt.Affinity.logistic(350, 150)

# String DSL (lowercase kind names)
# wt.affinity.value
# wt.affinity["netmhcpan"].score
# affinity.score - wt.affinity.score
```

`wt.` expressions are for ranking, not filters. Returns NaN when WT columns don't exist (non-variant inputs).

### len and count() — peptide-level expressions

`len` reads the peptide length; `count('C')` counts amino acid occurrences. Both work with scope prefixes:

```python
# String DSL
# len                        # peptide length
# count('C')                 # cysteine count
# wt.len                     # wildtype peptide length
# wt.count('C')              # wildtype cysteine count
# count('C') - wt.count('C') # gained/lost cysteines
```

### Sorting

Sort surviving peptides after filtering:

```python
predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["HLA-A*02:01"],
    filter_by=(Affinity <= 500) | (Presentation.rank <= 2.0),
    rank_by=[Presentation.score, Affinity.score],  # first non-NaN wins
)
```

On the CLI:

```bash
--rank-by pMHC_presentation,pMHC_affinity
--rank-by "netmhcpan_affinity,mhcflurry_presentation"  # tool-qualified
```

### Quick reference: Python to CLI string

| Python DSL | CLI string form |
|---|---|
| `Affinity <= 500` | `affinity <= 500` or `ba <= 500` or `ic50 <= 500` |
| `Affinity.rank <= 2` | `affinity.rank <= 2` |
| `Affinity.score >= 0.5` | `affinity.score >= 0.5` |
| `Affinity["netmhcpan"] <= 500` | `netmhcpan_ba <= 500` |
| `Presentation["mhcflurry"].rank <= 2` | `mhcflurry_el.rank <= 2` |
| `Column("cysteine_count") <= 2` | `column(cysteine_count) <= 2` |
| `(A <= 500) \| (B.rank <= 2)` | `affinity <= 500 \| el.rank <= 2` |
| `0.5 * Affinity.score + ...` | *Python-only* |
| `.logistic()`, `.ascending_cdf()`, `.descending_cdf()`, `.clip()` | *Python-only* |
| `mean()`, `geomean()`, `minimum()`, `maximum()`, `median()` | *Python-only* |
| `wt.Affinity.score` | `wt.affinity.score` |
| `len`, `count('C')` | `len`, `count('C')` |
| `wt.len`, `wt.count('C')` | `wt.len`, `wt.count('C')` |
| `Column("x")` in arithmetic | *Python-only* |

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

## Advanced example: neoantigen scoring pipeline

Full Python API for neoantigen analysis — variant prediction, wildtype comparison, reference proteome check, peptide properties, and composite scoring across two models:

```python
from topiary import (
    TopiaryPredictor, Affinity, Presentation, Column, wt,
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
    0.25 * Affinity["netmhcpan"].logistic(350, 150)
    + 0.25 * Affinity["mhcflurry"].logistic(350, 150)

    # Presentation
    + 0.2 * Presentation["mhcflurry"].score

    # Differential binding: mutant binds better than wildtype
    + 0.15 * (Affinity["netmhcpan"].logistic(350, 150)
              - wt.Affinity["netmhcpan"].logistic(350, 150))

    # Manufacturability: penalize cysteines and unstable peptides
    - 0.05 * Column("cysteine_count")
    - 0.05 * Column("instability_index").clip(lo=0, hi=100).ascending_cdf(50, 20)

    # Immunogenicity: reward aromatic TCR-facing residues
    + 0.05 * Column("tcr_aromaticity")
)
```

The CLI equivalent for the filtering portion:

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
