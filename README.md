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

## Installation

```bash
pip install topiary
```

For Ensembl-based features (variant annotation, gene lookups), download reference data:

```bash
pyensembl install --release 93 --species human
```

Tab completion: `activate-global-python-argcomplete`

## Quick start

### Scan a FASTA for MHC binders

```bash
topiary \
  --fasta proteins.fasta \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --ic50-cutoff 500 \
  --output-csv results.csv
```

### Score specific peptides

```bash
topiary \
  --peptide-csv peptides.csv \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --ranking "ba <= 500 & el.score >= 0.5" \
  --output-csv results.csv
```

### Find neoantigen candidates from somatic variants

```bash
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --ranking "ba <= 500 | el.rank <= 2" \
  --rank-by "0.6 * affinity.descending_cdf(500, 200) + 0.4 * presentation.score" \
  --only-novel-epitopes \
  --output-csv epitopes.csv
```

### Multi-model scoring with composite ranking

```bash
topiary \
  --fasta antigens.fasta \
  --mhc-predictor netmhcpan mhcflurry \
  --mhc-alleles HLA-A*02:01 \
  --ranking "ba <= 500" \
  --rank-by "mean(affinity['netmhcpan'].logistic(350, 150), affinity['mhcflurry'].logistic(350, 150))" \
  --output-csv results.csv
```

### Python API

```python
from topiary import TopiaryPredictor, Affinity, Presentation, wt, mean
from mhctools import NetMHCpan, MHCflurry

predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
    filter_by=(Affinity <= 500) | (Presentation.rank <= 2.0),
    rank_by=mean(
        Affinity["netmhcpan"].logistic(350, 150),
        Affinity["mhcflurry"].logistic(350, 150),
    ),
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

## Inputs

### Sequences and peptides

| Flag | Format | Behavior |
|------|--------|----------|
| `--fasta FILE` | FASTA with full-length proteins | Sliding-window scan |
| `--peptide-fasta FILE` | FASTA where each entry is one peptide | Scored as-is |
| `--sequence-csv FILE` | CSV with `sequence` column (+ optional `name`) | Sliding-window scan |
| `--peptide-csv FILE` | CSV with `peptide` column (+ optional `name`) | Scored as-is |

### Genomic variants

| Flag | Description |
|------|-------------|
| `--vcf FILE` | VCF file of somatic variants |
| `--maf FILE` | TCGA MAF file |
| `--variant CHR POS REF ALT` | Individual variant |
| `--protein-change GENE CHANGE` | Direct protein change, e.g. `--protein-change EGFR T790M` |

### Gene and transcript lookups

Pull protein sequences from Ensembl automatically:

| Flag | Example |
|------|---------|
| `--gene-names NAME [...]` | `--gene-names BRAF TP53 EGFR` |
| `--gene-ids ID [...]` | `--gene-ids ENSG00000157764` |
| `--transcript-ids ID [...]` | `--transcript-ids ENST00000288602` |
| `--ensembl-proteome` | Scan the entire Ensembl proteome |
| `--cta` | Cancer-testis antigen genes (requires `pirlygenes`) |
| `--ensembl-release N` | Use a specific Ensembl release |

Multiple input flags can be combined in a single run.

### Restrict to regions

Limit prediction to specific protein regions (only applies to sequence inputs):

```bash
--regions spike:319-541 nucleocapsid:0-50
```

### RNA expression filtering

For variant workflows, filter by gene or transcript expression:

```bash
--rna-gene-fpkm-tracking-file genes.fpkm_tracking
--rna-min-gene-expression 4.0
--rna-transcript-fpkm-tracking-file isoforms.fpkm_tracking
--rna-min-transcript-expression 1.5
```

Also supports StringTie GTF: `--rna-transcript-fpkm-gtf-file`.

## MHC prediction models

Specify one or more predictors and alleles:

```bash
--mhc-predictor netmhcpan mhcflurry \
--mhc-alleles HLA-A*02:01,HLA-B*07:02
```

All predictors come from [mhctools](https://github.com/openvax/mhctools). Multiple models can be used together — the expression DSL handles disambiguation.

| CLI name | Class | Predicts |
|----------|-------|----------|
| `netmhcpan` | NetMHCpan (auto-detects version) | affinity + presentation |
| `netmhcpan4` / `netmhcpan41` | NetMHCpan4 / 41 | affinity + presentation |
| `netmhcpan4-ba` / `netmhcpan4-el` | NetMHCpan4_BA / _EL | single mode |
| `netmhcpan42` / `netmhcpan42-ba` / `netmhcpan42-el` | NetMHCpan42 variants | NetMHCpan 4.2 |
| `mhcflurry` | MHCflurry | affinity + presentation + processing |
| `mixmhcpred` | MixMHCpred | presentation |
| `netmhciipan` / `netmhciipan4` | NetMHCIIpan variants | MHC-II affinity + presentation |
| `netmhciipan43` / `netmhciipan43-ba` / `netmhciipan43-el` | NetMHCIIpan43 variants | NetMHCIIpan 4.3 |
| `bigmhc` / `bigmhc-el` / `bigmhc-im` | BigMHC variants | presentation / immunogenicity |
| `netmhcstabpan` | NetMHCstabpan | pMHC stability |
| `pepsickle` / `netchop` | Pepsickle / NetChop | proteasomal cleavage |
| `netmhcpan-iedb` / `netmhccons-iedb` / `smm-iedb` | IEDB web API | no local install needed |
| `random` | RandomBindingPredictor | random (for testing) |

**Peptide lengths:** `--mhc-epitope-lengths 8,9,10,11` (defaults come from the predictor).

## Filtering and ranking

Topiary has an expression language for filtering and ranking predictions. It works identically in the Python API and as CLI strings.

### Prediction kinds and fields

| Accessor | Aliases | What it measures |
|----------|---------|------------------|
| `Affinity` | `ba`, `aff`, `ic50` | Binding affinity (IC50 nM) |
| `Presentation` | `el` | Presentation / eluted ligand score |
| `Stability` | | pMHC complex stability |
| `Processing` | | Antigen processing / cleavage |

Each has three fields: `.value` (raw), `.rank` (percentile, lower = better), `.score` (normalized, higher = better). Default is `.value`, so `Affinity <= 500` means `Affinity.value <= 500`.

### Filters

```python
# Python                                    # CLI string
Affinity <= 500                             # "affinity <= 500" or "ba <= 500"
Affinity.rank <= 2.0                        # "affinity.rank <= 2"
Presentation.score >= 0.5                   # "el.score >= 0.5"
(Affinity <= 500) | (Presentation.rank <= 2)  # "affinity <= 500 | el.rank <= 2"
(Affinity <= 500) & (Presentation.rank <= 2)  # "affinity <= 500 & el.rank <= 2"
```

CLI flags:

```bash
--ranking "affinity <= 500 | el.rank <= 2"
--ic50-cutoff 500          # shorthand for affinity <= 500
--percentile-cutoff 2.0    # shorthand for affinity.rank <= 2
```

### Ranking and transforms

Sort surviving peptides with `--rank-by`. Supports arithmetic, transforms, and aggregations:

```bash
# Simple: sort by presentation score, fall back to affinity
--rank-by pMHC_presentation,pMHC_affinity

# Composite score with normalization
--rank-by "0.6 * affinity.descending_cdf(500, 200) + 0.4 * presentation.score"

# Average across models
--rank-by "mean(affinity['netmhcpan'].logistic(350, 150), affinity['mhcflurry'].logistic(350, 150))"

# Chain transforms
--rank-by "affinity.value.clip(1, 50000).log()"
```

Available transforms:

| Transform | What it does |
|-----------|-------------|
| `.descending_cdf(mean, std)` | Lower input → higher output (for IC50, rank) |
| `.ascending_cdf(mean, std)` | Higher input → higher output (for scores) |
| `.logistic(midpoint, width)` | Sigmoid normalization |
| `.clip(lo, hi)` | Clamp to range |
| `.hinge()` | `max(0, x)` |
| `.log()` / `.log2()` / `.log10()` / `.log1p()` | Logarithms |
| `.sqrt()` / `.exp()` | Square root, exponential |
| `abs(...)` | Absolute value |

Aggregations: `mean()`, `geomean()`, `minimum()`, `maximum()`, `median()`

### Multi-model disambiguation

When multiple models produce the same kind, qualify with brackets (Python) or underscores (CLI):

```python
Affinity["netmhcpan"] <= 500               # "netmhcpan_ba <= 500"
Affinity["mhcflurry"].score                # "mhcflurry_affinity.score"
```

### Scope prefixes: wildtype and alternate peptide contexts

The `wt.` prefix reads predictions for the wildtype peptide at the same position. Use it for differential binding:

```python
# Python                                    # CLI string
wt.Affinity.score                           # wt.affinity.score
Affinity.score - wt.Affinity.score          # affinity.score - wt.affinity.score
```

`shuffled.` and `self.` prefixes work the same way for shuffled decoy and self-proteome contexts. All return NaN when the corresponding columns don't exist.

### Peptide-level expressions

`len` reads the peptide length. `count('X')` counts amino acid occurrences. Both work with scope prefixes:

```bash
--rank-by "len"                             # peptide length
--rank-by "count('C')"                      # cysteine count
--rank-by "count('C') - wt.count('C')"      # gained/lost cysteines vs wildtype
```

### Column references

`column(name)` brings any DataFrame column into expressions — peptide properties, read counts, custom annotations:

```bash
--ranking "column(cysteine_count) <= 2"
--rank-by "0.5 * affinity.logistic(350, 150) - 0.2 * column(cysteine_count)"
```

Missing columns get a helpful error: `Column 'hydrophobicty' not found. Did you mean: ['hydrophobicity']?`

### Quick reference

| Python DSL | CLI string |
|---|---|
| `Affinity <= 500` | `affinity <= 500` / `ba <= 500` / `ic50 <= 500` |
| `Affinity.rank <= 2` | `affinity.rank <= 2` |
| `Affinity["netmhcpan"] <= 500` | `netmhcpan_ba <= 500` |
| `(A <= 500) \| (B.rank <= 2)` | `affinity <= 500 \| el.rank <= 2` |
| `0.5 * Affinity.score + 0.5 * Presentation.score` | `0.5 * affinity.score + 0.5 * presentation.score` |
| `Affinity.logistic(350, 150)` | `affinity.logistic(350, 150)` |
| `mean(Affinity.score, Presentation.score)` | `mean(affinity.score, presentation.score)` |
| `wt.Affinity.score` | `wt.affinity.score` |
| `Len()` / `Count("C")` | `len` / `count('C')` |
| `Column("cysteine_count")` | `column(cysteine_count)` |

## Exclusion filtering

Exclude peptides that appear in reference proteomes — useful for tumor-specific or pathogen-specific peptides:

```bash
--exclude-ensembl                    # Exclude peptides in the human Ensembl proteome
--exclude-non-cta                    # Exclude non-CTA proteins (requires pirlygenes)
--exclude-tissues heart_muscle lung  # Exclude genes expressed in these tissues
--exclude-fasta reference.fasta      # Exclude peptides in custom reference sequences
--exclude-mode substring             # "substring" (default) or "exact"
```

## Output

```bash
--output-csv results.csv
--output-html results.html
--output-csv-sep "\t"
--subset-output-columns peptide allele affinity
--rename-output-column value ic50
```

**All predictions:** `source_sequence_name`, `peptide`, `peptide_offset`, `peptide_length`, `allele`, `kind`, `score`, `value`, `affinity`, `percentile_rank`, `prediction_method_name`

**Variant predictions add:** `variant`, `gene`, `gene_id`, `transcript_id`, `transcript_name`, `effect`, `effect_type`, `contains_mutant_residues`, `mutation_start_in_peptide`, `mutation_end_in_peptide`

## Peptide properties

Compute amino acid properties and use them in ranking:

```python
from topiary.properties import add_peptide_properties

df = predictor.predict_from_named_sequences(seqs)
df = add_peptide_properties(df, groups=["manufacturability"])

# Properties become ranking signals
score = Affinity.logistic(350, 150) - 0.1 * Column("cysteine_count")
# CLI: --rank-by "affinity.logistic(350, 150) - 0.1 * column(cysteine_count)"
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
