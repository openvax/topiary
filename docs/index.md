# Topiary

Predict which peptides from protein sequences will be presented by MHC molecules, making them potential T-cell epitopes. Used in cancer immunotherapy research to find mutant peptides (neoantigens) that the immune system could target.

**Core idea:** Given protein sequences + HLA alleles + one or more MHC prediction models, Topiary scans all possible peptides and returns those predicted to be presented by MHC, ranked by any combination of binding affinity, presentation score, processing score, and stability.

## Features

- **Multiple prediction models** — NetMHCpan, MHCflurry, NetMHCIIpan, etc. via [mhctools](https://github.com/openvax/mhctools)
- **Multi-model disambiguation** — `Affinity["netmhcpan"]` bracket syntax when combining models
- **Composable ranking DSL** — filter, rank, and score with operator expressions
- **Transforms** — `.logistic()`, `.ascending_cdf()`, `.descending_cdf()`, `.clip()`, `.hinge()`, `.log()` for composite scoring
- **Aggregations** — `mean()`, `geomean()`, `minimum()`, `maximum()`, `median()` for combining expressions
- **Arbitrary column access** — `Column("cysteine_count")` brings any DataFrame column into the DSL
- **Wildtype comparison** — `wt.Affinity.score` for differential binding analysis
- **Peptide-level expressions** — `len`, `count('C')`, `wt.len`, `wt.count('C')` for peptide properties in the DSL
- **Peptide properties** — charge, hydrophobicity, aromaticity, manufacturability, TCR-facing residue analysis
- **Multiple input modes** — VCF/MAF variants, FASTA, CSV, gene names, Ensembl lookups, CTA gene sets
- **Tissue-aware exclusion** — exclude peptides from vital-organ proteomes
- **Tab completion** — `pip install 'topiary[completion]'`

## Installation

```bash
pip install topiary
```

For Ensembl-based features:

```bash
pyensembl install --release 93 --species human
```

For cancer-testis antigen and tissue expression features:

```bash
pip install pirlygenes
```

## Quick example

```python
from topiary import TopiaryPredictor, Affinity, Presentation
from mhctools import NetMHCpan

predictor = TopiaryPredictor(
    models=NetMHCpan,
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
    filter=(Affinity <= 500) | (Presentation.rank <= 2.0),
    rank_by=Presentation.score,
)

df = predictor.predict_from_named_sequences({
    "BRAF_V600E": "MAALSGGGGG...LATEKSRWSG",
})
```

See the [Quickstart](quickstart.md) for more examples, [Ranking DSL](ranking.md) for the expression system, and [API Reference](api.md) for full details.
