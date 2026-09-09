# Quickstart

## CLI

```bash
# From a FASTA of protein sequences
topiary --fasta proteins.fasta \
  --mhc-predictor netmhcpan --mhc-alleles A0201,B0702 \
  --filter-by "affinity <= 500 | presentation.rank <= 2" \
  --output-csv results.csv

# From specific genes
topiary --gene-names BRAF TP53 EGFR \
  --mhc-predictor netmhcpan --mhc-alleles A0201 \
  --filter-by "affinity <= 500" \
  --output-csv results.csv

# Score specific peptides (no sliding window)
topiary --peptide-csv peptides.csv \
  --mhc-predictor netmhcpan --mhc-alleles A0201 \
  --ic50-cutoff 500 \
  --output-csv results.csv

# CTA proteins, excluding vital organ peptides
topiary --cta \
  --exclude-tissues heart_muscle lung liver \
  --mhc-predictor netmhcpan --mhc-alleles A0201 \
  --filter-by "affinity <= 500" \
  --output-csv results.csv

# From genomic variants (VCF)
topiary --vcf somatic.vcf \
  --mhc-predictor netmhcpan --mhc-alleles A0201 \
  --filter-by "affinity <= 500" \
  --only-novel-epitopes \
  --output-csv results.csv

# Multi-model with tool-qualified ranking
topiary --fasta proteins.fasta \
  --mhc-predictor netmhcpan --mhc-alleles A0201 \
  --filter-by "netmhcpan_ba <= 500 & column(cysteine_count) <= 2" \
  --sort-by "netmhcpan_affinity,mhcflurry_presentation" \
  --output-csv results.csv
```

## Python API

### Basic prediction

```python
from topiary import TopiaryPredictor, Affinity
from mhctools import NetMHCpan

predictor = TopiaryPredictor(
    models=NetMHCpan,
    alleles=["A0201"],
    filter_by=Affinity <= 500,
)
df = predictor.predict_from_sequences(["MASIINFEKLGGG"])
```

The returned DataFrame has one row per `(peptide, allele, kind)` with
columns including:

| Column | Meaning |
|---|---|
| `peptide` | Predicted peptide sequence |
| `allele` | HLA allele |
| `peptide_length` | Length of the peptide |
| `kind` | Prediction type (`pMHC_affinity`, `pMHC_presentation`, etc.) |
| `affinity` | Predicted IC50 binding affinity (nM) |
| `percentile_rank` | Percentile rank (lower = stronger binder) |
| `score` | Model score (not necessarily a probability or bounded by 1) |
| `prediction_method_name` | Which predictor produced this row |

Variant-derived predictions add `gene`, `variant`, `effect`,
`contains_mutant_residues`, `wt_peptide`, and more — see the
[API Reference](api.md) for the full column list.

### Whole-peptide half-life models

With Topiary 5.53.0, the Python API accepts mhctools' `PeptiVerse` and
`PlifePred2` instances, classes, or public class names. Configure the external
model installation through mhctools, then score **the complete peptide**:

```python
from mhctools import PeptiVerse
from topiary import TopiaryPredictor, evaluate_scores, parse

model = PeptiVerse(peptiverse_home="/path/to/PeptiVerse")
df = TopiaryPredictor(models=model).predict_from_named_peptides({
    "candidate": "SIINFEKLGGALQ",
})
scores = evaluate_scores(df, parse("peptide_view(serum_half_life.value)"))
```

No HLA allele is needed. Protein/fragment scanning rejects these peptide-only
models and points to `predict_from_named_peptides`; Topiary does not invent
scanning lengths or assign the full peptide's half-life to its inner ligands.

`serum_half_life`, `blood_half_life`, and `pMHC_stability` are separate kinds.
The first two describe the peptide without MHC; the third describes a
peptide–MHC complex. Known physical half-lives use `value` in hours. An unknown
unit stays missing rather than being replaced by the native `score`.
Do not assume a half-life score is a probability, or infer a percentile that
the model does not supply. Use explicit `peptide_view(...)` when combining
peptide-level evidence with per-allele predictions.

Topiary tests the real wrapper interfaces with synthetic sidecar output through
prediction, cache replay, TSV, wide/long conversion, and ranking. This verifies
data transport, **not biological accuracy or suitability for long vaccines**.
The external models and their runtime environments are not bundled. PlifePred2
has upstream [workspace-safety](https://github.com/openvax/mhctools/issues/310)
and [unit/assay validation](https://github.com/openvax/mhctools/issues/311)
issues addressed in mhctools 3.39.0, now required by Topiary. Its native
output is not a verified half-life in hours by default.

These models do not supply per-bond extracellular peptidase maps. Site-level
cleavage, full assay/chemistry provenance and configuration-aware cache identity
remain tracked in [Topiary #288](https://github.com/openvax/topiary/issues/288).

### Multiple models

```python
from topiary import Affinity, Presentation
from mhctools import NetMHCpan, MHCflurry

predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["A0201", "A0301", "B0702"],
    filter_by=Affinity <= 500,
    sort_by=Presentation.score,
)
```

When multiple models produce the same prediction kind, qualify with brackets:

```python
Affinity["netmhcpan"] <= 500           # NetMHCpan only
Affinity["mhcflurry"].score            # MHCflurry only
Presentation["mhcflurry"].rank <= 2    # MHCflurry presentation
```

See [Ranking DSL](ranking.md) for full details.

### Input formats

```python
from topiary.inputs import (
    read_fasta, read_peptide_csv, read_peptide_fasta,
    read_sequence_csv, slice_regions,
)

# Protein FASTA (scanned with sliding window)
seqs = read_fasta("proteins.fasta")

# Peptide FASTA (each entry predicted as-is, no sliding window)
seqs = read_peptide_fasta("peptides.fasta")

# CSV with 'peptide' column (as-is)
seqs = read_peptide_csv("peptides.csv")

# CSV with 'sequence' column (sliding window)
seqs = read_sequence_csv("proteins.csv")

# Region slicing (half-open intervals)
sliced = slice_regions(seqs, {
    "spike": [(319, 541)],           # RBD only
    "nucleocapsid": [(0, 50), (350, 419)],
})
```

### Ensembl and tissue lookups

```python
from topiary.sources import (
    sequences_from_gene_names,
    sequences_from_gene_ids,
    sequences_from_transcript_ids,
    tissue_expressed_sequences,
    cta_sequences,
    ensembl_proteome,
    available_tissues,
)

# Specific genes (uses longest protein-coding transcript)
seqs = sequences_from_gene_names(["BRAF", "TP53", "EGFR"])

# By Ensembl ID
seqs = sequences_from_gene_ids(["ENSG00000157764"])
seqs = sequences_from_transcript_ids(["ENST00000288602"])

# Tissue-expressed genes (requires pirlygenes)
seqs = tissue_expressed_sequences(["testis"], min_ntpm=10.0)

# CTA genes (requires pirlygenes)
seqs = cta_sequences()

# Full Ensembl proteome
seqs = ensembl_proteome()

# List available tissues
print(available_tissues())
```

### Exclusion

```python
from topiary.inputs import exclude_by
from topiary.sources import tissue_expressed_sequences

vital = tissue_expressed_sequences(["heart_muscle", "lung", "liver"])

# Substring mode: heart 8-mer inside CTA 9-mer -> excluded
df = exclude_by(df, vital, mode="substring")

# Exact mode: only full peptide matches
df = exclude_by(df, vital, mode="exact")
```

### Peptide properties

```python
from topiary.properties import add_peptide_properties

df = add_peptide_properties(df, groups=["manufacturability"])
# Adds: charge, hydrophobicity, aromaticity, molecular_weight,
#   cysteine_count, instability_index, max_7mer_hydrophobicity,
#   cterm_7mer_hydrophobicity, difficult_nterm, difficult_cterm, asp_pro_bonds
```

See [Peptide Properties](properties.md) for all available properties and groups.

### From somatic variants

```python
from varcode import load_vcf

variants = load_vcf("somatic.vcf")
df = predictor.predict_from_variants(variants)
# DataFrame includes:
#   fragment_id, source_type, variant, gene, gene_id, transcript_id,
#   transcript_name, effect, effect_type, contains_mutant_residues,
#   overlaps_target, mutation_start_in_peptide, mutation_end_in_peptide,
#   wt_peptide, wt_peptide_length, ...
```

### From protein fragments (non-variant sources)

```python
from topiary import ProteinFragment

fragments = [
    ProteinFragment(
        fragment_id="HPV16_E6__abc12345",
        source_type="viral:hpv16",
        sequence="MHQKRTAMFQDPQERPRKLPQLCTELQTTIHDIILECVYCKQQLLRREVYDFAFRDLCIVYRDGNPYAVCDKCLKFYSKISEYRHYCYSLYGTTLEQQYNKPLCDLLIRCINCQKPLCPEEKQRHLDKKQRFHNIRGRWTGRCMSCCRSSRTRRETQL",
    ),
]
df = predictor.predict_from_fragments(fragments)
# Any origin — variants, SVs, ERVs, CTAs, viral, allergen, autoantigen,
# synthetic. fragment_id threads through so downstream tools (vaxrank,
# vaccine-window selection) can group peptides back to their fragment.
```

### From cached predictions (skip the predictor call)

Use a pre-computed prediction table instead of running the predictor
live — for reproducibility, iterating on filters/ranking without
paying the predictor cost, or ingesting output from another tool.

From a prior topiary run:

```python
from topiary import CachedPredictor, TopiaryPredictor

cache = CachedPredictor.from_topiary_output("prior_run.parquet")
predictor = TopiaryPredictor(models=cache)
df = predictor.predict_from_variants(variants)
```

From mhcflurry or NetMHCpan output:

```python
cache = CachedPredictor.from_mhcflurry("mhcflurry_predictions.csv")
cache = CachedPredictor.from_netmhcpan_stdout("netmhcpan_run.out")
```

From the CLI (format auto-detected):

```bash
topiary --peptide-csv peptides.csv \
    --mhc-cache-file netmhcpan_run.out \
    --output-csv results.csv
```

Full details — all loaders, sharding, version invariants, and
the CLI flag reference — in [Cached Predictions](cached.md).
