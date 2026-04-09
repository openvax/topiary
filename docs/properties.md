# Peptide Properties

Compute amino acid properties on any peptide column and use them in ranking expressions via `Column()`.

## Usage

```python
from topiary.properties import add_peptide_properties

# Add all properties
df = add_peptide_properties(df)

# Named group
df = add_peptide_properties(df, groups=["manufacturability"])

# Specific properties
df = add_peptide_properties(df, include=["charge", "cysteine_count"])

# Properties for a comparison peptide (e.g. wildtype)
df = add_peptide_properties(df, peptide_column="wt_peptide", prefix="wt_")
```

The function returns a copy — the original DataFrame is not modified.

## Named groups

### `"core"` — basic biophysical properties

| Column | Description |
|--------|-------------|
| `charge` | Net charge at pH 7.4 (K,R = +1; D,E = -1; H = +0.1) |
| `hydrophobicity` | Mean Kyte-Doolittle hydropathy score |
| `aromaticity` | Count of aromatic residues (F, W, Y) |
| `molecular_weight` | Molecular weight in Daltons |

### `"manufacturability"` — peptide synthesis feasibility

Includes all core properties, plus:

| Column | Description | Priority |
|--------|-------------|----------|
| `cysteine_count` | Number of cysteine residues (disulfide bond risk) | Highest |
| `instability_index` | Guruprasad dipeptide instability (>40 = unstable) | High |
| `max_7mer_hydrophobicity` | Peak hydropathy in any 7-residue window (aggregation hotspot) | High |
| `cterm_7mer_hydrophobicity` | Mean hydropathy of C-terminal 7 residues | Medium |
| `difficult_nterm` | N-terminal Q, E, or C (problematic for synthesis) | Medium |
| `difficult_cterm` | C-terminal P or C (blocks coupling / aggregation) | Medium |
| `asp_pro_bonds` | Count of Asp-Pro dipeptides (hydrolysis-prone) | Low |

These mirror the manufacturability criteria used by [Vaxrank](https://github.com/openvax/vaxrank) for vaccine peptide selection.

### `"immunogenicity"` — TCR recognition signals

Includes all core properties, plus:

| Column | Description |
|--------|-------------|
| `tcr_charge` | Net charge of TCR-facing residues |
| `tcr_aromaticity` | Count of aromatic residues at TCR-facing positions |
| `tcr_hydrophobicity` | Mean hydropathy of TCR-facing residues |

TCR-facing positions depend on peptide length (MHC-I):

| Length | TCR-facing positions (0-indexed) |
|--------|----------------------------------|
| 8-mer | 3, 4, 5, 6 |
| 9-mer | 3, 4, 5, 7 |
| 10-mer | 3, 4, 5, 6, 8 |
| 11-mer | 3, 4, 5, 6, 7, 9 |

For peptides outside 8-11 residues, TCR properties are NaN.

## Using properties in ranking

Properties become ranking signals via `Column()`:

```python
from topiary.ranking import Affinity, Column

# Simple: penalize cysteines
score = Affinity.logistic(350, 150) - 0.1 * Column("cysteine_count")

# Complex: combine multiple signals
score = (
    0.5 * Affinity.logistic(350, 150)
    - 0.1 * Column("cysteine_count")
    - 0.1 * abs(Column("charge"))           # prefer neutral peptides
    + 0.1 * Column("tcr_aromaticity")       # reward aromatic TCR contacts
    - 0.05 * Column("instability_index").clip(lo=0, hi=100).left_cdf(50, 20)
)
```

On the CLI, filter by property values:

```bash
--ranking "affinity <= 500 & column(cysteine_count) <= 1"
```

## Comparing mutant vs wildtype properties

```python
# Compute properties for both mutant and WT peptides
df = add_peptide_properties(df, groups=["core"])
df = add_peptide_properties(df, peptide_column="wt_peptide", prefix="wt_",
                            groups=["core"])

# Compare: is the mutant more hydrophobic than wildtype?
Column("hydrophobicity") - Column("wt_hydrophobicity")
```

## Available properties

```python
from topiary.properties import available_properties

# Returns dict of property_name -> set of groups it belongs to
print(available_properties())
```
