# Topiary

Topiary predicts which peptides from an antigen source are presented by a
patient's MHC molecules, and therefore which could be seen by T cells, then
filters and ranks them. Give it protein sequences (or peptides, somatic
variants, or another pipeline's report) plus a list of MHC alleles, and it
scores every candidate peptide with one or more
[mhctools](https://github.com/openvax/mhctools) prediction models and returns a
DataFrame ranked by any combination of binding affinity, presentation,
processing and stability.

It is used in cancer immunotherapy research to find mutant peptides
(neoantigens) that the immune system could target, and in viral epitope mapping.
In the openvax stack, Topiary translates variants with
[varcode](https://github.com/openvax/varcode) and
[pyensembl](https://github.com/openvax/pyensembl), reads RNA evidence from
[isovar](https://github.com/openvax/isovar), and supplies the predictions and
ranking expressions that [vaxrank](https://github.com/openvax/vaxrank) uses.

## Install

Topiary requires Python 3.11 or newer.

```bash
pip install topiary
pip install mhcflurry && mhcflurry-downloads fetch
```

MHC predictors are installed separately and called through mhctools. MHCflurry
installs with pip as above; NetMHCpan and the other DTU tools need a license
from [DTU Health Tech](https://services.healthtech.dtu.dk/). Variant
annotation, gene lookups and `SelfProteome` need a local Ensembl annotation
database:

```bash
pyensembl install --release "$(python -c 'from pyensembl import ensembl_grch38; print(ensembl_grch38.release)')" --species human
```

The normal install includes Isovar for RNA-derived protein fragments,
PirlyGenes and OncoRef for cancer-testis antigens and tissue expression,
and WEIRDO for peptide foreignness. See
[installation details](cli.md#installation-details) for choosing a different
Ensembl release and compatible package versions.

## Quick start

Score two viral peptides against two HLA alleles with MHCflurry:

```python
from mhctools import MHCflurry
from topiary import TopiaryPredictor

peptides = {"influenza_M1": "GILGFVFTL", "CMV_pp65": "TPRVTGGGAM"}
columns = ["peptide", "allele", "kind", "value", "score", "percentile_rank"]

predictor = TopiaryPredictor(
    models=[MHCflurry],
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
)
df = predictor.predict_from_named_peptides(peptides)
print(df[columns].to_string(index=False))
```

```
   peptide      allele               kind        value    score  percentile_rank
 GILGFVFTL HLA-A*02:01      pMHC_affinity    25.841361 0.699441         0.130863
 GILGFVFTL HLA-B*07:02      pMHC_affinity 14185.201956 0.116437         8.844587
 GILGFVFTL HLA-A*02:01  pMHC_presentation          NaN 0.850049         0.193547
 GILGFVFTL             antigen_processing          NaN 0.175628              NaN
TPRVTGGGAM HLA-A*02:01      pMHC_affinity   556.503394 0.415730         1.541447
TPRVTGGGAM HLA-B*07:02      pMHC_affinity    14.428557 0.753303         0.003906
TPRVTGGGAM HLA-B*07:02  pMHC_presentation          NaN 0.941776         0.062445
TPRVTGGGAM             antigen_processing          NaN 0.300611              NaN
```

Each row is one peptide, allele and prediction kind. `value` is the raw
prediction (IC50 in nM for binding affinity), `score` is higher-is-better and
`percentile_rank` is lower-is-better. MHCflurry predicts affinity for each
allele, presentation for the genotype (reported under its best allele) and
antigen processing with no allele.

`filter_by` keeps only the predictions that pass an expression:

```python
from topiary import Affinity

predictor = TopiaryPredictor(
    models=[MHCflurry],
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
    filter_by=Affinity <= 500,
)
df = predictor.predict_from_named_peptides(peptides)
print(df[columns].to_string(index=False))
```

```
   peptide      allele               kind     value    score  percentile_rank
 GILGFVFTL HLA-A*02:01      pMHC_affinity 25.841361 0.699441         0.130863
 GILGFVFTL HLA-A*02:01  pMHC_presentation       NaN 0.850049         0.193547
 GILGFVFTL             antigen_processing       NaN 0.175628              NaN
TPRVTGGGAM HLA-B*07:02      pMHC_affinity 14.428557 0.753303         0.003906
TPRVTGGGAM HLA-B*07:02  pMHC_presentation       NaN 0.941776         0.062445
TPRVTGGGAM             antigen_processing       NaN 0.300611              NaN
```

`sort_by` ranks with the same expressions; the [ranking DSL](ranking.md) has
the full grammar. The [command-line interface](cli.md) runs the same pipeline
from variants, sequence files or gene names.

## Where to go next

1. [Quickstart](quickstart.md): scanning whole proteins, more input formats,
   Ensembl and tissue lookups, exclusion filtering, multiple models.
2. [Command-line interface](cli.md): predictor names, input modes, output
   columns and exit status.
3. [Ranking DSL](ranking.md): the filter and ranking expressions, in Python and
   as strings.
4. [Protein fragments](fragments.md): one antigen record for somatic variants,
   fusions, ERVs, CTAs, viral and synthetic sources.
5. [Cached predictions](cached.md): iterate on filters and ranking without
   re-running a predictor.
6. [API reference](api.md): classes, functions and complete column lists.

Several MHC prediction models can run in one pass and be compared or combined in
one ranking expression: NetMHCpan, [MHCflurry](https://github.com/openvax/mhcflurry),
[NetMHCIIpan](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/) and
others. Reports from [pVACseq](https://pvactools.readthedocs.io/),
[LENS](https://uselens.io/) and [Exacto](exacto.md) can be read in and
ranked beside fresh predictions, and peptides can be excluded by vital-organ
proteome membership or prioritized by RNA expression.
