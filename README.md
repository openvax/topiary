[![Tests](https://github.com/openvax/topiary/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/openvax/topiary/actions/workflows/tests.yml)
<a href="https://coveralls.io/github/openvax/topiary?branch=master">
    <img src="https://coveralls.io/repos/openvax/topiary/badge.svg?branch=master&service=github" alt="Coverage Status" />
</a>
<a href="https://pypi.python.org/pypi/topiary/">
    <img src="https://img.shields.io/pypi/v/topiary.svg?maxAge=1000" alt="PyPI" />
</a>

# Topiary

Topiary predicts which peptides from an antigen source are presented by a
patient's MHC molecules, then filters and ranks them. Inputs are protein
sequences, peptides, somatic variants or another pipeline's report, plus a list
of MHC alleles; the output is a DataFrame (or CSV/HTML file) with one row per
peptide, allele and prediction. It is used for personalized cancer vaccine
design, viral epitope mapping and characterizing T-cell responses. In the
openvax stack, Topiary runs MHC predictors through
[mhctools](https://github.com/openvax/mhctools), translates variants with
[varcode](https://github.com/openvax/varcode) and
[pyensembl](https://github.com/openvax/pyensembl), reads RNA evidence from
[isovar](https://github.com/openvax/isovar), and supplies the predictions and
ranking expressions that [vaxrank](https://github.com/openvax/vaxrank) uses.

## Install

Topiary requires Python 3.11 or newer.

```bash
pip install topiary
```

MHC predictors are installed separately and called through mhctools.
MHCflurry installs with pip; its models are a separate download:

```bash
pip install mhcflurry
mhcflurry-downloads fetch
```

NetMHCpan and the other DTU tools need a license from
[DTU Health Tech](https://services.healthtech.dtu.dk/). Variant annotation,
gene lookups and `SelfProteome` need a local Ensembl annotation database:

```bash
pyensembl install --release "$(python -c 'from pyensembl import ensembl_grch38; print(ensembl_grch38.release)')" --species human
```

Isovar, PirlyGenes, OncoRef and WEIRDO are installed with Topiary; there are no
extras to choose. [Installation details](https://openvax.github.io/topiary/cli/#installation-details)
covers other Ensembl releases and compatible package versions.

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

Pass `filter_by=Affinity <= 500` (with `from topiary import Affinity`) to keep
only the pairs predicted to bind at 500 nM or tighter, and `sort_by=` to rank
them. Both take expressions that combine prediction fields, arithmetic,
normalizing transforms and any DataFrame column; see the
[ranking DSL guide](https://openvax.github.io/topiary/ranking/). To scan whole
proteins with a sliding window, use `predict_from_named_sequences`; for
[varcode](https://github.com/openvax/varcode) variants, use
`predict_from_variants`.

## Command line

The `topiary` command runs the same pipeline from variants, sequence files or
gene names. This run translates KRAS G12D, scores the mutant peptides and keeps
the binders (Ensembl data for `--genome grch38` must be installed as above):

```bash
topiary \
  --variant 12 25245350 C T \
  --genome grch38 \
  --mhc-predictor mhcflurry \
  --mhc-alleles 'HLA-A*11:01,HLA-C*08:02' \
  --filter-by "affinity <= 500" \
  --only-novel-epitopes
```

```
  peptide      allele               kind      value    score  percentile_rank                      source_sequence_name
VVGADGVGK HLA-A*11:01      pMHC_affinity 126.927288 0.552337         0.505538 KRAS_p.G12D_chr12_g.25245350C_T__567b9678
VVGADGVGK HLA-A*11:01  pMHC_presentation        NaN 0.727242         0.375163 KRAS_p.G12D_chr12_g.25245350C_T__567b9678
VVGADGVGK             antigen_processing        NaN 0.465049              NaN KRAS_p.G12D_chr12_g.25245350C_T__567b9678
GADGVGKSA HLA-C*08:02      pMHC_affinity 108.830353 0.566554         0.282237 KRAS_p.G12D_chr12_g.25245350C_T__567b9678
GADGVGKSA HLA-C*08:02  pMHC_presentation        NaN 0.534686         0.686717 KRAS_p.G12D_chr12_g.25245350C_T__567b9678
GADGVGKSA             antigen_processing        NaN 0.215621              NaN KRAS_p.G12D_chr12_g.25245350C_T__567b9678
6 prediction rows (2 unique peptides, 2 named alleles)
```

`--vcf`, `--maf`, `--fasta`, `--peptide-csv` and `--gene-names` select other
inputs, and `--output-csv results.csv` saves every row instead of printing a
preview. The [command-line guide](https://openvax.github.io/topiary/cli/)
lists predictor names, input modes, output columns and exit status.

## Documentation

The full documentation is at
[openvax.github.io/topiary](https://openvax.github.io/topiary/). In reading
order:

1. [Quickstart](https://openvax.github.io/topiary/quickstart/): protein scans, input files, Ensembl and tissue lookups, multiple models.
2. [Command-line interface](https://openvax.github.io/topiary/cli/): predictor names, input modes, output columns, exit status.
3. [Ranking DSL](https://openvax.github.io/topiary/ranking/): filter and ranking expressions in Python and as strings.
4. [Protein fragments](https://openvax.github.io/topiary/fragments/): one antigen record for variants, fusions, ERVs, CTAs, viral and synthetic sources.
5. [Cached predictions](https://openvax.github.io/topiary/cached/): replay stored scores instead of calling a predictor.
6. [API reference](https://openvax.github.io/topiary/api/): classes, functions and complete column lists.

Other guides cover RNA expression, peptide properties, cross-reactivity, and
reading pVACseq, LENS and Exacto reports.

## Development

```bash
./develop.sh                          # editable install (uses uv)
pip install -r requirements-dev.txt   # lint, test and docs tools
./lint.sh
./test.sh                             # full suite: needs Ensembl data and predictors
./test.sh -m isovar --strict-markers  # synthetic Isovar tests, no Ensembl download
mkdocs build --strict                 # docs site
```

Every pull request bumps `__version__` in `topiary/__init__.py` and adds a
[CHANGELOG](https://github.com/openvax/topiary/blob/master/CHANGELOG.md) entry;
[RELEASING.md](https://github.com/openvax/topiary/blob/master/RELEASING.md)
describes the release steps.

## License

Apache License 2.0. See [LICENSE](https://github.com/openvax/topiary/blob/master/LICENSE).
