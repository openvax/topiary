[![Tests](https://github.com/openvax/topiary/actions/workflows/tests.yml/badge.svg)](https://github.com/openvax/topiary/actions/workflows/tests.yml)
<a href="https://coveralls.io/github/openvax/topiary?branch=master">
    <img src="https://coveralls.io/repos/openvax/topiary/badge.svg?branch=master&service=github" alt="Coverage Status" />
</a>
<a href="https://pypi.python.org/pypi/topiary/">
    <img src="https://img.shields.io/pypi/v/topiary.svg?maxAge=1000" alt="PyPI" />
</a>

# Topiary

Topiary predicts which peptides from an antigen source are presented by a
patient's MHC molecules, then filters and ranks them. Inputs are protein
sequences, peptides, somatic variants or a pipeline's report, plus a list of HLA
alleles; output is a DataFrame (or CSV/TSV/HTML) with one row per
peptide-allele-prediction. Binding predictions come from
[mhctools](https://github.com/openvax/mhctools) models such as NetMHCpan and
MHCflurry.

It is used for personalized cancer vaccine design, viral epitope mapping, and
characterizing T-cell responses.

Documentation: [openvax.github.io/topiary](https://openvax.github.io/topiary/)

## Installation

Topiary requires Python 3.10 or newer.

```bash
pip install topiary
```

MHC binding predictors are installed separately and called through mhctools.
MHCflurry installs with pip:

```bash
pip install mhcflurry && mhcflurry-downloads fetch
```

NetMHCpan and the other DTU tools require a license from
[DTU Health Tech](https://services.healthtech.dtu.dk/).

Variant annotation, gene lookups and `SelfProteome` need a local Ensembl
annotation database:

```bash
pyensembl install --release "$(python -c 'from pyensembl import ensembl_grch38; print(ensembl_grch38.release)')" --species human
```

Two optional extras add input sources: `pip install 'topiary[pirlygenes]'` for
cancer-testis antigens and tissue expression, and
`pip install 'topiary[isovar]'` for protein fragments assembled from RNA reads.
See [installation details](https://openvax.github.io/topiary/cli/#installation-details)
for choosing a different Ensembl release and for the optional-package versions.

## Predict binding for a few peptides

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

Each row of `df` is one peptide, allele and prediction kind, with `value` (the
raw prediction, IC50 in nM for binding affinity), `score` (normalized, higher is
better) and `percentile_rank` (lower is better). Several models can run
together — `models=[NetMHCpan, MHCflurry]` — and each contributes its own rows.

To scan whole proteins with a sliding window instead of scoring peptides as
given, use `predict_from_named_sequences({"BRAF": "MAALSGGGG..."})`.

## Predict from somatic variants

The CLI runs the same pipeline from a VCF, translating each variant, tiling
peptides across the mutated region, and keeping only peptides that overlap it:

```bash
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --filter-by "ba <= 500 | el.rank <= 2" \
  --only-novel-epitopes \
  --output-csv epitopes.csv
```

The equivalent Python call is `predictor.predict_from_variants(variants)` on a
[varcode](https://github.com/openvax/varcode) variant collection.

## Filter and rank

Filters and ranking expressions are written the same way in Python and as
strings for the CLI:

```python
from topiary import TopiaryPredictor, Affinity, Presentation
from mhctools import NetMHCpan

predictor = TopiaryPredictor(
    models=[NetMHCpan],
    alleles=["HLA-A*02:01"],
    filter_by=(Affinity <= 500) | (Presentation.rank <= 2.0),
    sort_by=Affinity.logistic(350, 150),
)
```

`Affinity`, `Presentation`, `Stability` and `Processing` each read a prediction
kind, through `.value`, `.rank` or `.score`. Expressions combine with
arithmetic and boolean operators, normalizing transforms, and any column in the
DataFrame — RNA expression, peptide properties, custom annotations. The string
forms (`affinity <= 500 | el.rank <= 2`) go to `--filter-by` and `--sort-by`.
The [ranking DSL guide](https://openvax.github.io/topiary/ranking/) has the
full grammar.

## Where to read more

| Guide | Covers |
|---|---|
| [Quickstart](https://openvax.github.io/topiary/quickstart/) | Input formats, Ensembl lookups, exclusion filtering, peptide-only models |
| [Command-line interface](https://openvax.github.io/topiary/cli/) | Predictor names, input modes, output columns, exit status |
| [Ranking DSL](https://openvax.github.io/topiary/ranking/) | Filter and ranking expressions, multi-model disambiguation |
| [Protein fragments](https://openvax.github.io/topiary/fragments/) | One antigen record for variants, fusions, ERVs, CTAs, viral and synthetic sources |
| [RNA expression](https://openvax.github.io/topiary/expression/) | Annotating predictions with Salmon, Kallisto, RSEM and other quantification |
| [Cached predictions](https://openvax.github.io/topiary/cached/) | Replaying stored scores instead of calling a predictor |
| [Peptide properties](https://openvax.github.io/topiary/properties/) | Charge, hydrophobicity, manufacturability and TCR-facing signals |
| [Cross-reactivity](https://openvax.github.io/topiary/self_proteome/) | Nearest-self search and presentation evidence |
| [Reading pVACseq](https://openvax.github.io/topiary/pvacseq/), [Exacto](https://openvax.github.io/topiary/exacto/), [combining sources](https://openvax.github.io/topiary/combined-sources/) | Importing other pipelines' reports and ranking them together |
| [SV evidence reports](https://openvax.github.io/topiary/sv-interest/) | Complete structural-variant nomination inventories, including candidates with no recovered protein |
| [API reference](https://openvax.github.io/topiary/api/) | Classes, functions and complete column lists |

## Contributing

Run `./lint.sh` and `./test.sh` before submitting changes. The full suite uses
real Ensembl annotations and external MHC predictors; collecting tests does not
load either:

```bash
python -m pytest tests --collect-only
```

The synthetic Isovar integration tests need only the optional package:

```bash
pip install '.[isovar]' pytest pytest-cov pytest-xdist
./test.sh -m isovar --strict-markers
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
