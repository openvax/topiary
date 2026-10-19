# Topiary

Topiary predicts which peptides from a protein sequence will be presented by
MHC molecules, and therefore which could be seen by T cells. Give it protein
sequences (or peptides, somatic variants, or another pipeline's report) plus a
list of HLA alleles, and it scans every candidate peptide with one or more
[mhctools](https://github.com/openvax/mhctools) prediction models and returns
the predicted binders as a DataFrame, ranked by any combination of binding
affinity, presentation, processing and stability.

It is used in cancer immunotherapy research to find mutant peptides
(neoantigens) that the immune system could target, and in viral epitope mapping.

## Installation

Topiary requires Python 3.11 or newer.

```bash
pip install topiary
```

Binding predictors are installed separately and called through mhctools:
MHCflurry with `pip install mhcflurry && mhcflurry-downloads fetch`, NetMHCpan
and the other DTU tools under a license from
[DTU Health Tech](https://services.healthtech.dtu.dk/).

Variant annotation, gene lookups and `SelfProteome` need a local Ensembl
annotation database:

```bash
pyensembl install --release "$(python -c 'from pyensembl import ensembl_grch38; print(ensembl_grch38.release)')" --species human
```

Two extras add input sources: `pip install 'topiary[pirlygenes]'` for
cancer-testis antigens and tissue expression, and `pip install 'topiary[isovar]'`
for protein fragments assembled from RNA reads. See
[installation details](cli.md#installation-details) for choosing a different
Ensembl release and for the optional-package versions.

## A first prediction

Score the [PRAME](https://www.uniprot.org/uniprotkb/P78395/) cancer-testis
antigen against two HLA class I alleles, keeping strong binders and presentation
hits, sorted by presentation score:

```python
from topiary import TopiaryPredictor
from mhctools import NetMHCpan

PRAME = (
    "MERRRLWGSIQSRYISMSVWTSPRRLVELAGQSLLKDEALAIAALELLPRELFPPLFMAA"
    "FDGRHSQTLKAMVQAWPFTCLPLGVLMKGQHLHLETFKAVLDGLDVLLAQEVRPRRWKLQ"
    "VLDLRKNSHQDFWTVWSGNRASLYSFPEPEAAQPMTKKRKVDGLSTEAEQPFIPVEVLVD"
    "LFLKEGACDELFSYLIEKVKRKKNVLRLCCKKLKIFAMPMQDIKMILKMVQLDSIEDLEV"
    "TCTWKLPTLAKFSPYLGQMINLRRLLLSHIHASSYISPEKEEQYIAQFTSQFLSLQCLQA"
    "LYVDSLFFLRGRLDQLLRHVMNPLETLSITNCRLSEGDVMHLSQSPSVSQLSVLSLSGVM"
    "LTDVSPEPLQALLERASATLQDLVFDECGITDDQLLALLPSLSHCSQLTTLSFYGNSISI"
    "SALQSLLQHLIGLSNLTHVLYPVPLESYEDIHGTLHLERLAYLHARLRELLCELGRPSMV"
    "WLSANPCPHCGDRTFYDPEPILCPCFMPN"
)

predictor = TopiaryPredictor(
    models=NetMHCpan,
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
    filter_by="affinity <= 500 | el.rank <= 2",
    sort_by="el.score",
)

df = predictor.predict_from_named_sequences({"PRAME": PRAME})
```

## Where to go next

- [Quickstart](quickstart.md) — more input formats, Ensembl and tissue lookups,
  exclusion filtering.
- [Command-line interface](cli.md) — predictor names, input modes, output
  columns and exit status.
- [Ranking DSL](ranking.md) — the filter and ranking expressions, in Python and
  as strings.
- [Protein fragments](fragments.md) — one antigen record for somatic variants,
  fusions, ERVs, CTAs, viral and synthetic sources.
- [Cached predictions](cached.md) — iterate on filters and ranking without
  re-running a predictor.
- [API reference](api.md) — classes, functions and complete column lists.

Several MHC prediction models can run in one pass and be compared or combined in
one ranking expression: NetMHCpan, [MHCflurry](https://github.com/openvax/mhcflurry),
[NetMHCIIpan](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/) and
others. Reports from [pVACseq](https://pvactools.readthedocs.io/),
[LENS](https://www.uselens.io/) and [Exacto](exacto.md) can be read in and
ranked beside fresh predictions, and peptides can be excluded by vital-organ
proteome membership or prioritized by RNA expression.
