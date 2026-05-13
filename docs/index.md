# Topiary

Predict which peptides from protein sequences will be presented by MHC molecules, making them potential T-cell epitopes. Used in cancer immunotherapy research to find mutant peptides (neoantigens) that the immune system could target.

**Core idea:** Given protein sequences + HLA alleles + one or more MHC prediction models, Topiary scans all possible peptides and returns those predicted to be presented by MHC, ranked by any combination of binding affinity, presentation score, processing score, and stability.

## Features

- **Multiple MHC prediction models** — [NetMHCpan](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/), [MHCflurry](https://github.com/openvax/mhcflurry), [NetMHCIIpan](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/), etc. via [mhctools](https://github.com/openvax/mhctools); combine and rank across models
- **Composable ranking DSL** — filter, rank, and score with operator expressions over affinity, presentation, stability, wildtype comparisons, and peptide properties
- **Universal antigen abstraction** — `ProteinFragment` runs somatic variants, fusions, ERVs, CTAs, viral, and synthetic antigens through one pipeline
- **Cached predictions** — `CachedPredictor` reuses pre-computed scores ([mhctools](https://github.com/openvax/mhctools) output, [NetMHC](https://services.healthtech.dtu.dk/services/NetMHC-4.0/) stdout, generic TSV) so you can iterate on filters and ranking without re-running the predictor
- **Multiple input modes** — VCF/MAF variants, FASTA, CSV, gene names, [LENS](https://www.uselens.io/) reports
- **Expression- and tissue-aware prioritization** — exclude peptides from vital-organ proteomes, prioritize by RNA expression

## Quick example

Score the [PRAME](https://www.uniprot.org/uniprotkb/P78395/) cancer-testis antigen against two HLA class I alleles, keeping strong binders and presentation hits and sorting by presentation score:

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

## Installation

Requires Python ≥ 3.9.

```bash
pip install topiary
```

For Ensembl-based features (variant annotation, gene lookups,
`SelfProteome`):

```bash
pyensembl install --release 112 --species human
```

For cancer-testis antigen and tissue expression features:

```bash
pip install pirlygenes
```

See the [Quickstart](quickstart.md) for more examples, [Protein Fragments](fragments.md) for the universal antigen abstraction, [Cached Predictions](cached.md) for running from pre-computed scores, [Ranking DSL](ranking.md) for the expression system, and [API Reference](api.md) for full details.
