[![Build Status](https://travis-ci.org/hammerlab/topiary.svg?branch=master)](https://travis-ci.org/hammerlab/topiary) [![Coverage Status](https://coveralls.io/repos/hammerlab/topiary/badge.svg?branch=master&service=github)](https://coveralls.io/github/hammerlab/topiary?branch=master) [![DOI](https://zenodo.org/badge/18834/hammerlab/topiary.svg)](https://zenodo.org/badge/latestdoi/18834/hammerlab/topiary)

# Topiary
Predict mutation-derived cancer T-cell epitopes from (1) somatic variants (2) tumor RNA expression data, and (3) patient HLA type.

## Example

```sh
./topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --ic50-cutoff 500 \
  --percentile-cutoff 2.0 \
  --mhc-epitope-lengths 8-11 \
  --rna-gene-fpkm-tracking-file genes.fpkm_tracking \
  --rna-min-gene-expression 4.0 \
  --rna-transcript-fpkm-tracking-file isoforms.fpkm_tracking \
  --rna-min-transcript-expression 1.5 \
  --output-csv epitopes.csv \
  --output-html epitopes.html
```

## Installation

You can install Topiary and all of the libraries it depends on by running:
```
pip install topiary
```

You'll need to download the reference genome sequences and annotations for a
recent Ensembl release (e.g. 81) by running:

```
pyensembl install --release 81 --species human
```

If you want to work with variants which were aligned against the older reference
GRCh37, you will need to also download its annotation data, which is contained
in Ensembl release 75:

```
pyensembl install --release 75 --species human
```


## Commandline Arguments

### Genomic Variants

Specify some variants by giving at least one of the following options. They can
be used in combination and repeated.

* `--vcf VCF_FILENAME`: Load a [VCF](http://www.1000genomes.org/wiki/analysis/variant%20call%20format/vcf-variant-call-format-version-41) file
* `--maf MAF_FILENAME`: Load a TCGA [MAF](https://wiki.nci.nih.gov/display/TCGA/Mutation+Annotation+Format+%28MAF%29+Specification) file
* `--variant CHR POS REF ALT : Specify an individual variant (requires --ensembl-version)`

### Output Format

* `--output-csv OUTPUT_CSV_FILENAME`: Path to an output CSV file
* `--output-html OUTPUT_HTML_FILENAME`: Path to an output HTML file

### RNA Expression Filtering

Optional flags to use Cufflinks expression estimates for dropping epitopes
arising from genes or transcripts that are not highly expressed.

* `--rna-gene-fpkm-tracking-file RNA_GENE_FPKM_TRACKING_FILE`: Cufflinks FPKM tracking file
containing gene expression estimates.
* `--rna-min-gene-expression RNA_MIN_GENE_EXPRESSION`: Minimum FPKM for genes
* `--rna-transcript-fpkm-tracking-file RNA_TRANSCRIPT_FPKM_TRACKING_FILE`: Cufflinks FPKM tracking
file containing transcript expression estimates.
* `--rna-min-transcript-expression RNA_MIN_TRANSCRIPT_EXPRESSION`: Minimum FPKM
for transcripts
* `--rna-transcript-fpkm-gtf-file RNA_TRANSCRIPT_FPKM_GTF_FILE`: StringTie GTF file
file containing transcript expression estimates.

### Choose an MHC Binding Predictor

You *must* choose an MHC binding predictor using one of the following values
for the `--mhc-predictor` flag:

* `netmhc`: Local [NetMHC](http://www.cbs.dtu.dk/cgi-bin/nph-sw_request?netMHC) predictor (Topiary will attempt to automatically detect whether NetMHC 3.x or 4.0 is available)
* `netmhcpan`: Local [NetMHCpan](http://www.cbs.dtu.dk/cgi-bin/nph-sw_request?netMHCpan) predictor
* `netmhciipan`: Local [NetMHCIIpan](http://www.cbs.dtu.dk/cgi-bin/nph-sw_request?netMHCIIpan) predictor
* `netmhccons`: Local [NetMHCcons](http://www.cbs.dtu.dk/cgi-bin/nph-sw_request?netMHCcons)
* `random`: Random IC50 values
* `smm`: Local [SMM](http://www.mhc-pathway.net/smm) predictor
* `smm-pmbec`: Local [SMM-PMBEC](http://www.mhc-pathway.net/smmpmbec) predictor
* `netmhcpan-iedb`: Use NetMHCpan via the IEDB web API
* `netmhccons-iedb`: Use NetMHCcons via the IEDB web API
* `smm-iedb`: Use SMM via the IEDB web API
* `smm-pmbec-iedb`: Use SMM-PMBEC via the IEDB web API

### MHC Alleles
You must specify the alleles to perform binding prediction for using one of
the following flags:

* `--mhc-alleles-file MHC_ALLELES_FILE`: Text file containing one allele name per
line
* `--mhc-alleles MHC_ALLELES`: Comma separated list of allele names,
e.g. "HLA-A02:01,HLA-B07:02"

### Peptide Length

* `--mhc-epitope-lengths MHC_EPITOPE_LENGTHS`: comma separated list of integers
specifying which peptide lengths to use for MHC binding prediction

### Binding Prediction Filtering

* `--only-novel-epitopes`: Topiary will normally keep all predicted epitopes,
even those which occur in a given self-ligandome or don't overlap a mutated region
of a protein. Use this flag to drop any epitopes which don't contain mutations
or that occur elsewhere in the self-ligandome.
* `--ic50-cutoff IC50_CUTOFF`: Drop peptides with predicted IC50 nM greater
than this value (typical value is 500.0)
* `--percentile-cutoff PERCENTILE_CUTOFF`: Drop peptides with percentile rank
of their predicted IC50 (among predictions for a particular allele) fall below
this threshold (lower values are stricter filters, typical value is 2.0)

### Misc

* `--padding-around-mutation PADDING_AROUND_MUTATION`: Include more unmutated residues
around the mutation (useful when not using `--only-novel-epitopes`)
* `--self-filter-directory SELF_FILTER_DIRECTORY`: Directory of files named by MHC allele
containing a self peptide ligandome (peptides which should be excluded from
results)
* `--skip-variant-errors`: If a particular mutation causes an exception to be raised
during annotation, you can skip it using this flag.

