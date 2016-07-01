
from topiary.cli.console_scripts import topiary

def test_topiary_cli():
    topiary(["--mhc-predictor", "netmhcpan", "--vcf", ""])
def topiary(raw_args=None):
    """
    Script to generate epitope predictions from somatic cancer variants
    and (optionally) tumor RNA-seq data.

    Example usage:
        topiary \
            --mhc-predictor netmhcpan
            --mhc-alleles-file HLA.txt
            --vcf somatic.vcf
            --rna-gene-fpkm-file genes.fpkm_tracking
            --rna-transcript-fpkm-file isoforms.fpkm_tracking
            --filter-ic50 500
            --filter-percentile 2
            --output results.csv
    """
    if raw_args is None:
        raw_args = sys.args[1:]

    arg_parser = make_arg_parser()
    parsed_args = arg_parser.parse_args()
    print("Topiary commandline arguments:")
    print(parsed_args)

    epitopes = predict_epitopes_from_args(parsed_args)
    gene_expression_dict = rna_gene_expression_dict_from_args(parsed_args)
    transcript_expression_dict = rna_transcript_expression_dict_from_args(parsed_args)
    df = epitopes_to_dataframe(
        epitopes,
        gene_expression_dict=gene_expression_dict,
        transcript_expression_dict=transcript_expression_dict)
    write_outputs(df, parsed_args)
    print("Total count: %d" % len(df))
