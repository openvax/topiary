from topiary.cli.arg_parser import arg_parser

from .data import cancer_test_variants
"""
Helper functions for building commandline arguments which are used in
tests of the Topiary script and its components.
"""

def build_commandline_args_list(
        epitope_lengths=[9, 10],
        alleles=["HLA-A*02:01", "C0701"],
        predictor="netmhcpan",
        variants=cancer_test_variants):
    """
    Creates list of commandline arguments to Topiary script
    """
    args_list = [
        "--mhc-predictor", "netmhc",
        "--mhc-epitope-lengths", ",".join(str(x) for x in epitope_lengths),
        "--mhc-alleles", ",".join(alleles),
        "--reference-name", "GRCh38",
        "--only-novel-epitopes",
    ]
    for variant in variants:
        args_list.append("--variant")
        args_list.append(str(variant.contig))
        args_list.append(str(variant.start))
        args_list.append(variant.ref)
        args_list.append(variant.alt)
    return args_list

def build_and_parse_cli_args(**kwargs):
    args_list = build_commandline_args_list(**kwargs)
    return arg_parser.parse_args(args_list)
