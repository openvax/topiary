from nose.tools import eq_

from topiary import predict_epitopes_from_args
from topiary.commandline_args import arg_parser

from .data import cancer_test_variants


def test_cancer_epitopes_from_args():
    epitope_lengths = [9, 10]
    alleles = ["HLA-A*02:01", "C0701"]
    args_list = [
        "--mhc-predictor", "netmhc",
        "--mhc-epitope-lengths", ",".join(str(x) for x in epitope_lengths),
        "--mhc-alleles", ",".join(alleles),
        "--reference-name", "GRCh38",
        "--only-novel-epitopes",
    ]
    for variant in cancer_test_variants:
        args_list.append("--variant")
        args_list.append(str(variant.contig))
        args_list.append(str(variant.start))
        args_list.append(variant.ref)
        args_list.append(variant.alt)

    parsed_args = arg_parser.parse_args(args_list)
    epitope_predictions = predict_epitopes_from_args(parsed_args)
    expected_number_of_epitopes = 0
    for epitope_length in epitope_lengths:
        expected_number_of_epitopes += epitope_length * len(cancer_test_variants) * len(alleles)
    eq_(len(epitope_predictions), expected_number_of_epitopes)
