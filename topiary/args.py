# Copyright (c) 2015. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Common commandline arguments used by scripts
"""

import argparse

from .common import parse_int_list

parser = argparse.ArgumentParser()

#
# Genomic Variants
#
variant_arg_group = parser.add_argument_group(
    title="Variants",
    description="Genomic variant files")

variant_arg_group.add_argument("--vcf",
    default=[],
    action="append",
    help="Genomic variants in VCF format",)

variant_arg_group.add_argument("--maf",
    default=[],
    action="append",
    help="Genomic variants in TCGA's MAF format",)

#
# MHC Binding Prediction
#

mhc_predictor_arg_group = parser.add_argument_group(
    title="MHC",
    description="Which MHC binding predictor to use (default NetMHCpan)")

mhc_predictor_arg_group.add_argument("--mhc-random",
    default=False,
    action="store_true",
    help="Random values instead for MHC binding prediction")

mhc_predictor_arg_group.add_argument("--mhc-iedb",
    default=False,
    action="store_true",
    help="Use IEDB's web API for MHC binding")

mhc_predictor_arg_group.add_argument("--mhc-cons",
    default=False,
    action="store_true",
    help="Use local NetMHCcons binding predictor")

mhc_predictor_arg_group.add_argument("--mhc-cons",
    default=False,
    action="store_true",
    help="Use local NetMHCcons binding predictor")

mhc_options_arg_group = parser.add_argument_group(
    title="MHC Prediction Options",
    description="MHC Binding Prediction Options")

mhc_options_arg_group.add_argument("--mhc-epitope-lengths",
    default=[8, 9, 10, 11],
    type=parse_int_list,
    help="Lengths of epitopes to consider for MHC binding prediction")
