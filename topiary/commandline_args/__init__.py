# Copyright (c) 2015-2016. Mount Sinai School of Medicine
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

from __future__ import print_function, division, absolute_import

from argparse import ArgumentParser
from mhctools.cli import (
    add_mhc_args,
    mhc_alleles_from_args,
    mhc_binding_predictor_from_args,
)
from varcode.cli import add_variant_args, variant_collection_from_args

from .filtering import add_filter_args
from .rna import (
    add_rna_args,
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args,
)
from .sequence import add_sequence_args
from .errors import add_error_args
from .outputs import add_output_args, write_outputs

arg_parser = ArgumentParser()

add_rna_args(arg_parser)
add_mhc_args(arg_parser)
add_variant_args(arg_parser)
add_filter_args(arg_parser)
add_sequence_args(arg_parser)
add_error_args(arg_parser)
add_output_args(arg_parser)

__all__ = [
    "arg_parser",
    "add_rna_args",
    "rna_gene_expression_dict_from_args",
    "rna_transcript_expression_dict_from_args",
    "add_variant_args",
    "variant_collection_from_args",
    "add_filter_args",
    "add_mhc_args",
    "mhc_alleles_from_args",
    "mhc_binding_predictor_from_args",
    "add_sequence_args",
    "add_error_args",
    "add_output_args",
    "write_outputs",
]
