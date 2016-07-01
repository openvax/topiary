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

from argparse import ArgumentParser

from .variants import add_variant_args
from .mhc import add_mhc_args
from .filtering import add_filter_args
from .rna import (
    add_rna_args,

)
from .sequence import add_sequence_args
from .errors import add_error_args
from .outputs import add_output_args

def make_arg_parser(**kwargs):
    arg_parser = ArgumentParser(**kwargs)
    add_rna_args(arg_parser)
    add_mhc_args(arg_parser)
    add_variant_args(arg_parser)
    add_filter_args(arg_parser)
    add_sequence_args(arg_parser)
    add_error_args(arg_parser)
    add_output_args(arg_parser)
    return arg_parser
