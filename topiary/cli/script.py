# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import sys

import argcomplete

from ..cached import CachedPredictorCoverageError, PredictorSetupError
from .args import arg_parser, predict_epitopes_from_args

from .outputs import write_outputs


def parse_args(args_list=None):
    if args_list is None:
        args_list = sys.argv[1:]
    argcomplete.autocomplete(arg_parser)
    return arg_parser.parse_args(args_list)


def main(args_list=None):
    """
    Script entry-point to predict neo-epitopes from genomic variants using
    Topiary.
    """
    args = parse_args(args_list)
    try:
        df = predict_epitopes_from_args(args)
    except (
        OSError, ValueError, PredictorSetupError,
        CachedPredictorCoverageError,
    ) as e:
        # Each of these names something the user can act on, so each
        # gets a one-line message instead of a traceback: a missing or
        # unreadable input file, a bad argument or malformed input,
        # predictor setup still to finish, or a cache that cannot answer
        # for a peptide or occurrence (#296, #302, #304).
        #
        # All four are narrow on purpose. Bare KeyError would swallow an
        # unrelated dict-lookup bug; bare RuntimeError is worse, since
        # NotImplementedError and RecursionError subclass it, so an
        # abstract method left unimplemented would print as a clean user
        # error. Those must keep reaching the user as tracebacks.
        #
        # CachedPredictorCoverageError defines __str__, so its message
        # arrives unquoted and this needs no per-type formatting.
        message = str(e) or type(e).__name__
        arg_parser.error(message)
    write_outputs(df, args)
    print("Total count: %d" % len(df))
    return 0
