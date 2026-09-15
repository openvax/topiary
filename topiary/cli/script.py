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

r"""
Script to generate epitope predictions from somatic cancer variants
and (optionally) tumor RNA-seq data.

Example usage:
    topiary \
        --mhc-predictor netmhcpan \
        --mhc-alleles-file HLA.txt \
        --vcf somatic.vcf \
        --gene-expression genes.fpkm_tracking \
        --transcript-expression isoforms.fpkm_tracking \
        --ic50-cutoff 500 \
        --percentile-cutoff 2 \
        --output-csv results.csv
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
        # Each of these names something the caller can act on: a
        # missing input file, a bad argument, predictor setup still to
        # finish, or a cache that cannot answer for this peptide.
        #
        # All four are narrow deliberately. Bare KeyError would swallow
        # an unrelated dict-lookup bug, and bare RuntimeError is worse:
        # NotImplementedError and RecursionError subclass it, so an
        # unimplemented abstract method would print as a clean user
        # error. Those must keep reaching the user as tracebacks.
        #
        # str(e) suits all four: CachedPredictorCoverageError defines
        # __str__, so its message arrives without KeyError's repr
        # quoting and no per-type formatting is needed here. Removing
        # that __str__ would bring the quoting back (#296, #302, #304).
        message = str(e) or type(e).__name__
        arg_parser.error(message)
    write_outputs(df, args)
    print("Total count: %d" % len(df))
    return 0
