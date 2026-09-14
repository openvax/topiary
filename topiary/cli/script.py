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

from ..cached import CachedPredictorCoverageError
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
        OSError, ValueError, RuntimeError, CachedPredictorCoverageError,
    ) as e:
        # Every failure caught here is something the user can act on, so
        # each gets a one-line message rather than a traceback:
        #
        # - OSError: a missing or unreadable input file.
        # - ValueError: what the rest of this package raises for bad
        #   arguments and malformed input.
        # - RuntimeError: predictor setup the user has to finish -- e.g.
        #   mhcflurry installed with no model release fetched, whose
        #   message is "Run `mhcflurry-downloads fetch`". Actionable
        #   advice that was reaching them as a stack trace.
        # - CachedPredictorCoverageError: missed peptides with no
        #   fallback, or a coverage gap a flank or genotype mismatch
        #   leaves in a protein scan (#296, #302, #304).
        #
        # Deliberately not bare KeyError: an unrelated dict-lookup bug
        # elsewhere in this call graph should still surface as a
        # traceback rather than be reported as a clean CLI error.
        #
        # str(e) suits all four. CachedPredictorCoverageError defines
        # __str__ so its message arrives unquoted, rather than the CLI
        # unwrapping e.args[0] itself -- that unwrap needed a type guard
        # to avoid printing OSError's errno in place of its message, and
        # the guard was dropped once already while an adjacent comment
        # was edited (5.56.0, fixed in 5.56.1). There is no longer a
        # guard to drop.
        arg_parser.error(str(e))
    write_outputs(df, args)
    print("Total count: %d" % len(df))
    return 0
